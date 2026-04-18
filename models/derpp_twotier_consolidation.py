# Two-tier DER++ variant with STM → LTM consolidation.
#
# STM (short-term memory): FIFO buffer  → receives every incoming batch
# LTM (long-term memory):  reservoir    → populated ONLY via periodic
#                                          consolidation from STM, never
#                                          via direct per-step writes
#
# CE replay draws from STM; logit/MSE distillation draws from LTM.
# Consolidation scores STM samples every --consolidation_freq steps and
# pushes them into LTM in priority order (--strategy).

import torch
from torch.nn import functional as F

from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
from utils.fifo_buffer import FifoBuffer
from utils.consolidation import get_consolidation_fn


class DerppTwoTierConsolidation(ContinualModel):
    """Two-tier DER++ where LTM is fed exclusively via STM consolidation."""

    NAME = 'derpp_twotier_consolidation'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        add_rehearsal_args(parser)  # provides --buffer_size, --minibatch_size
        parser.add_argument('--stm_size', type=int, required=True,
                            help='Capacity of the FIFO short-term memory buffer.')
        parser.add_argument('--alpha', type=float, required=True,
                            help='Weight for the logit distillation (MSE) loss.')
        parser.add_argument('--beta', type=float, required=True,
                            help='Weight for the CE replay loss.')
        parser.add_argument('--ce_replay_mode', type=str, default='mixed',
                            choices=['stm_only', 'mixed'],
                            help='Source for CE replay batch. '
                                 'stm_only: draw entirely from STM. '
                                 'mixed: draw ce_stm_ratio from STM, remainder from LTM.')
        parser.add_argument('--ce_stm_ratio', type=float, default=0.5,
                            help='Fraction of the CE replay minibatch drawn from STM '
                                 'when ce_replay_mode=mixed. Must be in (0, 1). '
                                 'Default: 0.5.')
        parser.add_argument('--strategy', type=str, default='random',
                            choices=['random', 'diversity', 'loss', 'hybrid'],
                            help='Consolidation strategy for STM -> LTM transfer.')
        parser.add_argument('--consolidation_freq', type=int, default=100,
                            help='Consolidate STM -> LTM every N training steps.')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset=dataset)

        if args.ce_replay_mode == 'mixed':
            assert 0.0 < args.ce_stm_ratio < 1.0, (
                f'--ce_stm_ratio must be in (0, 1), got {args.ce_stm_ratio}.')

        assert args.stm_size < args.buffer_size, (
            f'--stm_size ({args.stm_size}) must be strictly less than '
            f'--buffer_size ({args.buffer_size}).')

        ltm_size = args.buffer_size - args.stm_size

        # LTM: reservoir buffer — populated only via consolidation
        self.ltm = Buffer(ltm_size)
        # STM: FIFO buffer — receives every incoming batch
        self.stm = FifoBuffer(capacity=self.args.stm_size)
        self.consolidate_fn = get_consolidation_fn(self.args.strategy)
        self.train_step = 0

    def _ce_replay_batch(self, minibatch_size):
        """Return (inputs, labels) for the CE replay loss.

        stm_only: all samples from STM.
        mixed:    ceil(ratio * size) from STM, remainder from LTM,
                  concatenated into one batch.
        """
        if self.args.ce_replay_mode == 'stm_only':
            buf_inputs, buf_labels, _ = self.stm.get_data(
                minibatch_size, transform=self.transform, device=self.device)
            return buf_inputs, buf_labels

        # mixed mode
        n_stm = max(1, round(minibatch_size * self.args.ce_stm_ratio))
        n_ltm = max(1, minibatch_size - n_stm)

        stm_inputs, stm_labels, _ = self.stm.get_data(
            n_stm, transform=self.transform, device=self.device)
        ltm_inputs, ltm_labels, _ = self.ltm.get_data(
            n_ltm, transform=self.transform, device=self.device)

        buf_inputs = torch.cat([stm_inputs, ltm_inputs], dim=0)
        buf_labels = torch.cat([stm_labels, ltm_labels], dim=0)
        return buf_inputs, buf_labels

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        self.opt.zero_grad()

        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)

        if not self.stm.is_empty() and not self.ltm.is_empty():
            # --- logit distillation loss: replay from LTM ---
            buf_inputs, _, buf_logits = self.ltm.get_data(
                self.args.minibatch_size, transform=self.transform, device=self.device)
            buf_outputs = self.net(buf_inputs)
            loss += self.args.alpha * F.mse_loss(buf_outputs, buf_logits)

            # --- CE replay loss: stm_only or mixed ---
            buf_inputs, buf_labels = self._ce_replay_batch(self.args.minibatch_size)
            buf_outputs = self.net(buf_inputs)
            loss += self.args.beta * self.loss(buf_outputs, buf_labels)

        loss.backward()
        self.opt.step()

        # STM receives every incoming batch directly
        self.stm.add_data(examples=not_aug_inputs,
                          labels=labels,
                          logits=outputs.data)

        # LTM is never written directly — only via consolidation below
        self.train_step += 1
        if (self.train_step % self.args.consolidation_freq == 0
                and not self.stm.is_empty()):
            self._consolidate()

        return loss.item()

    def _consolidate(self):
        """Score all STM samples and push them into LTM in priority order."""
        stm_ex, stm_lb, stm_lo = self.stm.get_filled_data()
        if stm_ex is None:
            return

        order = self.consolidate_fn(self.stm, self.ltm, self.net, self.device).cpu()

        self.ltm.add_data(
            examples=stm_ex[order],
            labels=stm_lb[order] if stm_lb is not None else None,
            logits=stm_lo[order] if stm_lo is not None else None)
