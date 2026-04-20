# Two-tier DER++ variant with cross-sampling across both buffers for both loss terms.
#
# Extends derpp_twotier_consolidation with independent mixing ratios:
#
#   MSE/distillation batch:
#       mse_ltm_ratio  fraction from LTM  (diverse, all tasks)
#       1-mse_ltm_ratio fraction from STM  (recent)
#
#   CE replay batch:
#       ce_stm_ratio   fraction from STM  (recent)
#       1-ce_stm_ratio fraction from LTM  (diverse)
#
# Both loss terms always draw from both buffers — no stm_only mode.
# LTM is populated exclusively via periodic consolidation (same as
# derpp_twotier_consolidation). Everything else is identical.

import torch
from torch.nn import functional as F

from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
from utils.fifo_buffer import FifoBuffer
from utils.consolidation import get_consolidation_fn


class DerppTwoTierCrossSampling(ContinualModel):
    """Two-tier DER++ with cross-sampling: both MSE and CE draw from both buffers.

    Total memory budget = --buffer_size.
    LTM capacity        = buffer_size - stm_size  (reservoir, consolidation-only writes).
    STM capacity        = stm_size                (FIFO, receives every batch).
    """

    NAME = 'derpp_twotier_crosssampling'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        add_rehearsal_args(parser)  # provides --buffer_size, --minibatch_size
        parser.add_argument('--stm_size', type=int, required=True,
                            help='Capacity of the FIFO short-term memory buffer. '
                                 'Must be strictly less than --buffer_size. '
                                 'LTM capacity = buffer_size - stm_size.')
        parser.add_argument('--alpha', type=float, required=True,
                            help='Weight for the logit distillation (MSE) loss.')
        parser.add_argument('--beta', type=float, required=True,
                            help='Weight for the CE replay loss.')
        parser.add_argument('--mse_ltm_ratio', type=float, default=0.8,
                            help='Fraction of the MSE/distillation minibatch drawn from LTM. '
                                 'Remainder drawn from STM. Must be in (0, 1). Default: 0.8.')
        parser.add_argument('--ce_stm_ratio', type=float, default=0.5,
                            help='Fraction of the CE replay minibatch drawn from STM. '
                                 'Remainder drawn from LTM. Must be in (0, 1). Default: 0.5.')
        parser.add_argument('--strategy', type=str, default='hybrid',
                            choices=['random', 'diversity', 'loss', 'hybrid'],
                            help='Consolidation strategy for STM -> LTM transfer.')
        parser.add_argument('--consolidation_freq', type=int, default=100,
                            help='Consolidate STM -> LTM every N training steps.')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset=dataset)

        assert args.stm_size < args.buffer_size, (
            f'--stm_size ({args.stm_size}) must be strictly less than '
            f'--buffer_size ({args.buffer_size}).')
        assert 0.0 <= args.mse_ltm_ratio <= 1.0, (
            f'--mse_ltm_ratio must be in [0, 1], got {args.mse_ltm_ratio}.')
        assert 0.0 <= args.ce_stm_ratio <= 1.0, (
            f'--ce_stm_ratio must be in [0, 1], got {args.ce_stm_ratio}.')

        ltm_size = args.buffer_size - args.stm_size

        # LTM: reservoir buffer — populated only via consolidation
        self.ltm = Buffer(ltm_size)
        # STM: FIFO buffer — receives every incoming batch
        self.stm = FifoBuffer(capacity=args.stm_size)
        self.consolidate_fn = get_consolidation_fn(args.strategy)
        self.train_step = 0

    # ------------------------------------------------------------------
    # Replay batch helpers
    # ------------------------------------------------------------------

    def _mse_replay_batch(self, minibatch_size):
        """Draw MSE/distillation batch from both buffers.

        mse_ltm_ratio fraction from LTM (diverse), remainder from STM (recent).
        Returns (inputs, logits).
        """
        n_ltm = max(1, round(minibatch_size * self.args.mse_ltm_ratio))
        n_stm = max(1, minibatch_size - n_ltm)

        ltm_inputs, _, ltm_logits = self.ltm.get_data(
            n_ltm, transform=self.transform, device=self.device)
        stm_inputs, _, stm_logits = self.stm.get_data(
            n_stm, transform=self.transform, device=self.device)

        buf_inputs = torch.cat([ltm_inputs, stm_inputs], dim=0)
        buf_logits = torch.cat([ltm_logits, stm_logits], dim=0)
        return buf_inputs, buf_logits

    def _ce_replay_batch(self, minibatch_size):
        """Draw CE replay batch from both buffers.

        ce_stm_ratio fraction from STM (recent), remainder from LTM (diverse).
        Returns (inputs, labels).
        """
        n_stm = max(1, round(minibatch_size * self.args.ce_stm_ratio))
        n_ltm = max(1, minibatch_size - n_stm)

        stm_inputs, stm_labels, _ = self.stm.get_data(
            n_stm, transform=self.transform, device=self.device)
        ltm_inputs, ltm_labels, _ = self.ltm.get_data(
            n_ltm, transform=self.transform, device=self.device)

        buf_inputs = torch.cat([stm_inputs, ltm_inputs], dim=0)
        buf_labels = torch.cat([stm_labels, ltm_labels], dim=0)
        return buf_inputs, buf_labels

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        self.opt.zero_grad()

        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)

        if not self.stm.is_empty() and not self.ltm.is_empty():
            # --- MSE/distillation loss: cross-sample from LTM (majority) + STM ---
            buf_inputs, buf_logits = self._mse_replay_batch(self.args.minibatch_size)
            buf_outputs = self.net(buf_inputs)
            loss += self.args.alpha * F.mse_loss(buf_outputs, buf_logits)

            # --- CE replay loss: cross-sample from STM (majority) + LTM ---
            buf_inputs, buf_labels = self._ce_replay_batch(self.args.minibatch_size)
            buf_outputs = self.net(buf_inputs)
            loss += self.args.beta * self.loss(buf_outputs, buf_labels)

        loss.backward()
        self.opt.step()

        # STM receives every incoming batch directly
        self.stm.add_data(examples=not_aug_inputs,
                          labels=labels,
                          logits=outputs.data)

        # LTM is never written directly — only via consolidation
        self.train_step += 1
        if (self.train_step % self.args.consolidation_freq == 0
                and not self.stm.is_empty()):
            self._consolidate()

        return loss.item()

    # ------------------------------------------------------------------
    # Consolidation (identical to derpp_twotier_consolidation)
    # ------------------------------------------------------------------

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
