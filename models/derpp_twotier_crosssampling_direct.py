# Two-tier DER++ with cross-sampling, no consolidation.
#
# Both buffers receive every incoming batch directly:
#   STM: FIFO replacement
#   LTM: reservoir sampling
#
# Both loss terms cross-sample from both buffers:
#   MSE/distillation: mse_ltm_ratio from LTM, remainder from STM
#   CE replay:        ce_stm_ratio from STM, remainder from LTM
#
# No consolidation, no scoring, no cold-start.
# Isolates the effect of cross-sampling ratios alone.

import torch
from torch.nn import functional as F

from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
from utils.fifo_buffer import FifoBuffer


class DerppTwoTierCrossSamplingDirect(ContinualModel):
    """Two-tier DER++ with cross-sampling and direct buffer writes (no consolidation).

    Total memory budget = --buffer_size.
    LTM capacity        = buffer_size - stm_size  (reservoir, written every step).
    STM capacity        = stm_size                (FIFO, written every step).
    """

    NAME = 'derpp_twotier_crosssampling_direct'
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
                                 'Remainder drawn from STM. Must be in [0, 1]. Default: 0.8.')
        parser.add_argument('--ce_stm_ratio', type=float, default=0.5,
                            help='Fraction of the CE replay minibatch drawn from STM. '
                                 'Remainder drawn from LTM. Must be in [0, 1]. Default: 0.5.')
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

        # LTM: reservoir buffer — written every step
        self.ltm = Buffer(ltm_size)
        # STM: FIFO buffer — written every step
        self.stm = FifoBuffer(capacity=args.stm_size)

    # ------------------------------------------------------------------
    # Replay batch helpers
    # ------------------------------------------------------------------

    def _mse_replay_batch(self, minibatch_size):
        """Draw MSE/distillation batch: mse_ltm_ratio from LTM, rest from STM."""
        n_ltm = max(1, round(minibatch_size * self.args.mse_ltm_ratio))
        n_stm = max(1, minibatch_size - n_ltm)

        ltm_inputs, _, ltm_logits = self.ltm.get_data(
            n_ltm, transform=self.transform, device=self.device)
        stm_inputs, _, stm_logits = self.stm.get_data(
            n_stm, transform=self.transform, device=self.device)

        return torch.cat([ltm_inputs, stm_inputs], dim=0), \
               torch.cat([ltm_logits, stm_logits], dim=0)

    def _ce_replay_batch(self, minibatch_size):
        """Draw CE replay batch: ce_stm_ratio from STM, rest from LTM."""
        n_stm = max(1, round(minibatch_size * self.args.ce_stm_ratio))
        n_ltm = max(1, minibatch_size - n_stm)

        stm_inputs, stm_labels, _ = self.stm.get_data(
            n_stm, transform=self.transform, device=self.device)
        ltm_inputs, ltm_labels, _ = self.ltm.get_data(
            n_ltm, transform=self.transform, device=self.device)

        return torch.cat([stm_inputs, ltm_inputs], dim=0), \
               torch.cat([stm_labels, ltm_labels], dim=0)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        self.opt.zero_grad()

        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)

        if not self.stm.is_empty() and not self.ltm.is_empty():
            # MSE/distillation loss: cross-sample from LTM (majority) + STM
            buf_inputs, buf_logits = self._mse_replay_batch(self.args.minibatch_size)
            buf_outputs = self.net(buf_inputs)
            loss += self.args.alpha * F.mse_loss(buf_outputs, buf_logits)

            # CE replay loss: cross-sample from STM (majority) + LTM
            buf_inputs, buf_labels = self._ce_replay_batch(self.args.minibatch_size)
            buf_outputs = self.net(buf_inputs)
            loss += self.args.beta * self.loss(buf_outputs, buf_labels)

        loss.backward()
        self.opt.step()

        # Both buffers receive every incoming batch directly
        self.ltm.add_data(examples=not_aug_inputs,
                          labels=labels,
                          logits=outputs.data)
        self.stm.add_data(examples=not_aug_inputs,
                          labels=labels,
                          logits=outputs.data)

        return loss.item()
