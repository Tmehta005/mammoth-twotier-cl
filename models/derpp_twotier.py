# Two-tier DER++ variant.
#
# STM (short-term memory): FIFO buffer  → used for CE replay loss
# LTM (long-term memory):  reservoir    → used for logit/MSE distillation loss
#
# Training logic is otherwise identical to models/derpp.py.

from torch.nn import functional as F

from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
from utils.fifo_buffer import FifoBuffer


class DerppTwoTier(ContinualModel):
    """DER++ with a two-tier replay buffer (FIFO STM + reservoir LTM)."""

    NAME = 'derpp_twotier'
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
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset=dataset)

        # LTM: reservoir buffer (--buffer_size slots)
        self.ltm = Buffer(self.args.buffer_size)
        # STM: FIFO buffer (--stm_size slots)
        self.stm = FifoBuffer(capacity=self.args.stm_size)

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

            # --- CE replay loss: replay from STM ---
            buf_inputs, buf_labels, _ = self.stm.get_data(
                self.args.minibatch_size, transform=self.transform, device=self.device)
            buf_outputs = self.net(buf_inputs)
            loss += self.args.beta * self.loss(buf_outputs, buf_labels)

        loss.backward()
        self.opt.step()

        # Insert current batch into both buffers
        self.ltm.add_data(examples=not_aug_inputs,
                          labels=labels,
                          logits=outputs.data)
        self.stm.add_data(examples=not_aug_inputs,
                          labels=labels,
                          logits=outputs.data)

        return loss.item()
