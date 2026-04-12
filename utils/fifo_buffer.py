"""
Simple FIFO  buffer for use as a short-term memory in two-tier replay.

Oldest samples are overwritten first. Sampling is uniform random over the
filled portion, matching the get_data interface of utils/buffer.py::Buffer.
"""

import numpy as np
import torch
import torch.nn as nn

from utils.augmentations import apply_transform


class FifoBuffer:
    """
    Fixed-capacity FIFO memory buffer.

    Writes cycle through slots 0..capacity-1; when full, the oldest slot is
    overwritten. Stores examples, labels, and logits (same attribute names as
    Buffer so call-sites look identical).
    """

    def __init__(self, capacity: int, device: str = "cpu"):
        self.capacity = capacity
        self.device = device
        self.write_ptr = 0        # next slot to write
        self.num_seen_examples = 0  # total examples ever added (for is_empty)
        # tensors are allocated on first add_data call
        self._initialized = False

    def _init_tensors(self, examples: torch.Tensor, labels: torch.Tensor,
                      logits: torch.Tensor) -> None:
        self.examples = torch.zeros(
            (self.capacity, *examples.shape[1:]), dtype=torch.float32, device=self.device)
        if labels is not None:
            self.labels = torch.zeros(
                (self.capacity,), dtype=torch.int64, device=self.device)
        if logits is not None:
            self.logits = torch.zeros(
                (self.capacity, *logits.shape[1:]), dtype=torch.float32, device=self.device)
        self._initialized = True

    def _filled_size(self) -> int:
        return min(self.num_seen_examples, self.capacity)


    def is_empty(self) -> bool:
        return self.num_seen_examples == 0

    def __len__(self) -> int:
        return self._filled_size()

    def add_data(self, examples: torch.Tensor, labels: torch.Tensor = None,
                 logits: torch.Tensor = None) -> None:
        """
        Insert a batch into the FIFO buffer.
        Each sample overwrites the slot at write_ptr, then write_ptr advances.
        """
        if not self._initialized:
            self._init_tensors(examples, labels, logits)

        for i in range(examples.shape[0]):
            idx = self.write_ptr % self.capacity
            self.examples[idx] = examples[i].to(self.device)
            if labels is not None and hasattr(self, 'labels'):
                self.labels[idx] = labels[i].to(self.device)
            if logits is not None and hasattr(self, 'logits'):
                self.logits[idx] = logits[i].to(self.device)
            self.write_ptr += 1
            self.num_seen_examples += 1

    def get_data(self, size: int, transform: nn.Module = None,
                 device: str = None):
        """
        Uniformly sample `size` items from the filled portion of the buffer.

        Returns a tuple (examples, labels, logits) for whichever attributes
        have been initialised — same tuple order as Buffer.get_data.
        """
        target_device = self.device if device is None else device
        filled = self._filled_size()
        size = min(size, filled)

        choice = np.random.choice(filled, size=size, replace=False)

        if transform is None:
            def transform(x): return x

        ret = (apply_transform(self.examples[choice], transform=transform).to(target_device),)
        if hasattr(self, 'labels'):
            ret += (self.labels[choice].to(target_device),)
        if hasattr(self, 'logits'):
            ret += (self.logits[choice].to(target_device),)
        return ret
