from typing import Iterator

import torch


class BufferList(torch.nn.Module):
    """
    Similar to torch.nn.ParameterList but for buffers
    i.e. tensors that are not Parameters but still part of the model's state.
    """

    def __init__(self, buffers: list[torch.Tensor] | None = None):
        super().__init__()
        if buffers is not None:
            self.extend(buffers)

    def extend(self, buffers: list[torch.Tensor]) -> "BufferList":
        offset = len(self)
        for i, buffer in enumerate(buffers):
            self.register_buffer(str(offset + i), buffer)
        return self

    def __len__(self) -> int:
        return len(self._buffers)

    def __iter__(self) -> Iterator[torch.Tensor]:
        return iter(self._buffers.values())

    def __getitem__(self, item):
        return self._buffers[str(item)]
