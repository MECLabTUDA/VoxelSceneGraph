from typing import Iterator, Sequence

import torch
from scene_graph_api.tensor_structures import AbstractMask, AbstractMaskList, BinaryMaskList, PolygonInstance, \
    PolygonList

AbstractMask = AbstractMask
AbstractMaskList = AbstractMaskList
BinaryMaskList = BinaryMaskList
PolygonInstance = PolygonInstance
PolygonList = PolygonList

_SIZE_T = tuple[int, ...]
_GET_ITEM_T = int | list[int] | list[bool] | torch.Tensor


class MaskListView(AbstractMaskList):
    def __init__(self, mask_list: AbstractMaskList, indexes: torch.LongTensor | None = None):
        self._mask_list = mask_list
        self._indexes = indexes if indexes is not None else \
            torch.tensor(list(range(len(mask_list))), dtype=torch.int64)

    def flip(self, dim: AbstractMaskList.FlipDim) -> "AbstractMaskList":
        return MaskListView(self._mask_list.flip(dim), self._indexes)

    def crop(self, box: Sequence) -> "AbstractMaskList":
        return MaskListView(self._mask_list.crop(box), self._indexes)

    def resize(self, size: _SIZE_T) -> "AbstractMaskList":
        return MaskListView(self._mask_list.resize(size), self._indexes)

    def to(self, *args, **kwargs) -> "AbstractMaskList":
        mask_list = self._mask_list.to(*args, **kwargs)
        return MaskListView(mask_list, self._indexes.to(device=mask_list.device()))

    def get_mask_tensor(self) -> torch.Tensor:
        return self._mask_list[self._indexes].get_mask_tensor()

    def __len__(self) -> int:
        return len(self._indexes)

    def __getitem__(self, index: _GET_ITEM_T) -> "AbstractMaskList":
        return MaskListView(self._mask_list, self._indexes[index])

    def __iter__(self) -> Iterator["AbstractMaskList"]:
        for idx in self._indexes:
            yield self._mask_list[idx]

    def __repr__(self) -> str:
        return f"MaskListView(mask_list={repr(self._mask_list)},\nindexes={self._indexes})"

    def device(self):
        raise self._mask_list.device()
