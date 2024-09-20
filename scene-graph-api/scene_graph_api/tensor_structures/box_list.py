"""
Copyright 2023 Antoine Sanner, Technical University of Darmstadt, Darmstadt, Germany

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import warnings
from enum import Enum
from os import PathLike
from typing import Any, Hashable
from typing import Iterable

import numpy as np
import torch
from typing_extensions import Self

from .box_list_fields import AnnotationField, PredictionField
from .box_list_ops import BoxListOps
from ..utils.indexing import FlipDim

_SIZE_T = tuple[int, ...]


class _Mode(Enum):
    """Bbox coordinate formats. Work with any number of dimensions."""
    zyxzyx = "zyxzyx"
    zyxdhw = "zyxdhw"


class BoxList:
    """
    This class represents a set of bounding boxes.
    The bounding boxes are represented as a Nx(2*n_dim) Tensor.
    In order to uniquely determine the bounding boxes with respect
    to an image, we also store the corresponding image dimensions.
    They can contain extra information that is specific to each bounding box, such as labels.

    Note: a BoxList is assumed to have at least the LABELS field and either/both LABELMAP and MASKS fields.
    Note: the SEGMENTATION field can only be present if the LABELMAP field is also present.
    """

    Mode = _Mode
    AnnotationField = AnnotationField
    PredictionField = PredictionField
    FlipDim = FlipDim

    def __init__(self, boxes: torch.Tensor | np.ndarray, image_size: _SIZE_T, mode: _Mode = _Mode.zyxzyx):
        super().__init__()

        device = boxes.device if isinstance(boxes, torch.Tensor) else torch.device("cpu")
        boxes = torch.as_tensor(boxes, dtype=torch.float32, device=device)

        n_dim = len(image_size)
        if boxes.dim() != 2:
            raise ValueError(f"bbox should have 2 dimensions, got {boxes.ndimension()}")
        if boxes.size(-1) != 2 * n_dim:
            raise ValueError(f"last dimension of bbox should have a size of {2 * n_dim}, got {boxes.size(-1)}")

        self.boxes: torch.Tensor = boxes
        self.size = image_size  # xyz ordered
        self.n_dim = n_dim
        self.mode = mode
        self.extra_fields = {}
        self.fields_indexing_power: dict[Hashable, int] = {}

    def __getitem__(self, item: slice | list[int] | list[bool] | torch.Tensor) -> Self:
        """
        Like a tensor, it accepts an index, a list/tensor of ints or a list/tensor of bools.
        IMPORTANT: the main goal is perhaps that when sampling boxes, we also do the proper tensor sampling on any
        labels, attributes... tensors stored as a field here...
        Note: fields with indexing power 0 are NEVER changed. Some models rely on it.
              For the more advanced behaviour, please refer to BoxListOps.indexing_with_segmentation_update.
        """
        bbox = type(self)(self.boxes[item], self.size, self.mode)
        for k, v in self.extra_fields.items():
            indexing_power = self.fields_indexing_power[k]
            # Note: (item,) * indexing_power does not work with tensors...
            #       E.g. we have to do "v[item]" with power 1, "v[item][:, item]" with power 2
            for dim in range(indexing_power):
                v = v[(slice(None),) * dim + (item,)]
            bbox.add_field(k, v, indexing_power=indexing_power)

        # Check if we have these special PredictionFields for relations
        if bbox.has_fields(
                PredictionField.REL_PAIR_IDXS,
                PredictionField.PRED_REL_LABELS,
                PredictionField.PRED_REL_CLS_SCORES
        ):
            bbox._reindex_predicted_relations(item, len(self))

        return bbox

    # noinspection PyProtectedMember
    def __getattr__(self, item: str):
        """Shorthand for getting fields."""
        if item.lower() in AnnotationField._value2member_map_:
            return self.get_field(AnnotationField(item.lower()), True)
        if item.lower() in PredictionField._value2member_map_:
            return self.get_field(PredictionField(item.lower()), True)
        return super().__getattribute__(item)

    # noinspection PyProtectedMember
    def __setattr__(self, key: str, value):
        """Shorthand for setting fields."""
        if key.lower() in AnnotationField._value2member_map_:
            key = AnnotationField(key.lower())
            self.add_field(key, value)
            return
        if key.lower() in PredictionField._value2member_map_:
            key = PredictionField(key.lower())
            self.add_field(key, value)
            return
        return super().__setattr__(key, value)

    def __len__(self) -> int:
        return self.boxes.shape[0]

    def __repr__(self) -> str:
        fields_repr = ", ".join(sorted(list(map(repr, self.extra_fields.keys()))))
        return f"{self.__class__.__name__}(num_boxes={len(self)}, image_size={self.size}, mode={self.mode.value}," \
               f"fields=[{fields_repr}])"

    def add_field(self, field: Hashable, field_data: Any, indexing_power: int | None = None):
        """
        Note: since most fields are now part of an enum, we can already infer the indexing power and will be ignored.
              indexing_power will only be used if the field is not known.
        :param field: the field key
        :param field_data: the data to store
        :param indexing_power: this class is specifically used to easily sample boxes and
                               their attributes (fields) through __getitem__.
                               However, all fields should not respond equally to the indexing:
                               - a labelmap should not be changed: indexing power of 1
                               - a label list matches the list of boxes: indexing power of 1
                               - a relation matrix contains a list of boxes along its 2 dimensions: indexing power of 2
                               - etc...
        """
        if isinstance(field, AnnotationField | PredictionField):
            registered_indexing_power = field.indexing_power()
            if indexing_power is not None and registered_indexing_power != indexing_power:
                warnings.warn(f"Field {field} must be used with indexing power {registered_indexing_power}, "
                              f"but was added with power {indexing_power}.")
            indexing_power = registered_indexing_power

        if indexing_power is None:
            raise ValueError(f"None indexing power used with an unknown field "
                             f"(neither AnnotationField nor PredictionField).")
        if indexing_power < 0:
            raise ValueError(f"Invalid indexing power (negative values are not allowed) ({indexing_power})")
        self.extra_fields[field] = field_data
        self.fields_indexing_power[field] = indexing_power

    def del_field(self, field: Hashable) -> bool:
        """
        :param field: the field key
        :return: success.
        """
        if not self.has_field(field):
            return False
        del self.extra_fields[field]
        del self.fields_indexing_power[field]
        return True

    def get_field(self, field: Hashable, raise_missing: bool = False) -> Any | None:
        """
        Return the specified field if present else None.
        Raise a RuntimeError if raise_missing and the field is missing.
        """
        if raise_missing and not self.has_field(field):
            raise KeyError(f"BoxList is missing field {field}.")
        return self.extra_fields.get(field)

    def has_field(self, field: Hashable) -> bool:
        """Return whether this BoxList has the specified field."""
        return field in self.extra_fields

    def has_fields(self, *fields: Hashable) -> bool:
        """Return whether this BoxList has the specified fields."""
        return all(field in self.extra_fields for field in fields)

    def fields(self) -> list:
        """Return a list with all field names for the BoxList."""
        return list(self.extra_fields.keys())

    def save(self, path: PathLike | str):
        """
        Save the BoxList at the designated location.
        Note: mostly an alias for `torch.save` but may also allow for some customization.
        """
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: PathLike | str) -> Self:
        """
        Load the BoxList from the designated location.
        Note: allows for custom processing of loaded objects,
              such as casting to a super-class (e.g. from scene_graph_prediction).
        """
        return cls.from_state_dict(torch.load(path, weights_only=True))

    def state_dict(self) -> dict[str, Any]:
        """Return the state dict for this BoxList."""
        # Map enum values to their string values to avoid explicitly saving classes and their module path
        extra_fields, fields_indexing_power = {}, {}
        for field in self.extra_fields.keys():
            if isinstance(field, AnnotationField | PredictionField):
                key = field.value
            else:
                key = field
            extra_fields[key] = self.extra_fields[field]
            fields_indexing_power[key] = self.fields_indexing_power[field]

        return {
            "boxes": self.boxes,
            "size": self.size,
            "mode": self.mode.value,
            "n_dim": self.n_dim,
            "extra_fields": extra_fields,
            "fields_indexing_power": fields_indexing_power
        }

    # noinspection PyProtectedMember
    @classmethod
    def from_state_dict(cls, state_dict: dict) -> Self:
        """Return the state dict for this BoxList."""
        try:
            # Convert all string values to their corresponding Enum value
            # noinspection PyTypeChecker
            boxlist = cls(state_dict["boxes"], state_dict["size"], _Mode._value2member_map_[state_dict["mode"]])
            boxlist.n_dim = state_dict["n_dim"]
            extra_fields, fields_indexing_power = {}, {}
            for key in state_dict["extra_fields"].keys():
                if key in AnnotationField._value2member_map_:
                    field = AnnotationField._value2member_map_[key]
                elif key in PredictionField._value2member_map_:
                    field = PredictionField._value2member_map_[key]
                else:
                    field = key
                extra_fields[field] = state_dict["extra_fields"][key]
                fields_indexing_power[field] = state_dict["fields_indexing_power"][key]
            boxlist.extra_fields = extra_fields
            boxlist.fields_indexing_power = fields_indexing_power
            return boxlist
        except KeyError as e:
            raise ValueError(f"Invalid state dict.") from e

    # Tensor-like methods
    def to(self, *args, **kwargs) -> Self:
        bbox = type(self)(self.boxes.to(*args, **kwargs), self.size, self.mode)
        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
            bbox.add_field(k, v, indexing_power=self.fields_indexing_power[k])
        return bbox

    def copy(self) -> Self:
        """Note: copies are not deep."""
        return type(self)(self.boxes, self.size, self.mode)

    def copy_with_fields(self, fields: Iterable[Hashable], skip_missing: bool = False) -> Self:
        """Note: copies are not deep."""
        bbox = type(self)(self.boxes, self.size, self.mode)
        for field in fields:
            if self.has_field(field):
                bbox.add_field(field, self.get_field(field), indexing_power=self.fields_indexing_power[field])
            elif not skip_missing:
                raise KeyError(f"Field '{field}' not found in {self}")
        return bbox

    def copy_with_all_fields(self) -> Self:
        """Note: copies are not deep."""
        return self[:]

    def convert(self, mode: _Mode) -> Self:
        """Convert the BoxList to a different coordinate mode."""

        if mode == self.mode:
            return self

        # We only have two modes, so don't need to check self.mode
        minimums, maximums = BoxListOps.split_into_zyxzyx(self)
        if mode == _Mode.zyxzyx:
            boxes = torch.cat(minimums + maximums, dim=-1)
        else:
            lengths = tuple(maximums[i] - minimums[i] + 1 for i in range(self.n_dim))
            boxes = torch.cat(minimums + lengths, dim=-1)

        bbox = type(self)(boxes, self.size, mode=mode)
        for k, v in self.extra_fields.items():
            bbox.add_field(k, v, indexing_power=self.fields_indexing_power[k])

        return bbox

    def _reindex_predicted_relations(self, item: slice | list[int] | list[bool] | torch.Tensor, previous_length: int):
        """
        The PRED_REL_CLS_SCORES, PRED_REL_LABELS, and REL_PAIR_IDXS fields have an indexing power of 0,
        but require some processing after a BoxList has been indexed.
        """

        mapping = torch.tensor(list(range(previous_length)), dtype=torch.int64, device=self.boxes.device)[item]

        # First figure out which boxes are kept
        box_indices_kept = torch.unique(mapping)

        # Then figure which relation pairs we can keep (where both subject and object are kept)
        rel_pair_idxs = self.REL_PAIR_IDXS
        # noinspection PyTypeChecker
        valid_rel_pair_idxs = torch.all(sum([rel_pair_idxs == kept_idx for kept_idx in box_indices_kept]), 1)

        # Then we can index all three fields
        rel_pair_idxs = rel_pair_idxs[valid_rel_pair_idxs]
        self.PRED_REL_CLS_SCORES = self.PRED_REL_CLS_SCORES[valid_rel_pair_idxs]
        self.PRED_REL_LABELS = self.PRED_REL_LABELS[valid_rel_pair_idxs]

        # Check if there is no box getting duplicated (i.e. we can use the easy algorithm)
        if mapping.numel() == torch.unique(mapping).numel():
            # Map the indices in rel_pair_idxs
            # 1. Flatten the index tensor
            # 2. Compute the match matrix with the mapping
            # 3. Compute the new index by finding the non-zero match(es)
            #    (we can have multiple matches when indexing with an int tensor)
            self.REL_PAIR_IDXS = torch.nonzero(rel_pair_idxs.view(-1)[:, None] == mapping)[:, 1].view(-1, 2)
            return

        # Else find all combinations for each pair
        match_matrix = (rel_pair_idxs.view(-1)[:, None] == mapping).view(rel_pair_idxs.shape[0], 2, mapping.numel())
        # Batched matmul to compute possible pairs
        combination_matrices = torch.bmm(match_matrix[:, 0, :, None].int(), match_matrix[:, 1, None].int())
        # Update fields
        non_zeros = torch.nonzero(combination_matrices)
        self.REL_PAIR_IDXS = non_zeros[:, 1:]  # Remove the batch index
        self.PRED_REL_CLS_SCORES = self.PRED_REL_CLS_SCORES[non_zeros[:, 0]]  # Use the batch index to resample
        self.PRED_REL_LABELS = self.PRED_REL_LABELS[non_zeros[:, 0]]  # Use the batch index to resample
