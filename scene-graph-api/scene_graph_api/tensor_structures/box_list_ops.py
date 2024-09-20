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
import copy
from functools import reduce
from typing import TypeVar, Sequence, Iterable, Hashable

import torch
from scipy.linalg import block_diag

from scene_graph_api.utils.tensor import affine_transformation_grid, compute_bounding_box
from .box_list_field_extractor import FieldExtractor
from .box_list_fields import AnnotationField, PredictionField
from .segmentation_mask import AbstractMaskList
from ..utils.indexing import FlipDim, sanitize_cropping_box

BoxListBase = TypeVar("BoxListBase", bound="BoxList")

_SIZE_T = tuple[int, ...]


class BoxListOps:
    """Helper class for various operations on BoxLists."""

    KNOWN_SEGMENTATION_FIELDS = [
        AnnotationField.LABELMAP,
        AnnotationField.SEGMENTATION,
        AnnotationField.MASKS,
        PredictionField.PRED_MASKS,
        PredictionField.PRED_SEGMENTATION
    ]

    EPS = 1e-5

    @staticmethod
    def split_into_zyxzyx(boxlist: BoxListBase) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        """
        Return the box content as zyxzyx
        :returns: (minimums, maximums) zyx ordered.
        """
        if boxlist.mode == boxlist.Mode.zyxzyx:
            bounds = boxlist.boxes.split(1, dim=-1)  # Sequence of length 2 * boxlist.n_dim
            return bounds[:boxlist.n_dim], bounds[boxlist.n_dim:]

        starts_lengths = boxlist.boxes.split(1, dim=-1)  # Sequence of length 2 * boxlist.n_dim
        starts = starts_lengths[:boxlist.n_dim]
        lengths = starts_lengths[boxlist.n_dim:]
        lengths = tuple(starts[i] + (lengths[i] - 1).clamp(min=0) for i in range(boxlist.n_dim))
        return starts, lengths

    @staticmethod
    def resize(
            boxlist: BoxListBase,
            size: _SIZE_T,
            additional_mask_fields: Iterable[Hashable] | None = None
    ) -> BoxListBase:
        """
        Return a resized copy of this bounding box.
        Note: resizes the MASKS field if present.
        Note: LABELMAP, and SEGMENTATION fields are copied (if present), as there is no easy way to resize them...
              ... unless the resizing is a no-op.
              If you insist on resizing them, consider using the BoxList.affine_transformation method
              from scene_graph_prediction.
        :param boxlist: BoxList to resize.
        :param size: with zyx ordering.
        :param additional_mask_fields: additional segmentation fields that need to be resized,
                                       (see KNOWN_SEGMENTATION_FIELDS).
        """
        # TODO add tests that only relevant fields get updated
        if size == boxlist.size:
            return boxlist

        assert len(size) == boxlist.n_dim
        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size, boxlist.size))
        if ratios == (ratios[0],) * boxlist.n_dim:
            ratio = ratios[0]
            scaled_box = boxlist.boxes * ratio
        else:
            minimums, maximums = BoxListOps.split_into_zyxzyx(boxlist)
            minimums = tuple(minimums[i] * ratios[i] for i in range(boxlist.n_dim))
            # We have to do these shenanigans because we expect the length of a box to always be (max-min+1)
            # In particular, it should also be true after the resize operation
            maximums = tuple((maximums[i] + 1) * ratios[i] - 1 for i in range(boxlist.n_dim))
            scaled_box = torch.cat(minimums + maximums, dim=-1)

        bbox = type(boxlist)(scaled_box, size, mode=boxlist.mode)

        all_mask_fields = set(BoxListOps.KNOWN_SEGMENTATION_FIELDS)
        if additional_mask_fields is not None:
            all_mask_fields.update(additional_mask_fields)

        # Copy all fields and resize relevant ones
        for k, v in boxlist.extra_fields.items():
            if k in all_mask_fields:
                # Resize segmentation fields
                if isinstance(v, AbstractMaskList):
                    v = v.resize(size)
                elif isinstance(v, torch.Tensor):
                    # We need to unsqueeze and then squeeze to go from DxHxW to NxCxDxHxW
                    # Note: We assume that if masks can have the shape NxDxHxW, then it is an AbstractMaskList
                    v = torch.nn.functional.interpolate(
                        input=v[None, None].float(),  # As NxCxDxHxW
                        size=size,
                        mode="bilinear" if boxlist.n_dim == 2 else "trilinear",
                        align_corners=False
                    )[0, 0].type_as(v)
                else:
                    raise ValueError(f"Field {k} is a segmentation field, "
                                     f"but is neither a Tensor or an AbstractMaskList: ({type(v)})")
            bbox.add_field(k, v, indexing_power=boxlist.fields_indexing_power[k])

        if (masks := bbox.get_field(AnnotationField.MASKS)) is not None:
            masks: AbstractMaskList = masks.resize(size)
            bbox.MASKS = masks

        return bbox

    @staticmethod
    def crop(
            boxlist: BoxListBase,
            box: Sequence[float],
            additional_mask_fields: Iterable[Hashable] | None = None
    ) -> BoxListBase:
        """
        Crop a rectangular region from this zyxzyx bounding box.
        Note: crops the MASKS, LABELMAP, and SEGMENTATION fields if present.
        """
        # TODO add tests that only relevant fields get updated
        assert len(box) == boxlist.n_dim * 2

        box = sanitize_cropping_box(box, boxlist.size)
        box_shape = tuple(int(box[boxlist.n_dim + dim] - box[dim] + 1) for dim in range(boxlist.n_dim))
        minimums, maximums = BoxListOps.split_into_zyxzyx(boxlist)

        # Separate clamp calls because of mix of int/Tensor arguments
        minimums = tuple(
            (minimums[dim] - box[dim]).clamp(min=0, max=box_shape[dim] - 1) for dim in range(boxlist.n_dim))
        maximums = tuple(
            (maximums[dim] - box[dim]).clamp(min=0, max=box_shape[dim] - 1) for dim in range(boxlist.n_dim))

        cropped_box = torch.cat(minimums + maximums, dim=-1)
        bbox = type(boxlist)(cropped_box, box_shape, mode=boxlist.Mode.zyxzyx).convert(boxlist.mode)

        all_mask_fields = set(BoxListOps.KNOWN_SEGMENTATION_FIELDS)
        if additional_mask_fields is not None:
            all_mask_fields.update(additional_mask_fields)

        # Copy all fields and crop relevant ones
        for k, v in boxlist.extra_fields.items():
            if k in all_mask_fields:
                # Crop segmentation fields
                if isinstance(v, AbstractMaskList):
                    v = v.crop(box)
                elif isinstance(v, torch.Tensor):
                    # Note: We assume that if masks can have the shape NxDxHxW, then it is an AbstractMaskList
                    slicer = tuple(slice(box[dim], box[dim + boxlist.n_dim] + 1) for dim in range(boxlist.n_dim))
                    v = v[slicer]
                else:
                    raise ValueError(f"Field {k} is a segmentation field, "
                                     f"but is neither a Tensor or an AbstractMaskList: ({type(v)})")
            bbox.add_field(k, v, indexing_power=boxlist.fields_indexing_power[k])

        return bbox

    @staticmethod
    def flip(
            boxlist: BoxListBase,
            dim: FlipDim,
            additional_mask_fields: Iterable[Hashable] | None = None
    ) -> BoxListBase:
        """
        Flip the BoxList along specified dimension.
        Note: support only for 1D, 2D or 3D.
        Note: flips the MASKS, LABELMAP, and SEGMENTATION fields if present.
        """
        # TODO add tests that only relevant fields get updated
        assert boxlist.n_dim >= abs(dim.to_idx_from_last())
        int_dim = boxlist.n_dim + dim.to_idx_from_last()

        minimums, maximums = BoxListOps.split_into_zyxzyx(boxlist)
        transposed_minimums = list(minimums)
        transposed_maximums = list(maximums)

        transposed_minimums[int_dim] = boxlist.size[int_dim] - maximums[int_dim] - 1
        transposed_maximums[int_dim] = boxlist.size[int_dim] - minimums[int_dim] - 1
        transposed_boxes = torch.cat(tuple(transposed_minimums + transposed_maximums), dim=-1)

        bbox = type(boxlist)(transposed_boxes, boxlist.size, mode=boxlist.Mode.zyxzyx)

        all_mask_fields = set(BoxListOps.KNOWN_SEGMENTATION_FIELDS)
        if additional_mask_fields is not None:
            all_mask_fields.update(additional_mask_fields)

        # Copy all fields and flip relevant ones
        for k, v in boxlist.extra_fields.items():
            if k in all_mask_fields:
                # Flip segmentation fields
                if isinstance(v, AbstractMaskList):
                    v = v.flip(dim)
                elif isinstance(v, torch.Tensor):
                    # Note: We assume that if masks can have the shape NxDxHxW, then it is an AbstractMaskList
                    v = torch.flip(v, dims=(v.dim() + dim.to_idx_from_last(),))
                else:
                    raise ValueError(f"Field {k} is a segmentation field, "
                                     f"but is neither a Tensor or an AbstractMaskList: ({type(v)})")
            bbox.add_field(k, v, indexing_power=boxlist.fields_indexing_power[k])

        return bbox.convert(boxlist.mode)

    @staticmethod
    def centers(boxlist: BoxListBase) -> torch.Tensor:
        """Compute the zyx center point of the boxes."""
        center_points = [
            (boxlist.boxes[:, dim] + boxlist.boxes[:, dim + boxlist.n_dim]) / 2
            for dim in range(boxlist.n_dim)
        ]
        return torch.stack(center_points, dim=1)

    @staticmethod
    def remove_small_boxes(boxlist: BoxListBase, min_size: int | tuple[int, ...]) -> BoxListBase:
        """
        Only keep boxes with all sides >= min_size.
        Note: Also updates the AnnotationField.LABELMAP field in place (if present).
              We do this because removing  small boxes if a pre-processing step rather than a sampling/inference one.
              So we need to update the labelmap, which we would otherwise not want to do.
        Note: Same note for the SEGMENTATION field. Updating it requires the labelmap.
        """
        if isinstance(min_size, int):
            min_size = (min_size,) * boxlist.n_dim
        assert len(min_size) == boxlist.n_dim

        zyxdhw_boxes = boxlist.convert(boxlist.Mode.zyxdhw).boxes
        starts_lengths = zyxdhw_boxes.unbind(dim=1)  # Sequence of length 2 * self.n_dim
        lengths = starts_lengths[boxlist.n_dim:]
        keep = reduce(lambda a, b: a & b,
                      [lengths[dim] >= min_size[dim] for dim in range(boxlist.n_dim)])

        try:
            return BoxListOps.indexing_with_segmentation_update(boxlist, keep)
        except ValueError:
            # LABELMAP extraction failed, so we use the normal indexing
            return boxlist[keep]

    @staticmethod
    def clip_to_image(boxlist: BoxListBase, remove_empty: bool = True):
        """Clip boxes to the image size. Optionally remove empty boxes."""
        boxes = boxlist.boxes.clone()

        for i in range(boxlist.n_dim):
            boxes[:, i].clamp_(min=0, max=boxlist.size[i] - 1)
            boxes[:, i + boxlist.n_dim].clamp_(min=0, max=boxlist.size[i] - 1)

        new_boxlist = boxlist.copy_with_all_fields()
        new_boxlist.boxes = boxes

        if not remove_empty:
            return new_boxlist

        # We want to keep boxes that are not 1x1x1
        # And that are not slices at the edge
        keep = reduce(lambda a, b: a & b, [(boxes[:, boxlist.n_dim + i] > boxes[:, i]) |
                                           (boxes[:, boxlist.n_dim + i] == boxes[:, i]) &
                                           (boxes[:, i] != boxlist.size[i] - 1) & (boxes[:, i] != 0)
                                           for i in range(boxlist.n_dim)])
        return new_boxlist[keep]

    @staticmethod
    def volume(boxlist: BoxListBase) -> torch.Tensor:
        """Compute the volume for 3D and the area for 2D."""
        boxes = boxlist.boxes
        if boxlist.mode == boxlist.Mode.zyxzyx:
            return reduce(
                lambda a, b: a * b,
                [boxes[:, boxlist.n_dim + i] - boxes[:, i] + 1 for i in range(boxlist.n_dim)]
            )

        return reduce(lambda a, b: a * b, [boxes[:, boxlist.n_dim + i] for i in range(boxlist.n_dim)])

    @staticmethod
    def normalized_area(boxlist: BoxListBase) -> torch.Tensor:
        """Computes the normalized area (width x height) no matter the number of dimensions (>=2)."""
        if boxlist.n_dim < 2:
            raise ValueError(f"Number of dimensions ({boxlist.n_dim}) too low.")

        boxes = boxlist.boxes
        img_area = boxlist.size[-1] * boxlist.size[-2]
        if boxlist.mode == boxlist.Mode.zyxzyx:
            return (boxes[:, -1] - boxes[:, -1 - boxlist.n_dim] + 1) * \
                (boxes[:, -2] - boxes[:, -2 - boxlist.n_dim] + 1) / \
                img_area

        return boxes[:, -1] * boxes[:, -2] / img_area

    # noinspection DuplicatedCode
    @staticmethod
    def iou(boxlist: BoxListBase, other_boxlist: BoxListBase) -> torch.FloatTensor:
        """
        Compute the intersection over union of two set of boxes.
        https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
        :param boxlist: bounding boxes, sized [N,2 * self.n_dim].
        :param other_boxlist: bounding boxes, sized [M,2 * self.n_dim].
        :return: iou sized [N,M]
        """
        boxlist = boxlist.convert(boxlist.Mode.zyxzyx)
        other_boxlist = other_boxlist.convert(boxlist.Mode.zyxzyx)

        volume1 = BoxListOps.volume(boxlist)
        volume2 = BoxListOps.volume(other_boxlist)

        box1, box2 = boxlist.boxes, other_boxlist.boxes

        lt = torch.max(box1[:, None, :boxlist.n_dim], box2[:, :boxlist.n_dim])  # [N,M,boxlist.n_dim]
        rb = torch.min(box1[:, None, boxlist.n_dim:], box2[:, boxlist.n_dim:])  # [N,M,boxlist.n_dim]

        dhw = (rb - lt + 1).clamp(min=0)  # [N,M,boxlist.n_dim]
        inter = reduce(lambda a, b: a * b, [dhw[:, :, i] for i in range(boxlist.n_dim)])
        union = volume1[:, None] + volume2 - inter + BoxListOps.EPS
        return inter / union

    @staticmethod
    def union(boxlist: BoxListBase, other_boxlist: BoxListBase) -> BoxListBase:
        """
        Computes the union region (one to one) of two set of boxes of the SAME length.
        :param boxlist: bounding boxes, sized [N,2 * self.n_dim].
        :param other_boxlist: bounding boxes, sized [N,2 * self.n_dim].
        :return: zyxzyx union, sized [N,2 * self.n_dim].
        """
        assert len(boxlist) == len(other_boxlist) and boxlist.size == other_boxlist.size
        boxlist1 = boxlist.convert(boxlist.Mode.zyxzyx)
        boxlist2 = other_boxlist.convert(boxlist.Mode.zyxzyx)
        union_box = torch.cat(
            (torch.min(boxlist1.boxes[:, :boxlist.n_dim], boxlist2.boxes[:, :boxlist.n_dim]),
             torch.max(boxlist1.boxes[:, boxlist.n_dim:], boxlist2.boxes[:, boxlist.n_dim:])),
            dim=1
        )
        return type(boxlist)(union_box, boxlist1.size, boxlist.Mode.zyxzyx)

    @staticmethod
    def intersection(boxlist: BoxListBase, other_boxlist: BoxListBase, filter_empty: bool = False) -> BoxListBase:
        """
        Compute the intersection region (one to one) of two set of boxes of the SAME length.
        :param boxlist: bounding boxes, sized [N,2 * self.n_dim].
        :param other_boxlist: bounding boxes, sized [N,2 * self.n_dim].
        :param filter_empty: Whether to remove empty boxes.
        :return: zyxzyx intersection, sized [N,2 * self.n_dim] if filter_empty is False else [M<N, 2 * self.n_dim].
        """
        assert len(boxlist) == len(other_boxlist) and boxlist.size == other_boxlist.size
        boxlist1 = boxlist.convert(boxlist.Mode.zyxzyx)
        other_boxlist = other_boxlist.convert(boxlist.Mode.zyxzyx)

        inter_box = torch.cat(
            (
                torch.max(boxlist1.boxes[:, :boxlist.n_dim], other_boxlist.boxes[:, :boxlist.n_dim]),
                torch.min(boxlist1.boxes[:, boxlist.n_dim:], other_boxlist.boxes[:, boxlist.n_dim:])
            ),
            dim=1
        )

        # Find invalid intersections by checking for each dim that for boxlists A and B: x1a < x2b or x1b < x2a
        # noinspection PyTypeChecker
        invalid_bbox: torch.Tensor = reduce(
            lambda a, b: a | b,
            [
                (other_boxlist.boxes[:, boxlist.n_dim + dim] < boxlist1.boxes[:, dim]) |
                (boxlist1.boxes[:, boxlist.n_dim + dim] < other_boxlist.boxes[:, dim])
                for dim in range(boxlist.n_dim)
            ]
        )
        if filter_empty:
            # Either filter the boxes out
            inter_box = inter_box[torch.logical_not(invalid_bbox)]
        else:
            # Or replace them with boxes of volume 0
            inter_box[invalid_bbox] = torch.tensor((0,) * boxlist.n_dim + (-1,) * boxlist.n_dim).to(inter_box)

        return type(boxlist)(inter_box, boxlist1.size, boxlist.Mode.zyxzyx)

    @staticmethod
    def cat(boxlists: Sequence[BoxListBase]) -> BoxListBase:
        """
        Concatenates a non-empty list of BoxList (having the same size, mode and field keys) into a single BoxList.
        Note: this method only makes sense if we have multiple BoxLists for the same image e.g. multiple RPN levels.
        Note: fields of different indexing power are concatenated the following way:
              - 0: only the field value of the first BoxList is kept
              - 1: all fields get torch.cat
              - 2: a 2D matrix is filled with diagonal blocks
        """
        from .box_list import BoxList

        # Assert sequence of BoxList
        assert isinstance(boxlists, Sequence) and len(boxlists) > 0
        assert all(isinstance(bbox, BoxList) for bbox in boxlists)
        # Assert same size
        size = boxlists[0].size
        assert all(bbox.size == size for bbox in boxlists)
        # Assert same mode
        mode = boxlists[0].mode
        assert all(bbox.mode == mode for bbox in boxlists)
        # Assert same fields
        fields = set(boxlists[0].fields())
        assert all(set(bbox.fields()) == fields for bbox in boxlists)

        # Cat bounding boxes
        cat_boxes = BoxList(torch.cat([bbox.boxes for bbox in boxlists], dim=0), size, mode)

        # Copy fields
        for field in fields:
            indexing_power = boxlists[0].fields_indexing_power[field]
            if indexing_power > 2:
                raise RuntimeError(f"Cat with fields of indexing power other than 1, and 2 ({field}) is not supported.")
            if indexing_power == 0:
                data = boxlists[0].get_field(field)
            elif indexing_power == 1:
                # Indexing power 1
                data = torch.cat([bbox.get_field(field) for bbox in boxlists], dim=0)
            else:
                # Indexing power 2
                matrix_list = [bbox.get_field(field).numpy() for bbox in boxlists]
                data = torch.from_numpy(block_diag(*matrix_list))
            cat_boxes.add_field(field, data, indexing_power=indexing_power)

        return cat_boxes

    @staticmethod
    def affine_transformation(
            boxlist: BoxListBase,
            translate: tuple[float, ...] = (0.,),
            scale: tuple[float, ...] = (1.,),
            rotate: tuple[float, ...] = (0,),
            output_labelmap: bool = True,
            output_segmentation: bool = False,
            output_masks: bool = False,
            raise_on_missing_bbox: bool = False
    ) -> BoxListBase:
        """
        This method is only available for datasets where the segmentation is available,
        as we use them to compute the new bounding boxes after the transformation.
        Warning: this function can be used with a prediction as the PRED_MASKS can be used to compute a LABELMAP,
                 however all generated mask fields will be AnnotationFields.
        Note: for more information regarding the arguments, see utils.tensor.affine_transformation_grid.
        Note: masks will also be converted to a BinaryMaskList.
        :raises RuntimeError: if raise_on_missing_bbox is True and any box has been removed (because it is out of frame)
        :returns: the transformed BoxList.
        """
        if len(boxlist) == 0:
            return boxlist

        orig_mode = boxlist.mode
        boxlist = boxlist.copy_with_all_fields().convert(boxlist.Mode.zyxzyx)  # We need zyxzyx for bounding box compute
        labelmap = FieldExtractor.labelmap(boxlist)
        if not isinstance(labelmap, torch.Tensor):
            raise ValueError(f"Expected labelmap to be a Tensor, got {type(labelmap)} instead...")

        # Compute the sampling grid
        grid = affine_transformation_grid(boxlist.size, translate, scale, rotate)
        # Adjust the device
        grid = grid.to(device=boxlist.boxes.device)

        transformed_labelmap = torch.nn.functional.grid_sample(
            labelmap[None, None].float(),
            grid,
            align_corners=False,
            mode="nearest"
        )[0, 0].to(labelmap)
        boxlist.LABELMAP = transformed_labelmap

        # Compute the new bounding boxes (zyxzyx)
        # noinspection PyTypeChecker
        boxlist.boxes = torch.cat(
            tuple(
                compute_bounding_box(transformed_labelmap == box_idx + 1)
                for box_idx in range(len(boxlist))
            )
        ).to(boxlist.boxes).view(-1, 2 * boxlist.n_dim)

        # Remove any box that might have disappeared
        filtered_boxlist = BoxListOps.remove_small_boxes(boxlist, 1)
        if raise_on_missing_bbox and len(filtered_boxlist) < len(boxlist):
            raise RuntimeError(f"{len(boxlist) - len(filtered_boxlist)} are missing after the transformation.")

        if output_segmentation:
            filtered_boxlist.SEGMENTATION = FieldExtractor.segmentation(filtered_boxlist)

        if output_masks:
            filtered_boxlist.MASKS = FieldExtractor.masks(filtered_boxlist)

        if not output_labelmap:
            filtered_boxlist.del_field(filtered_boxlist.AnnotationField.LABELMAP)

        return filtered_boxlist.convert(orig_mode)

    @staticmethod
    def indexing_with_segmentation_update(
            boxlist: BoxListBase,
            item: slice | list[int] | list[bool] | torch.Tensor
    ) -> BoxListBase:
        """
        The default behaviour for boxlist[item] is to only update MASKS as it comes for free.
        However, the LABELMAP and the SEGMENTATION are not affected.
        This method also updates these fields.
        Note: if a box gets duplicated, then only the lowest index will be present in the LABELMAP.
        :raises ValueError: if the LABELMAP cannot be extracted.
        """
        # Check that we can extract the LABELMAP before anything else
        orig_labelmap = FieldExtractor.labelmap(boxlist)

        new_boxlist = boxlist[item]

        # First determine the new mapping
        mapping = torch.tensor(list(range(len(boxlist))), dtype=torch.int64, device=boxlist.boxes.device)[item]

        # Get all possible segmentations (we always work with the LABELMAP instead of the MASKS)
        labelmap = torch.zeros_like(orig_labelmap)
        segmentation = boxlist.get_field(boxlist.AnnotationField.SEGMENTATION)
        pred_segmentation = boxlist.get_field(boxlist.PredictionField.PRED_SEGMENTATION)
        if segmentation is not None:
            segmentation = copy.deepcopy(segmentation)
        if pred_segmentation is not None:
            pred_segmentation = copy.deepcopy(pred_segmentation)

        # For each original box
        for box_id in range(1, len(boxlist) + 1):
            # Find if it is still present are indexing
            # Note: Check that id/labelmap have the same int type
            # noinspection PyUnresolvedReferences
            box_id_mapped = (mapping == box_id - 1).nonzero().to(labelmap)
            box_mask = [orig_labelmap == box_id]

            if len(box_id_mapped) > 0:
                # The box is there and only the LABELMAP needs to be updated (id remapping)
                # Note: we only ever use the lowest id in the mapping for which we have a match
                labelmap[box_mask] = box_id_mapped[0] + 1
            else:
                # The box is not there anymore. The LABELMAP was already zeroed so only
                if segmentation is not None:
                    segmentation[box_mask] = 0
                if pred_segmentation is not None:
                    pred_segmentation[box_mask] = 0

        # Finally set the fields that the original boxlist had
        if boxlist.has_field(boxlist.AnnotationField.LABELMAP):
            new_boxlist.LABELMAP = labelmap
        if segmentation is not None:
            new_boxlist.SEGMENTATION = segmentation
        if pred_segmentation is not None:
            new_boxlist.PRED_SEGMENTATION = pred_segmentation

        return new_boxlist
