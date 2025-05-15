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

import json
from os import PathLike

import nibabel as nib
import numpy as np
from PIL import ImageColor

from scene_graph_api.scene import SceneGraph as _SceneGraph
from . import Attribute
from .Keypoint import Keypoint
from .Object import BoundingBox
from .Relation import Relation
from ..knowledge import KnowledgeGraph


class SceneGraph(_SceneGraph):
    def __init__(
            self,
            knowledge_graph: KnowledgeGraph,
            image_affine: np.ndarray,
            image_header: nib.Nifti1Header | None,
            bounding_box_objects: list[BoundingBox],
            relations: list[Relation],
            object_labelmap: np.ndarray,
            image_level_attributes: list[Attribute] | None = None,
            image_level_keypoints: list[Keypoint] | None = None,
            obj_common_attributes: list[Attribute] | None = None,
            obj_common_keypoints: list[Keypoint] | None = None,
            rel_common_attributes: list[Attribute] | None = None,
            rel_common_keypoints: list[Keypoint] | None = None,
    ):
        """Note: the labelmap/affine are expected to be depth-first."""
        super().__init__(
            knowledge_graph,
            image_affine,
            image_header,
            bounding_box_objects,
            relations,
            object_labelmap,
            image_level_attributes=image_level_attributes,
            image_level_keypoints=image_level_keypoints,
            obj_common_attributes=obj_common_attributes,
            obj_common_keypoints=obj_common_keypoints,
            rel_common_attributes=rel_common_attributes,
            rel_common_keypoints=rel_common_keypoints
        )

        # Counter used for generating new ids
        self._next_available_id = max(self._bb_by_id.keys()) + 1

        # Masks used for display
        self.object_hitboxes: np.ndarray = np.empty(0)
        self.object_overlay: np.ndarray = np.empty(0)
        self.compute_objects_overlay()

    def save(self, path: str | PathLike) -> bool:
        """
        Save as a json file at given destination path.
        Note: automatically remaps as contiguous indexing.
        :returns: success.
        """
        try:
            # Copy and use super call
            sg_copy = self.copy()
            _SceneGraph.remap_ids_as_contiguous(sg_copy)
            with open(path, "w") as f:
                json.dump(sg_copy.to_json(), f)
            return True
        except FileNotFoundError:
            return False

    def add_bounding_box(self, bb: BoundingBox):
        """Adds the bounding box to the scene graph. Used when merging/splitting bounding boxes."""
        if bb.class_id not in self._bb_by_class_id:
            self._bb_by_class_id[bb.class_id] = []
        self._bb_by_class_id[bb.class_id].append(bb)
        self._bb_by_id[bb.id] = bb

    def remove_bounding_box(self, bb: BoundingBox):
        """
        Removes the bounding box to the scene graph. Used when merging/splitting bounding boxes.
        Note: this method is unsafe and does not clean any relation associated to the removed object.
        :raises: a ValueError if missing.
        """
        self._bb_by_class_id[bb.class_id].remove(bb)
        del self._bb_by_id[bb.id]

    def add_relation(self, rel: Relation):
        """
        Adds the relation to the scene graph.
        Used when merging/splitting segmentations and updating their relations.
        """
        self.relations_by_rule_id[rel.rule_id].append(rel)

    def remove_relation(self, rel: Relation):
        """
        Removes the relation to the scene graph.
        Used when merging/splitting segmentations and updating their relations.
        :raises: a ValueError if missing.
        """
        self.relations_by_rule_id[rel.rule_id].remove(rel)

    def remap_ids_as_contiguous(self):
        """WARNING: is not UI compatible as some Widgets rely on the id to identify the widget for a bb..."""
        raise RuntimeError("This method is not compatible with the way that UI elements us ids to identify objects."
                           "If you wish, to remap ids as contiguous, make a copy and use the super() method.")

    def compute_objects_overlay(self):
        """
        Initialize some masks that are expensive to compute and only used for display
        self.object_hitboxes is used for knowing which object has been clicked (if any).
        self.object_overlay is the same semantic information, but instead of ids we have RGB colors.
        """
        # # Object hitboxes: array used to determine which object was clicked on the UI
        # # The difference with the object labelmap is that hitboxes are drawn
        # self.object_hitboxes = np.zeros_like(self.object_labelmap)
        # # First draw bounding boxes from largest to smallest
        # for bounding_box in sorted(
        #         list(self._bb_by_id.values()),
        #         key=lambda bb: np.prod(np.array(bb.bounding_box[1]) - np.array(bb.bounding_box[0])),
        #         reverse=True
        # ):
        #     obj_class = self.knowledge_graph.get_object_class_by_id(bounding_box.class_id)
        #     if obj_class is None or obj_class.has_mask:
        #         continue
        #     slicer = [
        #         slice(coord_top_left, coord_bottom_right + 1)
        #         for coord_top_left, coord_bottom_right in zip(*bounding_box.bounding_box)
        #     ]
        #     self.object_hitboxes[tuple(slicer)] = bounding_box.id

        # # Then draw segmentations, such that they have priority in case of overlap
        # mask = self.object_labelmap > 0
        # self.object_hitboxes[mask] = self.object_labelmap[mask]

        # Note: we keep a separate array in case we want to change some behaviours later
        self.object_hitboxes = self.object_labelmap.copy()

        # Object overlay i.e. object hitboxes but with RGB colors instead of ids
        self.object_overlay = np.zeros(list(self.object_hitboxes.shape) + [3], dtype=self.object_hitboxes.dtype)
        for obj_id in np.unique(self.object_hitboxes):
            if obj_id == 0:
                continue
            # We have to do this in case the object id is unknown
            if obj_id not in self._bb_by_id:
                hex_color = "#000000"
            else:
                # Have to do this in case the object class is unknown
                obj_class = self.knowledge_graph.get_object_class_by_id(self._bb_by_id[obj_id].class_id)
                hex_color = obj_class.color if obj_class is not None else "#ffffff"
            color = ImageColor.getcolor(hex_color, "RGB")
            self.object_overlay[self.object_hitboxes == obj_id] = color

    def get_next_available_bounding_box_id(self) -> int:
        """Return an id that is not currently used for any bounding box in the graph."""
        # Note: we cannot do that because when we split components, we use their old id
        # (which might have been already assigned here)
        # next_id = 1
        # while next_id in self._bb_by_id:
        #     next_id += 1
        # return next_id
        next_id = self._next_available_id
        self._next_available_id += 1
        return next_id
