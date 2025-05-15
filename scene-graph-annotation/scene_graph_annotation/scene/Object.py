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

import numpy as np

from scene_graph_api.scene import BoundingBox, Object

BoundingBox = BoundingBox
Object = Object


class CompositeBoundingBox(BoundingBox):
    """
    BoundingBox instance obtained by combining two other bounding boxes of the same class.
    The main idea is that sometimes, it's annoying to label a bounding box from a single segmentation component.
    So this class attempts to fix that.
    As such, after being saved to json, the information pertaining to the original two segmentations is lot.
    Note: the composite bounding box will have the attributes of its first component (except: the id).
    Note: the 2 bounding boxes MUST have the same class for this class to make sense.
    Note: An attribute modification does not affect any of the components.
    """

    def __init__(
            self,
            new_id: int,
            new_name: str,
            comp1: BoundingBox,
            comp2: BoundingBox,
            comp1_mask: np.ndarray | None,
            comp2_mask: np.ndarray | None,
    ):
        """
        :param new_id: id for the CompositeComponent
        :param comp1: component1 being merged
        :param comp2: component2 being merged
        :param comp1_mask: mask for component1
        :param comp2_mask: mask for component2
        """
        # Compute the new bb
        new_bb_min, new_bb_max = [], []
        for coordinates in zip(
                comp1.bounding_box[0],
                comp1.bounding_box[1],
                comp2.bounding_box[0],
                comp2.bounding_box[1]
        ):
            new_bb_min.append(min(coordinates))
            new_bb_max.append(max(coordinates))
        new_bb = tuple(new_bb_min), tuple(new_bb_max)

        super().__init__(
            comp1.class_id,
            new_id,
            new_name,
            new_bb,
            common_attributes=[attr.copy() for attr in comp1.common_attributes],
            common_keypoints=[attr.copy() for attr in comp1.common_keypoints],
            attributes=[attr.copy() for attr in comp1.attributes],
            keypoints=[kp.copy() for kp in comp1.keypoints],
        )
        self.comp1 = comp1
        self.comp2 = comp2

        # These two masks are saved so that we can roll back the hitbox/overlay
        # if we need to split back this composite bb
        self.comp1_mask = comp1_mask
        self.comp2_mask = comp2_mask

    @classmethod
    def from_json(cls, json_dict: dict) -> BoundingBox | None:
        raise NotImplementedError("When saving a Composite Bounding Box to JSON, "
                                  "it is converted to a regular Bounding Box. "
                                  "As such a Composite Segmentation should never be directly loaded from JSON. "
                                  "Maybe try BoundingBox.from_json.")


class CompositeBoundingBoxWithSegmentation(CompositeBoundingBox):
    """
    Segmentation instance obtained by combining two other segmentations of the same class.
    The main idea is that during the conversion of a segmentation to a scene graph, each connected component becomes
    a single object. As such, objects can be split due to occlusion.
    So this class attempts to fix that.
    As such, after being saved to json, the information pertaining to the original two segmentations is lot.
    Note: the composite segmentation will have the attributes of its first component (except: the id).
    Note: the 2 segmentations MUST have the same class for this class to make sense.
    Note: An attribute modification does not affect any of the components.

    Note: this class is actually not needed anymore since split/merge methods have been moved.
    """

    @classmethod
    def from_json(cls, json_dict: dict) -> BoundingBox | None:
        raise NotImplementedError("When saving a Composite Segmentation to JSON, "
                                  "it is converted to a regular Segmentation. "
                                  "As such a Composite Segmentation should never be directly loaded from JSON. "
                                  "Maybe try Segmentation.from_json.")
