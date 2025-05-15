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

from abc import ABC
from logging import Logger

from typing_extensions import Self

from .Attribute import Attribute
from .Keypoint import Keypoint
from .SceneGraphComponent import SceneGraphComponent
from ..knowledge import KnowledgeGraph, ObjectClass
from ..utils.parsing import *


class Object(SceneGraphComponent, ABC):
    """
    An object instance in a scene graph.
    We track object-class common and specific attributes and keypoints separately, because their ids are independent.
    """
    _class_id_key = "class_id"
    _id_key = "id"
    _name_key = "name"
    _common_attributes_key = "common_attributes"
    _common_keypoints_key = "common_keypoints"
    _attributes_key = "attributes"
    _keypoints_key = "keypoints"

    def __init__(
            self,
            class_id: int,
            obj_id: int,
            obj_name: str,
            common_attributes: list[Attribute] | None = None,
            common_keypoints: list[Keypoint] | None = None,
            attributes: list[Attribute] | None = None,
            keypoints: list[Keypoint] | None = None
    ):
        self.class_id = class_id
        self.id = obj_id
        self.name = obj_name
        self.common_attributes = common_attributes if common_attributes is not None else []
        self.common_keypoints = common_keypoints if common_keypoints is not None else []
        self.attributes = attributes if attributes is not None else []
        self.keypoints = keypoints if keypoints is not None else []

    @classmethod
    def from_json(cls, json_dict: dict) -> Self:
        class_id = int(json_dict[cls._class_id_key])
        obj_id = int(json_dict[cls._id_key])
        name = json_dict[cls._name_key]
        common_attrs = [Attribute.from_json(obj_dict) for obj_dict in json_dict.get(cls._common_attributes_key, [])]
        common_kps = [Keypoint.from_json(obj_dict) for obj_dict in json_dict.get(cls._common_keypoints_key, [])]
        attributes = [Attribute.from_json(obj_dict) for obj_dict in json_dict.get(cls._attributes_key, [])]
        keypoints = [Keypoint.from_json(obj_dict) for obj_dict in json_dict.get(cls._keypoints_key, [])]

        return Object(
            class_id, obj_id, name,
            common_attributes=common_attrs, common_keypoints=common_kps,
            attributes=attributes, keypoints=keypoints
        )

    def to_json(self) -> dict:
        return {
            self._class_id_key: self.class_id, self._id_key: self.id, self._name_key: self.name,
            self._common_attributes_key: [attr.to_json() for attr in self.common_attributes],
            self._common_keypoints_key: [kp.to_json() for kp in self.common_keypoints],
            self._attributes_key: [attr.to_json() for attr in self.attributes],
            self._keypoints_key: [kp.to_json() for kp in self.keypoints],
        }

    @staticmethod
    def default_name(knowledge: KnowledgeGraph, obj_class_id: int, obj_id: int) -> str:
        """Return a default name for the object."""
        obj_class = knowledge.get_object_class_by_id(obj_class_id)
        if obj_class is not None:
            class_name = obj_class.name
        else:
            class_name = Object.__name__

        if obj_class.is_unique:
            # Unique object so no need for an id
            return class_name
        return f"{class_name} {obj_id}"

    def validate(self, knowledge: KnowledgeGraph, logger: Logger) -> bool:
        """
        Checks that object class is defined in the knowledge graph.
        Attempts to cast the values stored in the JSON file to the correct type.
        Checks that no attribute defined in the object class is missing.
        """
        context_str = f"In Object with id {self.id}:"
        obj_class = knowledge.get_object_class_by_id(self.class_id)
        if obj_class is None:
            logger.error(f"{context_str} Object class id {self.class_id} does not exist in the knowledge graph.")
            return False

        return self.validate_attributes(obj_class, knowledge.obj_common, logger)

    def validate_attributes(self, obj_class: ObjectClass, obj_common_class: ObjectClass, logger: Logger) -> bool:
        """Validate class-common and -specific attributes."""
        context_str = f"In Object with id {self.id}:"
        success = True

        for attr_set, cur_obj_class, name in [
            [self.common_attributes, obj_common_class, "common attributes"],
            [self.attributes, obj_class, "attributes"],
        ]:
            # Validate attribute values
            seen_attr_ids = set()
            for attr in attr_set:
                success &= attr.validate_type_and_value(cur_obj_class, logger)
                seen_attr_ids.add(attr.id)

            # Check that no attribute defined in the object class is missing
            class_attr_ids = {attr.id for attr in cur_obj_class.attributes}
            missing_ids = class_attr_ids.difference(seen_attr_ids)
            if missing_ids:
                logger.error(f"{context_str} Some {name} with ids (" + ", ".join([str(i) for i in missing_ids]) +
                             ") are missing.")
                success = False

        return success

    def validate_keypoint_length(self, n_dim: int, logger: Logger) -> bool:
        """
        Validates that the number of dimensions in the bounding box (and keypoints) matches the one of the segmentation.
        """
        success = True

        # Also check the length of any keypoint
        for kp in self.keypoints:
            success &= kp.validate(n_dim, logger)

        return success

    def get_common_attribute_by_id(self, attr_id: int) -> Attribute | None:
        """Returns the Attribute corresponding to the given id if found."""
        for attr in self.common_attributes:
            if attr.id == attr_id:
                return attr

    def get_common_keypoint_by_id(self, kp_id: int) -> Keypoint | None:
        """Returns the Keypoint corresponding to the given id if found."""
        for kp in self.common_keypoints:
            if kp.id == kp_id:
                return kp

    def get_attribute_by_id(self, attr_id: int) -> Attribute | None:
        """Returns the Attribute corresponding to the given id if found."""
        for attr in self.attributes:
            if attr.id == attr_id:
                return attr

    def get_keypoint_by_id(self, kp_id: int) -> Keypoint | None:
        """Returns the Keypoint corresponding to the given id if found."""
        for kp in self.keypoints:
            if kp.id == kp_id:
                return kp

    def __repr__(self):
        common_attrs_repr = ",".join(map(repr, self.common_attributes))
        common_kps_repr = ",".join(map(repr, self.common_keypoints))
        attrs_repr = ",".join(map(repr, self.attributes))
        kps_repr = ",".join(map(repr, self.keypoints))
        return (f"Object(class_id={self.class_id}, id={self.id}, name='{self.name}', "
                f"obj_common_attributes=[{common_attrs_repr}], obj_common_attributes=[{common_kps_repr}], "
                f"attributes=[{attrs_repr}], keypoints=[{kps_repr}])")

    @classmethod
    def schema(cls) -> dict:
        """JSON schema for structure validation."""
        return {
            "id": cls.schema_name(),
            "type": "object",
            "properties": {
                cls._class_id_key: {"$ref": ID_SCHEMA_NAME},
                cls._id_key: {"$ref": ID_SCHEMA_NAME},
                cls._name_key: {"type": "string"},
                cls._common_attributes_key: {
                    "type": "array",
                    "items": {"$ref": Attribute.schema_name()}
                },
                cls._common_keypoints_key: {
                    "type": "array",
                    "items": {"$ref": Keypoint.schema_name()}
                },
                cls._attributes_key: {
                    "type": "array",
                    "items": {"$ref": Attribute.schema_name()}
                },
                cls._keypoints_key: {
                    "type": "array",
                    "items": {"$ref": Keypoint.schema_name()}
                },
            },
            "required": [cls._class_id_key, cls._id_key, cls._name_key],
            "additionalProperties": False
        }


class BoundingBox(Object):
    """
    Bounding box instance in a scene graph. Expected format: ((z1 if 3D), y1, x1), (z2 if 3D), y2, x2).
    Can optionally hava a corresponding segmentation mask.

    Note: legacy class from the time, where segmentations had a different class. Now it's only a flag.
          But lots of code still references the Object class so, we leave the two classes split.
    """
    _bb_key = "bounding_box"

    def __init__(
            self,
            class_id: int,
            obj_id: int,
            obj_name: str,
            bounding_box: list[list[float]] | tuple[tuple[float, ...], tuple[float, ...]],
            common_attributes: list[Attribute] | None = None,
            common_keypoints: list[Keypoint] | None = None,
            attributes: list[Attribute] | None = None,
            keypoints: list[Keypoint] | None = None
    ):
        super().__init__(
            class_id, obj_id, obj_name,
            common_attributes=common_attributes, common_keypoints=common_keypoints,
            attributes=attributes, keypoints=keypoints
        )
        self.bounding_box: tuple[tuple[float, ...], tuple[float, ...]] = tuple(bounding_box[0]), tuple(bounding_box[1])

    @classmethod
    def from_json(cls, json_dict: dict) -> Self:
        base_object = super().from_json(json_dict)
        bbox = tuple(map(int, json_dict[cls._bb_key][0])), tuple(map(int, json_dict[cls._bb_key][1]))

        return BoundingBox(
            base_object.class_id, base_object.id, base_object.name, bounding_box=bbox,
            common_attributes=base_object.common_attributes, common_keypoints=base_object.common_keypoints,
            attributes=base_object.attributes, keypoints=base_object.keypoints
        )

    def to_json(self) -> dict:
        out = super().to_json()
        out[self._bb_key] = [list(self.bounding_box[0]), list(self.bounding_box[1])]
        return out

    def validate(self, knowledge: KnowledgeGraph, logger: Logger) -> bool:
        """Also checks that the bounding box coordinates are in the format upper left, bottom right."""
        success = super().validate(knowledge, logger)
        context_str = f"In BoundingBox with id {self.id}:"

        # Check that the number of coordinates is even
        if len(self.bounding_box[0]) != len(self.bounding_box[1]):
            logger.error(f"{context_str} invalid bounding box, please check the coordinates.")
            return False

        # Check that the bounding box coordinates are valid i.e. top right, bottom left corner
        for top_right, bottom_left in zip(self.bounding_box[0], self.bounding_box[1]):
            if top_right > bottom_left:
                logger.error(f"{context_str} invalid bounding box, please check the coordinates.")
                return False

        return success

    def validate_bounding_box_length(self, n_dim: int, logger: Logger) -> bool:
        """
        Validates that the number of dimensions in the bounding box (and keypoints) matches the one of the segmentation.
        """
        context_str = f"In Object with id {self.id}:"
        success = True

        # Check that the coordinate lists have as many coordinates as there are dimensions in the volume
        for idx in range(2):
            if len(self.bounding_box[idx]) != n_dim:
                logger.error(f"{context_str} invalid bounding box, the coordinate list number {idx} "
                             f"does not have as many coordinates ({len(self.bounding_box[idx])}) as "
                             f"there are dimensions in the target volume ({n_dim}).")
                success = False

        return success

    def __repr__(self):
        common_attrs_repr = ",".join(map(repr, self.common_attributes))
        common_kps_repr = ",".join(map(repr, self.common_keypoints))
        attrs_repr = ",".join(map(repr, self.attributes))
        kps_repr = ",".join(map(repr, self.keypoints))
        return (f"BoundingBox(class_id={self.class_id}, id={self.id}, name='{self.name}', "
                f"obj_common_attributes=[{common_attrs_repr}], obj_common_attributes=[{common_kps_repr}], "
                f"attributes=[{attrs_repr}], keypoints=[{kps_repr}], bounding_box={self.bounding_box})")

    def size(self) -> tuple[int, ...]:
        """Returns the size of the bounding box (depth first)."""
        return tuple(self.bounding_box[1][dim] - self.bounding_box[0][dim] for dim in range(len(self.bounding_box[0])))

    @classmethod
    def schema(cls) -> dict:
        base_schema = super().schema()
        base_schema["id"] = cls.schema_name()
        base_schema["properties"][cls._bb_key] = {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 1,
            },
            "minItems": 2,
            "maxItems": 2,
        }
        base_schema["required"].append(cls._bb_key)
        return base_schema
