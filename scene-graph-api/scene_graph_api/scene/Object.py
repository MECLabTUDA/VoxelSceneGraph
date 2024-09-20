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

from .ObjectAttribute import ObjectAttribute
from .SceneGraphComponent import SceneGraphComponent
from ..knowledge import KnowledgeGraph, ObjectClass
from ..utils.parsing import *


class Object(SceneGraphComponent, ABC):
    """
    An object instance in a scene graph.
    Can either be a segmentation or a bounding box.
    The bounding box key is optional and a segmentation cannot be a bounding box.
    A composite object also cannot contain a or be a bounding box.
    Bounding box expected in format list of size 2*D where D is the number of dimensions in the image.
    """
    _class_id_key = "class_id"
    _id_key = "id"
    _name_key = "name"
    _attributes_key = "attributes"

    def __init__(
            self,
            class_id: int,
            obj_id: int,
            obj_name: str,
            attributes: list[ObjectAttribute] | None
    ):
        self.class_id = class_id
        self.id = obj_id
        self.name = obj_name
        self.attributes = attributes if attributes is not None else []

    @classmethod
    def from_json(cls, json_dict: dict) -> Self:
        """
        Used to load common fields between object classes i.e. class id, object id, object name and attributes.
        Also returns whether loading was successful.
        """
        # Note: code moved to class BoundingBox. Class Object should be merged with it at some point...
        raise NotImplementedError

    def to_json(self) -> dict:
        # Note: code moved to class BoundingBox. Class Object should be merged with it at some point...
        raise NotImplementedError

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

        return self.validate_attributes(obj_class, logger)

    def validate_attributes(self, obj_class: ObjectClass, logger: Logger) -> bool:
        """Validate attributes."""
        context_str = f"In Object with id {self.id}:"
        success = True

        # Validate attribute values
        seen_attr_ids = set()
        for attr in self.attributes:
            success &= attr.validate_type_and_value(obj_class, logger)
            seen_attr_ids.add(attr.id)

        # Check that no attribute defined in the object class is missing
        class_attr_ids = {attr.id for attr in obj_class.attributes}
        missing_ids = class_attr_ids.difference(seen_attr_ids)
        if missing_ids:
            logger.error(f"{context_str} Some attributes with ids (" + ", ".join([str(i) for i in missing_ids]) +
                         ") are missing.")
            success = False

        return success

    def get_attribute_by_id(self, attr_id: int) -> ObjectAttribute:
        """Returns the ObjectAttribute corresponding to the given id if found."""
        for attr in self.attributes:
            if attr.id == attr_id:
                return attr

    def __repr__(self):
        # Currently only added for tests to support using Objects
        return "Object()"

    @classmethod
    def schema(cls) -> dict:
        """JSON schema for structure validation."""
        return {}


class BoundingBox(Object):
    """
    Bounding box instance in a scene graph. Expected format: ((z1 if 3D), y1, x1), (z2 if 3D), y2, x2).
    Can optionally hava a corresponding segmentation mask.

    Note: legacy class from the time where segmentations had a different class. Now it's only a flag.
          But lot's of code still references the Object class so, we leave the two classes split.
    """
    _bb_key = "bounding_box"

    def __init__(
            self,
            class_id: int,
            obj_id: int,
            obj_name: str,
            attributes: list[ObjectAttribute] | None,
            bounding_box: list[list[int]] | tuple[tuple[int, ...], tuple[int, ...]]
    ):
        super().__init__(class_id, obj_id, obj_name, attributes)
        self.bounding_box: tuple[tuple[int, ...], tuple[int, ...]] = tuple(bounding_box[0]), tuple(bounding_box[1])

    @classmethod
    def from_json(cls, json_dict: dict) -> Self:
        class_id = int(json_dict[cls._class_id_key])
        obj_id = int(json_dict[cls._id_key])
        name = json_dict[cls._name_key]
        attributes = [ObjectAttribute.from_json(obj_dict) for obj_dict in json_dict.get(cls._attributes_key, [])]
        bbox = tuple(map(int, json_dict[cls._bb_key][0])), tuple(map(int, json_dict[cls._bb_key][1]))

        return BoundingBox(class_id, obj_id, name, attributes, bounding_box=bbox)

    def to_json(self) -> dict:
        return {
            self._class_id_key: self.class_id, self._id_key: self.id, self._name_key: self.name,
            self._attributes_key: [attr.to_json() for attr in self.attributes],
            self._bb_key: [list(self.bounding_box[0]), list(self.bounding_box[1])],
        }

    def validate(self, knowledge: KnowledgeGraph, logger: Logger) -> bool:
        """Also checks that the bounding box coordinates are in the format upper left, bottom right."""
        success = super().validate(knowledge, logger)
        context_str = f"In Object with id {self.id}:"

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

    def validate_bounding_box_length_with_mask(self, volume_dim_cnt: int, logger: Logger) -> bool:
        """Checks that the number of dimensions in the bounding box matches the one of the segmentation."""
        context_str = f"In Object with id {self.id}:"
        # Check that the coordinate lists have as many coordinates as there are dimensions in the volume
        for idx in range(2):
            if len(self.bounding_box[idx]) != volume_dim_cnt:
                logger.error(f"{context_str} invalid bounding box, the coordinate list number {idx} "
                             f"does not have as many coordinates ({len(self.bounding_box[idx])}) as "
                             f"there are dimensions in the target volume ({volume_dim_cnt}).")
                return False
        return True

    def __repr__(self):
        attrs_repr = ",".join(map(repr, self.attributes))
        return f"BoundingBox(class_id={self.class_id}, id={self.id}, name='{self.name}', attributes=[{attrs_repr}]," \
               f"bounding_box={self.bounding_box})"

    def size(self) -> tuple[int, ...]:
        """Returns the size of the bounding box (depth first)."""
        return tuple(self.bounding_box[1][dim] - self.bounding_box[0][dim] for dim in range(len(self.bounding_box[0])))

    @classmethod
    def schema(cls) -> dict:
        return {
            "id": cls.schema_name(),
            "type": "object",
            "properties": {
                cls._class_id_key: {"$ref": ID_SCHEMA_NAME},
                cls._id_key: {"$ref": ID_SCHEMA_NAME},
                cls._name_key: {"type": "string"},
                cls._attributes_key: {
                    "type": "array",
                    "items": {"$ref": ObjectAttribute.schema_name()}
                },
                cls._bb_key: {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 1,
                    },
                    "minItems": 2,
                    "maxItems": 2,
                },
            },
            "required": [cls._class_id_key, cls._id_key, cls._name_key, cls._bb_key],
            "additionalProperties": False
        }
