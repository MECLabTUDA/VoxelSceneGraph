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
from typing import Any

from typing_extensions import Self

from .SceneGraphComponent import SceneGraphComponent
from ..knowledge import ObjectClass
from ..utils.parsing import *


class ObjectAttribute(SceneGraphComponent, ABC):
    """Object attribute instance in a scene graph."""
    _id_key = "id"
    _value_key = "value"

    def __init__(self, attr_id: int, value: Any):
        self.id = attr_id
        self.value = value

    @classmethod
    def from_json(cls, json_dict: dict) -> Self:
        attr_id = int(json_dict[cls._id_key])
        value = json_dict[cls._value_key]
        return cls(attr_id, value)

    def to_json(self) -> dict:
        return {
            self._id_key: self.id,
            # Store value and not idx in enum to avoid issues when reordering occurs in the knowledge graph
            self._value_key: self.value
        }

    def validate_type_and_value(self, parent_obj_class: ObjectClass, logger: Logger) -> bool:
        """Checks that the attribute id and values match the content of the parent object class."""
        context_str = f"In Attribute with id {self.id}:"
        # Check that attribute is defined in object class
        class_attr = parent_obj_class.get_attribute_by_id(self.id)
        if class_attr is None:
            logger.error(f"{context_str} Attribute does not exist in the knowledge graph."
                         f"for object class id {parent_obj_class.id}")
            return False

        # Check that value type is coherent with the knowledge graph
        self.value, success = class_attr.validate_and_cast_value_instance(self.value, context_str, logger)
        return success

    def copy(self):
        """
        Returns a deep copy of the attribute.
        Used when merging objects to avoid modifications of the base objects.
        """
        return type(self)(self.id, self.value)

    def __repr__(self):
        value_str = f"'{self.value}'" if isinstance(self.value, str) else str(self.value)
        return f"ObjectAttribute(id={self.id}, value={value_str})"

    @classmethod
    def schema(cls) -> dict:
        return {
            "id": "ClassFilter",  # Ugly... but avoids creating schemas for subclasses
            "type": "object",
            "properties": {
                cls._id_key: {"$ref": ID_SCHEMA_NAME},
                cls._value_key: {},
            },
            "required": [cls._id_key, cls._value_key],
            "additionalProperties": False
        }
