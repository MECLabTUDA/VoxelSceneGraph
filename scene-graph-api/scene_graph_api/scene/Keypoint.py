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

from .SceneGraphComponent import SceneGraphComponent
from ..utils.parsing import *


class Keypoint(SceneGraphComponent, ABC):
    """A keypoint instance in a scene graph."""
    _id_key = "id"
    _value_key = "value"

    def __init__(self, kp_id: int, value: list[float] | tuple[float, ...]):
        self.id = kp_id
        self.value = tuple(map(float, value))

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

    def validate(self, volume_dim_cnt: int, logger: Logger) -> bool:
        """Checks that the attribute id and values match the content of the parent object class."""
        context_str = f"In Keypoint with id {self.id}:"
        # Check that the keypoint length is correct
        if len(self.value) != volume_dim_cnt:
            logger.error(f"{context_str} invalid keypoint, it "
                         f"does not have as many coordinates ({len(self.value)}) as "
                         f"there are dimensions in the target volume ({volume_dim_cnt}).")
            return False
        return True

    def copy(self):
        """
        Returns a deep copy of the attribute.
        Used when merging objects to avoid modifications of the base objects.
        """
        return type(self)(self.id, self.value)

    def __repr__(self):
        return f"Keypoint(id={self.id}, value={self.value})"

    @classmethod
    def schema(cls) -> dict:
        return {
            "id": cls.schema_name(),
            "type": "object",
            "properties": {
                cls._id_key: {"$ref": ID_SCHEMA_NAME},
                cls._value_key: {"type": "array", "items": {"type": "number"}},
            },
            "required": [cls._id_key, cls._value_key],
            "additionalProperties": False
        }
