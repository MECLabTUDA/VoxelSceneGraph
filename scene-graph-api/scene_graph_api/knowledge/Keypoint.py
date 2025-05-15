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

from typing_extensions import Self

from .KnowledgeComponent import KnowledgeComponent
from ..utils.parsing import *


class Keypoint(KnowledgeComponent):
    """Component defining the label id, and name of a keypoint."""
    _id_key = "id"
    _name_key = "name"

    def __init__(
            self,
            class_id: int,
            name: str = ""
    ):
        self.id = class_id
        self.name = name

    @classmethod
    def from_json(cls, json_dict: dict) -> Self:
        return cls(
            class_id=int(json_dict[cls._id_key]),
            name=json_dict[cls._name_key]
        )

    def to_json(self) -> dict:
        return {
            self._id_key: self.id,
            self._name_key: self.name
        }

    def __repr__(self):
        return f"Keypoint(id={self.id}, name='{self.name}')"

    @classmethod
    def schema(cls) -> dict:
        return {
            "id": cls.schema_name(),
            "type": "object",
            "properties": {
                cls._id_key: {"$ref": ID_SCHEMA_NAME},
                cls._name_key: {"type": "string"},
            },
            "required": [cls._id_key, cls._name_key],
            "additionalProperties": False
        }
