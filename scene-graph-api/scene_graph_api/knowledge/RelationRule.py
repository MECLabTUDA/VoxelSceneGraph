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

from logging import Logger

from typing_extensions import Self

from .ClassFilter import ClassFilter, BlacklistFilter
from .KnowledgeComponent import KnowledgeComponent
from ..utils.parsing import *


class RelationRule(KnowledgeComponent):
    """
    Component defining a rule i.e. subject, object and class filter.
    Note: the id should be at least 1, as 0 is reserved for the background.
    """

    _name_key = "name"
    _id_key = "id"
    _subject_key = "subject_filter"
    _object_key = "object_filter"

    def __init__(
            self,
            rule_id: int,
            name: str = "",
            subject_filter: ClassFilter | None = None,
            object_filter: ClassFilter | None = None
    ):
        self.id = rule_id
        self.name = name
        self.subject_filter = subject_filter if subject_filter is not None else BlacklistFilter([])
        self.object_filter = object_filter if object_filter is not None else BlacklistFilter([])

    @classmethod
    def from_json(cls, json_dict: dict) -> Self:
        rule_id = int(json_dict[cls._id_key])
        rule_name = json_dict[cls._name_key]
        subject_filter = ClassFilter.from_json(json_dict[cls._subject_key])
        object_filter = ClassFilter.from_json(json_dict[cls._object_key])

        return cls(rule_id, rule_name, subject_filter, object_filter)

    def to_json(self) -> dict:
        return {
            self._id_key: self.id,
            self._name_key: self.name,
            self._subject_key: self.subject_filter.to_json(),
            self._object_key: self.object_filter.to_json()
        }

    def validate_references(self, known_object_classes: list[int], logger: Logger) -> bool:
        """Checks that all object classes referenced are defined."""
        success = self.subject_filter.validate_references(1, known_object_classes, logger)
        success &= self.object_filter.validate_references(1, known_object_classes, logger)
        return success

    def __repr__(self):
        return f"RelationRule(id={self.id}, name='{self.name}', " \
               f"subject_filter={self.subject_filter}, object_filter={self.object_filter})"

    @classmethod
    def schema(cls) -> dict:
        return {
            "id": cls.schema_name(),
            "type": "object",
            "properties": {
                cls._id_key: {"$ref": ID_SCHEMA_NAME},
                cls._name_key: {"type": "string"},
                cls._subject_key: {"$ref": ClassFilter.schema_name()},
                cls._object_key: {"$ref": ClassFilter.schema_name()},
            },
            "required": [cls._id_key, cls._name_key, cls._subject_key, cls._object_key],
            "additionalProperties": False
        }
