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

from .Attribute import Attribute
from .ClassFilter import ClassFilter, BlacklistFilter
from .Keypoint import Keypoint
from .KnowledgeComponent import KnowledgeComponent
from ..utils.parsing import *


class RelationRule(KnowledgeComponent):
    """
    Component defining a rule i.e. subject, object and class filter. Eventually also attributes or keypoints!
    Note: the id should be at least 1, as 0 is reserved for the background.
    """

    _name_key = "name"
    _id_key = "id"
    _subject_key = "subject_filter"
    _object_key = "object_filter"
    _attributes_key = "attributes"
    _keypoints_key = "keypoints"

    def __init__(
            self,
            rule_id: int,
            name: str = "",
            subject_filter: ClassFilter | None = None,
            object_filter: ClassFilter | None = None,
            attributes: list[Attribute] | None = None,
            keypoints: list[Keypoint] | None = None
    ):
        self.id = rule_id
        self.name = name
        self.subject_filter = subject_filter if subject_filter is not None else BlacklistFilter([])
        self.object_filter = object_filter if object_filter is not None else BlacklistFilter([])
        self.attributes = attributes if attributes is not None else []
        self.keypoints = keypoints if keypoints is not None else []

    @classmethod
    def from_json(cls, json_dict: dict) -> Self:
        rule_id = int(json_dict[cls._id_key])
        rule_name = json_dict[cls._name_key]
        subject_filter = ClassFilter.from_json(json_dict[cls._subject_key])
        object_filter = ClassFilter.from_json(json_dict[cls._object_key])
        attributes = [Attribute.from_json(obj_dict) for obj_dict in json_dict.get(cls._attributes_key, [])]
        keypoints = [Keypoint.from_json(obj_dict) for obj_dict in json_dict.get(cls._keypoints_key, [])]

        return cls(rule_id, rule_name, subject_filter, object_filter, attributes=attributes, keypoints=keypoints)

    def to_json(self) -> dict:
        return {
            self._id_key: self.id,
            self._name_key: self.name,
            self._subject_key: self.subject_filter.to_json(),
            self._object_key: self.object_filter.to_json(),
            self._attributes_key: [attr.to_json() for attr in self.attributes],
            self._keypoints_key: [kp.to_json() for kp in self.keypoints],
        }

    def validate_references(self, known_object_classes: list[int], logger: Logger) -> bool:
        """Validates that all object classes referenced are defined."""
        success = self.subject_filter.validate_references(1, known_object_classes, logger)
        success &= self.object_filter.validate_references(1, known_object_classes, logger)
        return success

    def validate_attributes(self, logger: Logger) -> bool:
        """Validates that the relation attribute definitions are valid."""
        context_str = f"of Relation Rule ({self.id})"

        # Check attribute id and name unicity
        success = check_list_unicity([a.id for a in self.attributes], logger, f"Attribute Id {context_str}")
        success &= check_list_unicity(
            [a.name for a in self.attributes],
            logger,
            f"Attribute Name {context_str}",
            warn_only=True
        )

        return success

    def validate_keypoints(self, logger: Logger) -> bool:
        """Validates that the relation keypoint definitions are valid."""
        context_str = f"of Relation Rule ({self.id})"

        # Check keypoint id and name unicity
        success = check_list_unicity([a.id for a in self.keypoints], logger, f"Keypoint Id {context_str}")
        success &= check_list_unicity(
            [a.name for a in self.keypoints],
            logger,
            f"Keypoint Name {context_str}",
            warn_only=True
        )

        return success

    def __repr__(self):
        attrs_repr = ",".join(map(repr, self.attributes))
        kps_repr = ",".join(map(repr, self.keypoints))
        return (f"RelationRule(id={self.id}, name='{self.name}', "
                f"subject_filter={self.subject_filter}, object_filter={self.object_filter}, "
                f"attributes=[{attrs_repr}], keypoints=[{kps_repr}])")

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
                cls._attributes_key: {
                    "type": "array",
                    "items": {"$ref": Attribute.schema_name()}
                },
                cls._keypoints_key: {
                    "type": "array",
                    "items": {"$ref": Keypoint.schema_name()}
                },
            },
            "required": [cls._id_key, cls._name_key, cls._subject_key, cls._object_key],
            "additionalProperties": False
        }
