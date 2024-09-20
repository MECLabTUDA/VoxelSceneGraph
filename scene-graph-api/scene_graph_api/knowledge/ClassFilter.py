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

from abc import ABC, abstractmethod
from logging import Logger

from typing_extensions import Self

from .KnowledgeComponent import KnowledgeComponent
from ..utils.parsing import *


class ClassFilter(KnowledgeComponent, ABC):
    """
    Component used to determine whether an object class is allowed to be the subject/object of a rule.
    WARNING: subclasses need to be registered using register_filter_type().
    """
    _type_key = "type"
    _objects_classes_key = "classes"
    _registered_filter_types: dict[str, type] = {}

    def __init__(self, object_class_ids: list[int]):
        self.object_class_ids = list(set(object_class_ids))  # Make ids unique

    def __init_subclass__(cls, **kwargs):
        """This code automatically registers any subclass that has been initialized."""
        # Note: important to register before super call, as we need to update the schema
        cls._registered_filter_types[cls.get_filter_type()] = cls
        super().__init_subclass__(**kwargs)

    @classmethod
    def from_json(cls, json_dict: dict) -> Self:
        type_val = json_dict[cls._type_key]
        objects = list(map(int, json_dict[cls._objects_classes_key]))
        return cls._registered_filter_types[type_val](objects)

    def to_json(self) -> dict:
        return {
            self._type_key: self.get_filter_type(),
            self._objects_classes_key: self.object_class_ids
        }

    def validate_references(
            self,
            relation_rule_id: int,
            known_object_classes: list[int],
            logger: Logger
    ) -> bool:
        context_str = f"{self.get_filter_type()} of rule with id {relation_rule_id}:"
        success = True
        for class_id in self.object_class_ids:
            if class_id not in known_object_classes:
                logger.error(f"{context_str} Unknown object class id {class_id}.")
                success = False
        return success

    @classmethod
    def schema_name(cls) -> str:
        # Ugly... but avoids creating schemas for subclasses
        return "ClassFilter"

    @classmethod
    def schema(cls) -> dict:
        return {
            "id": "ClassFilter",  # Ugly... but avoids creating schemas for subclasses
            "type": "object",
            "properties": {
                cls._type_key: {
                    "type": "string",
                    "pattern": list_of_names_to_pattern_filter(cls._registered_filter_types.keys())
                },
                cls._objects_classes_key: {
                    "type": "array",
                    "uniqueItems": True,
                    "items": {"$ref": ID_SCHEMA_NAME}
                },
            },
            "required": [cls._type_key, cls._objects_classes_key],
            "additionalProperties": False
        }

    @classmethod
    @abstractmethod
    def get_filter_type(cls) -> str:
        """Returns the str key used to identify the filter type."""
        raise NotImplementedError

    @abstractmethod
    def get_authorized_classes(self, known_object_classes: list[int]) -> list[int]:
        """
        Given the list of all known object class ids,
        returns the list of authorized object class ids based on the filter type and the memorized ids.
        :param known_object_classes: the list of known object class ids
        :return: a list of authorized object class ids
        """
        raise NotImplementedError

    @abstractmethod
    def is_class_authorized(self, class_id: int):
        """Returns whether a specific class id is authorized."""
        raise NotImplementedError


class WhitelistFilter(ClassFilter):
    """Whitelist filter i.e. only ids in self._object_class_ids are authorized."""

    @classmethod
    def get_filter_type(cls) -> str:
        return "whitelist"

    def validate_references(
            self,
            relation_rule_id: int,
            known_object_classes: list[int],
            logger: Logger
    ) -> bool:
        """Also checks that a whitelist filter does not have an empty filter."""
        success = super().validate_references(relation_rule_id, known_object_classes, logger)

        context_str = f"{self.get_filter_type()} of rule with id {relation_rule_id}:"
        if not self.object_class_ids:
            logger.warning(f"{context_str} empty object class filter list.")

        return success

    def get_authorized_classes(self, known_object_classes: list[int]) -> list[int]:
        return self.object_class_ids.copy()

    def is_class_authorized(self, class_id: int):
        return class_id in self.object_class_ids

    def __repr__(self):
        return f"WhitelistFilter(object_class_ids={self.object_class_ids})"


class BlacklistFilter(ClassFilter):
    """Blacklist filter i.e. only ids NOT in self._object_class_ids are authorized."""

    @classmethod
    def get_filter_type(cls) -> str:
        return "blacklist"

    def get_authorized_classes(self, known_object_classes: list[int]) -> list[int]:
        return [class_id for class_id in known_object_classes if class_id not in self.object_class_ids]

    def is_class_authorized(self, class_id: int):
        return class_id not in self.object_class_ids

    def __repr__(self):
        return f"BlacklistFilter(object_class_ids={self.object_class_ids})"
