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

from .Object import Object
from .SceneGraphComponent import SceneGraphComponent
from ..knowledge import KnowledgeGraph


class Relation(SceneGraphComponent):
    """Relation between two objects in a scene graph."""

    _rule_id_key = "id"
    _subject_id_key = "subject_id"
    _object_id_key = "object_id"

    def __init__(self, rule_id: int, subject_id: int, object_id: int):
        self.rule_id = rule_id
        self.subject_id = subject_id
        self.object_id = object_id

    @classmethod
    def from_json(cls, json_dict: dict) -> Self:
        rule_id = int(json_dict[cls._rule_id_key])
        subj_id = int(json_dict[cls._subject_id_key])
        obj_id = int(json_dict[cls._object_id_key])

        return cls(rule_id, subj_id, obj_id)

    def to_json(self) -> dict:
        return {
            self._rule_id_key: self.rule_id,
            self._subject_id_key: self.subject_id,
            self._object_id_key: self.object_id,
        }

    def validate_references(self, graph: KnowledgeGraph, object_list: list[Object], logger: Logger) -> bool:
        """Checks that subject, and object ids are known and that their object class fits the relation rule."""
        context_str = f"In Relation with rule id {self.rule_id}:"
        # Check that the relation id matches a rule in the knowledge graph
        rule = graph.get_rule_by_id(self.rule_id)
        if rule is None:
            logger.error(f"{context_str} Relation does not exist in the knowledge graph.")
            return False

        # Check that subject ids are known
        subject_inst = object_inst = None
        for obj in object_list:
            if obj.id == self.subject_id:
                subject_inst = obj
            # No elif in the case that object == subject
            if obj.id == self.object_id:
                object_inst = obj
            if subject_inst is not None and object_inst is not None:
                break

        success = True
        if subject_inst is None:
            logger.error(f"{context_str} Subject with id {self.subject_id} does not match any object id.")
            success = False
        else:
            # Check that subject type matches filter
            if not rule.subject_filter.is_class_authorized(subject_inst.class_id):
                logger.error(f"{context_str} According to the knowledge graph, "
                             f"an object of class id {subject_inst.class_id} "
                             f"cannot be the subject of a relation of rule id {rule.id}")
                success = False

        if object_inst is None:
            logger.error(f"{context_str} Object with id {self.object_id} does not match any object id.")
            success = False
        else:
            # Check that object type matches filter
            if not rule.object_filter.is_class_authorized(object_inst.class_id):
                logger.error(f"{context_str} According to the knowledge graph, "
                             f"an object of class id {object_inst.class_id} "
                             f"cannot be the object of a relation of rule id {rule.id}")
                success = False

        return success

    def __repr__(self):
        return f"Relation(rule_id={self.rule_id}, subject_id={self.subject_id}, object_id={self.object_id})"

    @classmethod
    def schema(cls) -> dict:
        return {
            "id": cls.schema_name(),
            "type": "object",
            "properties": {
                cls._rule_id_key: {"type": "integer"},
                cls._subject_id_key: {"type": "integer"},
                cls._object_id_key: {"type": "integer"},
            },
            "required": [cls._rule_id_key, cls._subject_id_key, cls._object_id_key],
            "additionalProperties": False
        }
