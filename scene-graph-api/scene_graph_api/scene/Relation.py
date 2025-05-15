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
from .Keypoint import Keypoint
from .Object import Object
from .SceneGraphComponent import SceneGraphComponent
from ..knowledge import KnowledgeGraph, RelationRule


class Relation(SceneGraphComponent):
    """Relation between two objects in a scene graph."""

    _rule_id_key = "id"
    _subject_id_key = "subject_id"
    _object_id_key = "object_id"
    _common_attributes_key = "common_attributes"
    _common_keypoints_key = "common_keypoints"
    _attributes_key = "attributes"
    _keypoints_key = "keypoints"

    def __init__(
            self,
            rule_id: int,
            subject_id: int,
            object_id: int,
            common_attributes: list[Attribute] | None = None,
            common_keypoints: list[Keypoint] | None = None,
            attributes: list[Attribute] | None = None,
            keypoints: list[Keypoint] | None = None
    ):
        self.rule_id = rule_id
        self.subject_id = subject_id
        self.object_id = object_id
        self.common_attributes = common_attributes if common_attributes is not None else []
        self.common_keypoints = common_keypoints if common_keypoints is not None else []
        self.attributes = attributes if attributes is not None else []
        self.keypoints = keypoints if keypoints is not None else []

    @classmethod
    def from_json(cls, json_dict: dict) -> Self:
        rule_id = int(json_dict[cls._rule_id_key])
        subj_id = int(json_dict[cls._subject_id_key])
        obj_id = int(json_dict[cls._object_id_key])
        common_attrs = [Attribute.from_json(obj_dict) for obj_dict in json_dict.get(cls._common_attributes_key, [])]
        common_kps = [Keypoint.from_json(obj_dict) for obj_dict in json_dict.get(cls._common_keypoints_key, [])]
        attributes = [Attribute.from_json(obj_dict) for obj_dict in json_dict.get(cls._attributes_key, [])]
        keypoints = [Keypoint.from_json(obj_dict) for obj_dict in json_dict.get(cls._keypoints_key, [])]

        return cls(
            rule_id, subj_id, obj_id,
            common_attributes=common_attrs, common_keypoints=common_kps,
            attributes=attributes, keypoints=keypoints
        )

    def to_json(self) -> dict:
        return {
            self._rule_id_key: self.rule_id, self._subject_id_key: self.subject_id, self._object_id_key: self.object_id,
            self._common_attributes_key: [attr.to_json() for attr in self.common_attributes],
            self._common_keypoints_key: [kp.to_json() for kp in self.common_keypoints],
            self._attributes_key: [attr.to_json() for attr in self.attributes],
            self._keypoints_key: [kp.to_json() for kp in self.keypoints],
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

        return self.validate_attributes(rule, graph.rel_common, logger) and success

    def validate_attributes(self, rel_class: RelationRule, rel_common_class: RelationRule, logger: Logger) -> bool:
        """Validate class-common and -specific attributes."""
        context_str = f"In Relation with id {self.rule_id}:"
        success = True

        for attr_set, cur_obj_class, name in [
            [self.common_attributes, rel_common_class, "common attributes"],
            [self.attributes, rel_class, "attributes"],
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

    def validate_keypoint_length(self, volume_dim_cnt: int, logger: Logger) -> bool:
        """Validates that the number of dimensions in keypoints matches the one of the segmentation."""
        success = True

        # Also check the length of any keypoint
        for kp in self.common_keypoints:
            success &= kp.validate(volume_dim_cnt, logger)
        for kp in self.keypoints:
            success &= kp.validate(volume_dim_cnt, logger)

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
        attrs_repr = ",".join(map(repr, self.attributes))
        kps_repr = ",".join(map(repr, self.keypoints))
        return (f"Relation(rule_id={self.rule_id}, subject_id={self.subject_id}, object_id={self.object_id} "
                f"attributes=[{attrs_repr}], keypoints=[{kps_repr}])")

    @classmethod
    def schema(cls) -> dict:
        return {
            "id": cls.schema_name(),
            "type": "object",
            "properties": {
                cls._rule_id_key: {"type": "integer"},
                cls._subject_id_key: {"type": "integer"},
                cls._object_id_key: {"type": "integer"},
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
            "required": [cls._rule_id_key, cls._subject_id_key, cls._object_id_key],
            "additionalProperties": False
        }
