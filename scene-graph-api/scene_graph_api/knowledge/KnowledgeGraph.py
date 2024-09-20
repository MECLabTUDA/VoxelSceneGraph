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

import json
from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum
from logging import Logger
from os import PathLike
from pathlib import Path
from typing import Type

from typing_extensions import Self

from .KnowledgeComponent import KnowledgeComponent
from .ObjectAttribute import ObjectAttribute
from .ObjectClass import ObjectClass
from .RelationRule import RelationRule
from ..utils.indexing import contiguous_mapping
from ..utils.parsing import *


class KnowledgeGraph(KnowledgeComponent, ABC):
    """
    Root component on any knowledge graph.
    The type of the knowledge graph defines which kind of image will be annotated,
    e.g. radiology (Nifti, ...) bs natural images( .png, .jpg...). Each type may add additional fields.
    Contains a list of object classes and a list of rules, defining the overall knowledge graph.
    Rules are optional as one might want to use this software to only do attribute annotation.
    Additionally, a graph can be identified by its hash (or any identifying string).
    -> This is particularly useful to check which graph was used to annotate a specific scene graph.
    -> Hashes are computed when loading from JSON and updated when saving to JSON.
    """
    # TODO: attributes that are common to all object classes
    _object_classes_key = "classes"
    _rules_key = "rules"
    _image_level_attributes_key = "image"
    _type_key = "type"
    _hash_key = "hash"
    registered_types: dict[str, Type[KnowledgeGraph]] = {}

    def __init_subclass__(cls, **kwargs):
        """This code automatically registers any subclass that has been initialized."""
        # Note: important to register before super call, as we need to update the schema
        cls.registered_types[cls.get_graph_type()] = cls
        super().__init_subclass__(**kwargs)

    def __init__(
            self,
            classes: list[ObjectClass],
            rules: list[RelationRule] | None = None,
            image_level_attributes: list[ObjectAttribute] | None = None,
            hash_: int = 0
    ):
        self.classes = classes
        self.rules = rules if rules is not None else []
        self.image = ObjectClass(0, "Background", attributes=image_level_attributes)
        self.hash = hash_

    @classmethod
    def from_json(cls, json_dict: dict) -> Self:
        type_val = json_dict[cls._type_key]
        object_classes = [ObjectClass.from_json(obj_dict) for obj_dict in json_dict[cls._object_classes_key]]
        rules = [RelationRule.from_json(obj_dict) for obj_dict in json_dict.get(cls._rules_key, [])]
        image_level_attributes = [ObjectAttribute.from_json(obj_dict)
                                  for obj_dict in json_dict.get(cls._image_level_attributes_key, [])]
        hash_val = int(json_dict.get(cls._hash_key, hash(frozenset(json_dict))))

        graph = cls.registered_types[type_val](object_classes, rules, image_level_attributes, hash_val)
        graph._load_additional_fields(json_dict)

        return graph

    def to_json(self) -> dict:
        json_dict = {
            self._type_key: self.get_graph_type(),
            self._object_classes_key: [obj_class.to_json() for obj_class in self.classes],
            self._image_level_attributes_key: [attr.to_json() for attr in self.image.attributes],
            self._rules_key: [rule.to_json() for rule in self.rules]
        }
        self._save_additional_fields(json_dict)
        json_dict[self._hash_key] = hash(frozenset(json_dict))
        return json_dict

    def validate(self, logger: Logger) -> bool:
        """
        Checks that every object class reference is valid.
        These are present in relation rules and class filters.
        """

        success = True
        known_object_classes = [a.id for a in self.classes]

        # Validate object attributes
        for obj_class in self.classes:
            success &= obj_class.validate_attributes(logger)
        # Check that object class ids and names are unique
        success &= check_list_unicity(known_object_classes, logger, "Object Class id")
        success &= check_list_unicity([a.name for a in self.classes], logger, "Object Class name",
                                      warn_only=True)
        # Validate rules
        for rule in self.rules:
            success &= rule.validate_references(known_object_classes, logger)
        # Check that rule ids and names are unique
        success &= check_list_unicity([a.id for a in self.rules], logger, "Rule id")
        success &= check_list_unicity([a.name for a in self.rules], logger, "Rule name", warn_only=True)

        # Validate image level attributes
        success &= self.image.validate_attributes(logger)

        return success

    def save(self, path: str | PathLike) -> bool:
        """
        Save as a json file at given destination path.
        Note: automatically remaps as contiguous indexing.
        :returns: success.
        """
        try:
            with open(path, "w") as f:
                # TODO we don't always want that since the annotation data is not perfect
                #  We only want to do that after the Scene Graphs are created
                self.remap_ids_as_contiguous()
                json.dump(self.to_json(), f)
            return True
        except FileNotFoundError:
            return False

    @classmethod
    def load(cls, path: str | PathLike, logger: Logger) -> Self | None:
        """
        Loads the knowledge graph from the source path and validates the content.
        Only returns the graph if it is valid.
        """
        obj_dict = load_json_from_path(path, Path(path).as_posix() + ":", logger)
        if obj_dict is not None:
            validator = get_validator(cls.schema(), registry=KnowledgeComponent.SCHEMA_REGISTRY)
            errors = list(validator.iter_errors(obj_dict))
            if errors:
                for error in errors:
                    logger.error(error.message)
                return

            graph = KnowledgeGraph.from_json(obj_dict)
            if graph.validate(logger):
                return graph

    def get_object_class_by_id(self, class_id: int) -> ObjectClass | None:
        """Return the ObjectClass corresponding to the given id if found."""
        for obj_class in self.classes:
            if obj_class.id == class_id:
                return obj_class

    def get_rule_by_id(self, rule_id: int) -> RelationRule | None:
        """Return the RelationRule corresponding to the given id if found."""
        for rule in self.rules:
            if rule.id == rule_id:
                return rule

    def get_valid_rule_ids(self, subject_class_id: int, object_class_id: int) -> list[RelationRule]:
        """Given subject and object class ids, returns the list of all possible relation rules for the two objects."""
        possible_rules = []
        for rule in self.rules:
            if rule.subject_filter.is_class_authorized(subject_class_id) and \
                    rule.object_filter.is_class_authorized(object_class_id):
                possible_rules.append(rule)
        return possible_rules

    def remap_ids_as_contiguous(self) -> Self:
        """
        Remap all ids (in place) such they are contiguous. ObjectClasses and Rule ids start at 1.
        Note: object classes that are flagged as ignored are moved to the tail of the mapping.
        """
        # Class ids
        # Note: before we listed the classes in one go, so the output list of class ids was sorted.
        #       now that ignored classes are put last we need to do a final sort.
        all_class_ids_not_ignored = [obj_class.id for obj_class in self.classes if not obj_class.is_ignored]
        all_class_ids_ignored = [obj_class.id for obj_class in self.classes if obj_class.is_ignored]
        all_class_ids = all_class_ids_not_ignored + all_class_ids_ignored
        class_mapping = contiguous_mapping(all_class_ids, start=1)
        if class_mapping:
            # Only do the processing if there is something to do
            # Remap class ids
            for obj_class in self.classes:
                obj_class.id = class_mapping.get(obj_class.id, obj_class.id)
            self.classes.sort(key=lambda o: o.id)
            # Remap class ids in rule filters
            for rule in self.rules:
                for filter_ in [rule.subject_filter, rule.object_filter]:
                    filter_.object_class_ids = list(map(lambda i: class_mapping.get(i, i), filter_.object_class_ids))

        # Attribute ids
        for obj_class in self.classes:
            if not obj_class.attributes:
                continue
            for new_id, attr in enumerate(obj_class.attributes, 1):
                attr.id = new_id

        # Rule ids
        all_rule_ids = [rule.id for rule in self.rules]
        rule_mapping = contiguous_mapping(all_rule_ids, start=1)
        if rule_mapping:
            # Only do the processing if there is something to do
            # Remap rule ids
            for rule in self.rules:
                rule.id = rule_mapping.get(rule.id, rule.id)

        # Image level attribute ids
        if self.image.attributes:
            for new_id, attr in enumerate(self.image.attributes, 1):
                attr.id = new_id

        return self

    def __repr__(self):
        classes_repr = ",".join(map(repr, self.classes))
        rules_repr = ",".join(map(repr, self.rules))
        image_attr_repr = ",".join(map(repr, self.image.attributes))
        return (f"{type(self).__name__}(classes=[{classes_repr}],rules=[{rules_repr}],"
                f"image_level_attributes=({image_attr_repr}),hash={self.hash})")

    def copy(self) -> Self:
        return deepcopy(self)

    @classmethod
    def schema(cls) -> dict:
        # Construct a list of all possible constraints stemming from subtypes
        all_constraints = [
            graph_type.get_additional_schema_constraints()
            for graph_type in cls.registered_types.values()
        ]

        return {
            "id": "KnowledgeGraph",  # Ugly... but avoids creating schemas for subclasses
            "type": "object",
            "properties": {
                cls._type_key: {
                    "type": "string",
                    "pattern": list_of_names_to_pattern_filter(cls.registered_types.keys())
                },
                cls._object_classes_key: {"type": "array", "items": {"$ref": ObjectClass.schema_name()}},
                cls._rules_key: {"type": "array", "items": {"$ref": RelationRule.schema_name()}},
                cls._image_level_attributes_key: {"type": "array", "items": {"$ref": ObjectAttribute.schema_name()}},
                cls._hash_key: {"type": "integer"}
            },
            "required": [cls._type_key],
            "allOf": [c for c in all_constraints if c is not None],
            # Note: we need to drop this constraint as otherwise,
            # all class-specific fields are marked as disallowed when even one does not fit the constraints
            # "unevaluatedProperties": False
        }

    # Methods related to the data loading, that can be overwritten

    @classmethod
    @abstractmethod
    def get_graph_type(cls) -> str:
        """Returns the str key used to identify the knowledge graph type."""
        raise NotImplementedError

    @classmethod
    def get_additional_schema_constraints(cls) -> None | dict:
        """Return any additional type-specific constraints for the JSON schema or None."""
        return None

    def _load_additional_fields(self, json_dict: dict):
        """
        If a subclass has additional fields in the JSON schema,
        it should overwrite this method to set the appropriate member values.
        """
        pass

    def _save_additional_fields(self, json_dict: dict):
        """
        If a subclass has additional fields in the JSON schema,
        it should overwrite this method to save the appropriate values to JSON.
        """
        pass

    def to_coco(self):
        """
        Convert this graph to a COCO-compatible description.
        I.e. here we return a pycocotools3d.coco.abstractions.SSGDataset with empty images and annotations.
        """
        from pycocotools3d.coco.abstractions.relation_detection import SSGDataset
        from datetime import datetime

        # TODO ids as contiguous
        # TODO support attributes
        # TODO support image-level annotation

        # Suppress any additional information
        coco_format: SSGDataset = {
            "info": {
                "year": datetime.now().year,
                "version": str(self.hash),
                "description": "",
                "contributor": "",
                "url": "",
                "date_created": datetime.now().strftime("%Y-%m-%d"),
            },
            "categories": [
                {"id": obj_class.id, "name": obj_class.name}
                # FIXME why is the category order necessary for COCO evaluation?
                for obj_class in sorted(self.classes, key=lambda oc: oc.id)
            ],
            "images": [],
            "annotations": [],
            "predicates": [{"id": rule.id, "name": rule.name} for rule in self.rules],
            "relations": [],
        }

        return coco_format


class _ImageAxis(Enum):
    # Note: it's important that the name of the field remains the same as its value. Makes easier to cast from str.
    axial = "axial"
    coronal = "coronal"
    sagittal = "sagittal"

    def to_axis(self) -> int:
        """Map to the corresponding axis for depth-first arrays."""
        match self:
            case self.axial:
                return 0
            case self.coronal:
                return 1
            case self.sagittal:
                return 2


class RadiologyImageKG(KnowledgeGraph):
    """
    Knowledge graph for radiology images i.e. Nifti images.
    Contains additional fields for used to display an image correctly (window center and width).
    """

    ImageAxis = _ImageAxis

    _window_center_key = "window_center"
    _window_width_key = "window_width"
    _default_axis_key = "default_axis"

    def __init__(
            self,
            classes: list[ObjectClass],
            rules: list[RelationRule] | None = None,
            image_level_attributes: list[ObjectAttribute] | None = None,
            hash_: int = 0,
            window_center: int = 50,  # Used for display
            window_width: int = 100,  # Used for display
            default_axis: _ImageAxis = _ImageAxis.axial  # Used for display
    ):
        super().__init__(classes, rules, image_level_attributes, hash_)
        # Used for display
        self.window_center = window_center
        self.window_width = window_width
        self.default_axis = default_axis

    @classmethod
    def get_graph_type(cls) -> str:
        return "radiology"

    @classmethod
    def get_additional_schema_constraints(cls) -> None | dict:
        """Return any additional type-specific constraints for the JSON schema or None."""
        # noinspection PyProtectedMember
        return {
            "if": {"properties": {cls._type_key: {"const": cls.get_graph_type()}}},
            "then": {
                "properties": {
                    cls._window_center_key: {"type": "integer", "minimum": 1},
                    cls._window_width_key: {"type": "integer", "minimum": 1},
                    cls._default_axis_key: {
                        "type": "string",
                        "pattern": list_of_names_to_pattern_filter(_ImageAxis._member_map_)
                    },
                },
                "required": [cls._window_center_key, cls._window_width_key]
            }
        }

    def _load_additional_fields(self, json_dict: dict):
        self.window_center = int(json_dict.get(self._window_center_key, self.window_center))
        self.window_width = int(json_dict.get(self._window_width_key, self.window_width))
        self.default_axis = _ImageAxis[json_dict.get(self._default_axis_key, self.default_axis.value)]

    def _save_additional_fields(self, json_dict: dict):
        json_dict[self._window_center_key] = self.window_center
        json_dict[self._window_width_key] = self.window_width
        json_dict[self._default_axis_key] = self.default_axis.value


class NaturalImageKG(KnowledgeGraph):
    """Knowledge graph for natural images i.e. .png, .jpg, ..."""

    @classmethod
    def get_graph_type(cls) -> str:
        return "natural"
