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

import re
from logging import Logger

import numpy as np
from typing_extensions import Self

from .KnowledgeComponent import KnowledgeComponent
from .ObjectAttribute import ObjectAttribute
from ..utils.parsing import *


def _id_to_hex_color(some_id: int) -> str:
    """
    Given some id (e.g. object class id) returns a color in hex format.
    Returns a color from the color palette if some_id < length(palette).
    Else returns a random color.
    """
    # Default palette for up to 10 object classes
    # https://www.color-hex.com/color-palette/1018386 and https://www.color-hex.com/color-palette/1018399
    # noinspection SpellCheckingInspection
    color_palette = [
        "#ffa602", "#ff6362", "#bc5090", "#58508d", "#50868d",
        "#cc3f0c", "#9a6a36", "#336737", "#0d2b0d", "#c1babd"
    ]

    if some_id < len(color_palette):
        return color_palette[some_id]
    return "#" + "".join(np.random.choice(list("0123456789abcdefABCDEF"), 6))


class ObjectClass(KnowledgeComponent):
    """
    Component defining the label id, name, attributes... of a target object class.
    An object class can either be of a segmented object or of a bounding box obtained from a coarse segmentation.

    Optionally, one can use the "is_unique" flag to state that
    there can only be UP TO one object of this class per graph.
    SO if true, the connected components of the segmentation mask for this class will be disregarded and
    only one object will be created. This can be useful for instance for partially occluded anatomical structure.

    The "has_mask" flag declares whether in the initial annotation (bounding boxes or segmentation...),
    this object class has a mask. It of course must have one when creating a Scene Graph from mask-based data.
    But assuming that the input is bounding boxes only, then a rectangular mask will be generated.

    The color is the one used to draw segmentations and bounding boxes and is stored in hex format as a string.
    Name, attributes, is_bounding_box, is_unique, and color are optional.
    """
    _id_key = "id"
    _name_key = "name"
    _attributes_key = "attributes"
    _has_mask_key = "has_mask"
    _is_ignored_key = "is_ignored"
    _is_unique_key = "is_unique"
    _color_key = "color"

    def __init__(
            self,
            class_id: int,
            name: str = "",
            attributes: list[ObjectAttribute] | None = None,
            has_mask: bool = False,
            is_unique: bool = False,
            is_ignored: bool = False,
            color: str | None = None
    ):
        self.id = class_id
        self.name = name
        self.attributes = attributes if attributes is not None else []
        self.has_mask = has_mask
        self.is_unique = is_unique
        self.is_ignored = is_ignored
        self.color = color if color is not None else _id_to_hex_color(class_id - 1)

    @classmethod
    def from_json(cls, json_dict: dict) -> Self:
        class_id = int(json_dict[cls._id_key])
        name = json_dict[cls._name_key]
        has_mask = json_dict.get(cls._has_mask_key, False)
        is_unique = json_dict.get(cls._is_unique_key, False)
        is_ignored = json_dict.get(cls._is_ignored_key, False)
        color = json_dict.get(cls._color_key)
        attributes = [ObjectAttribute.from_json(obj_dict) for obj_dict in json_dict.get(cls._attributes_key, [])]

        return cls(
            class_id=class_id,
            name=name,
            attributes=attributes,
            has_mask=has_mask,
            is_unique=is_unique,
            is_ignored=is_ignored,
            color=color
        )

    def to_json(self) -> dict:
        return {
            self._id_key: self.id,
            self._name_key: self.name,
            self._attributes_key: [attr.to_json() for attr in self.attributes],
            self._has_mask_key: self.has_mask,
            self._is_unique_key: self.is_unique,
            self._is_ignored_key: self.is_ignored,
            self._color_key: self.color
        }

    def validate_attributes(self, logger: Logger) -> bool:
        """Validates that the object attribute definitions are valid."""
        context_str = f"of Object Class ({self.id})"

        # Check attribute id and name unicity
        success = check_list_unicity([a.id for a in self.attributes], logger, f"Attribute Id {context_str}")
        success &= check_list_unicity([a.name for a in self.attributes], logger, f"Attribute Name {context_str}",
                                      warn_only=True)

        # Check that the hex color is valid
        if not bool(re.match(r"^#[\da-fA-F]{6}$", self.color)):
            logger.error(f"{context_str} color \"{self.color}\" is not a valid hex color.")
            return False

        return success

    def get_attribute_by_id(self, attr_id: int) -> ObjectAttribute | None:
        """Returns the ObjectAttribute corresponding to the given id if found."""
        for attr in self.attributes:
            if attr.id == attr_id:
                return attr

    def __repr__(self):
        attrs_repr = ",".join(map(repr, self.attributes))
        return (f"ObjectClass(id={self.id}, name='{self.name}', attributes=[{attrs_repr}], has_mask={self.has_mask}, "
                f"is_unique={self.is_unique}, is_ignored={self.is_ignored}, color={self.color})")

    @classmethod
    def schema(cls) -> dict:
        return {
            "id": cls.schema_name(),
            "type": "object",
            "properties": {
                cls._id_key: {"$ref": ID_SCHEMA_NAME},
                cls._name_key: {"type": "string"},
                cls._attributes_key: {
                    "type": "array",
                    "items": {"$ref": ObjectAttribute.schema_name()}
                },
                cls._has_mask_key: {"type": "boolean"},
                cls._is_unique_key: {"type": "boolean"},
                cls._is_ignored_key: {"type": "boolean"},
                cls._color_key: {"type": "string", "pattern": r"^#[\da-fA-F]{6}$"},
            },
            "required": [cls._id_key, cls._name_key],
            "additionalProperties": False
        }
