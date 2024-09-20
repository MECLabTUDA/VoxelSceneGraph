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
from typing import Any

from typing_extensions import Self

from .KnowledgeComponent import KnowledgeComponent
from ..utils.parsing import *

# We have to cheat slightly to have access to this constant in the JSON schema
_ENUM_ATTR_TYPE = "enum"


class ObjectAttribute(KnowledgeComponent, ABC):
    """
    Component used to define an object class attribute e.g. string enum, int, float.
    WARNING: subclasses need to be registered using register_attribute_type().
    """
    _id_key = "id"
    _name_key = "name"
    _type_key = "type"
    _values_key = "values"  # Key to be used by any attribute type that needs to store additional data
    _registered_attribute_types: dict[str, type] = {}

    def __init__(self, attr_id: int, name: str):
        self.id = attr_id
        self.name = name

    def __init_subclass__(cls, **kwargs):
        """This code automatically registers any subclass that has been initialized."""
        # Note: important to register before super call, as we need to update the schema
        cls._registered_attribute_types[cls.get_attribute_type()] = cls
        super().__init_subclass__(**kwargs)

    @classmethod
    def from_json(cls, json_dict: dict) -> Self:
        attr_id = int(json_dict[cls._id_key])
        attr_name = json_dict[cls._name_key]
        type_val = json_dict[cls._type_key]

        if type_val == EnumAttribute.get_attribute_type():
            return EnumAttribute(attr_id, attr_name, json_dict.get(cls._values_key, []))
        return cls._registered_attribute_types[type_val](attr_id, attr_name)

    def to_json(self) -> dict:
        return {
            self._id_key: self.id,
            self._name_key: self.name,
            self._type_key: self.get_attribute_type()
        }

    @classmethod
    def schema_name(cls) -> str:
        # Ugly... but avoids creating schemas for subclasses
        return "ObjectAttribute"

    @classmethod
    def schema(cls) -> dict:
        return {
            "id": "ObjectAttribute",  # Ugly... but avoids creating schemas for subclasses
            "type": "object",
            "properties": {
                cls._id_key: {"$ref": ID_SCHEMA_NAME},
                cls._name_key: {"type": "string"},
                cls._type_key: {
                    "type": "string",
                    "pattern": list_of_names_to_pattern_filter(cls._registered_attribute_types.keys())
                },
                cls._values_key: {
                    "type": "array",
                    "uniqueItems": True,
                    "minItems": 1
                },
            },
            "required": [cls._id_key, cls._name_key, cls._type_key],
            # Make cls._values_key required if an enum
            "if": {"properties": {cls._type_key: {"const": _ENUM_ATTR_TYPE}}},
            "then": {"required": [cls._values_key]},
            "additionalProperties": False
        }

    @classmethod
    @abstractmethod
    def get_attribute_type(cls) -> str:
        """Returns the str key used to identify the attribute type."""
        raise NotImplementedError

    @abstractmethod
    def validate_and_cast_value_instance(self, value: Any, context_str: str, logger: Logger) -> tuple[Any, bool]:
        """
        Function used to check the type of the value and cast if necessary.
        Used when parsing a scene graph from JSON.
        Returns the value after the cast and whether the cast was successful.
        """
        raise NotImplementedError

    @abstractmethod
    def default_value(self) -> Any:
        """
        Returns a default value.
        Cannot even be a class method as parametrized attributes such as enums need to access to parameters.
        Used for attribute initialization.
        """
        raise NotImplementedError

    # API for torch conversion of attributes
    @staticmethod
    def is_long_tensor_compatible() -> bool:
        """
        Whether the values taken by this attribute class can be serialized to a torch long tensor.
        Note: we cannot have multiple dtypes in the same tensor and having multiple is annoying.
              So we choose long type for classification tasks.
        """
        return False

    def value_instance_to_long(self, value: Any) -> int:
        """Convert a value to an int for long storage (if possible)."""
        return 0


class StrAttribute(ObjectAttribute):
    """Attribute containing a string."""

    @classmethod
    def get_attribute_type(cls) -> str:
        return "str"

    def validate_and_cast_value_instance(self, value: Any, context_str: str, logger: Logger) -> tuple[str, bool]:
        return value, True

    def default_value(self) -> str:
        return ""

    def __repr__(self):
        return f"StrAttribute(id={self.id}, name='{self.name}')"


class IntAttribute(ObjectAttribute):
    """Attribute containing an int."""

    @classmethod
    def get_attribute_type(cls) -> str:
        return "int"

    def validate_and_cast_value_instance(self, value: Any, context_str: str, logger: Logger) -> tuple[int, bool]:
        res = expect_int(value, context_str, logger)
        return res, res is not None

    def default_value(self) -> int:
        return 0

    def __repr__(self):
        return f"IntAttribute(id={self.id}, name='{self.name}')"

    @staticmethod
    def is_long_tensor_compatible() -> bool:
        return True

    def value_instance_to_long(self, value: int) -> int:
        """Convert a value to an int for long storage (if possible)."""
        return int(value)


class FloatAttribute(ObjectAttribute):
    """Attribute containing a float."""

    @classmethod
    def get_attribute_type(cls) -> str:
        return "float"

    def validate_and_cast_value_instance(self, value: Any, context_str: str, logger: Logger) -> tuple[float, bool]:
        res = expect_float(value, context_str, logger)
        return res, res is not None

    def default_value(self) -> float:
        return 0.

    def __repr__(self):
        return f"FloatAttribute(id={self.id}, name='{self.name}')"

    @staticmethod
    def is_long_tensor_compatible() -> bool:
        return False


class BoolAttribute(ObjectAttribute):
    """Attribute containing a bool."""

    @classmethod
    def get_attribute_type(cls) -> str:
        return "bool"

    def validate_and_cast_value_instance(self, value: Any, context_str: str, logger: Logger) -> tuple[bool, bool]:
        res = expect_bool(value, context_str, logger)
        return res, res is not None

    def default_value(self) -> int:
        return False

    def __repr__(self):
        return f"BoolAttribute(id={self.id}, name='{self.name}')"

    @staticmethod
    def is_long_tensor_compatible() -> bool:
        return True

    def value_instance_to_long(self, value: bool) -> int:
        """Convert a value to an int for long storage (if possible)."""
        return int(value)


class EnumAttribute(ObjectAttribute):
    """Attribute containing a value from an enum.The enum cannot be empty."""

    def __init__(self, attr_id: int, name: str, values: list):
        super().__init__(attr_id, name)
        self.values = values

    @classmethod
    def get_attribute_type(cls) -> str:
        return _ENUM_ATTR_TYPE

    def to_json(self) -> dict:
        base_dict = super().to_json()
        base_dict[self._values_key] = self.values
        return base_dict

    def validate_and_cast_value_instance(self, value: Any, context_str: str, logger: Logger) -> tuple[Any, bool]:
        if value not in self.values:
            logger.error(f"{context_str} value {value} is not defined in the knowledge graph.")
            return value, False
        return value, True

    def default_value(self) -> Any:
        # Validated enums should have at least one value
        return self.values[0]

    def __repr__(self):
        return f"EnumAttribute(id={self.id}, name='{self.name}', values={self.values})"

    @staticmethod
    def is_long_tensor_compatible() -> bool:
        return True

    def value_instance_to_long(self, value: Any) -> int:
        """Convert a value to an int for long storage (if possible)."""
        if value not in self.values:
            raise ValueError(
                f"{value} is not part of this {self}. "
                f"Please make sure that you're using the correct knowledge graph."
            )
        return self.values.index(value)
