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

from referencing import Registry
from typing_extensions import Self

from ..utils.parsing import ID_SCHEMA_NAME, id_schema, schema_to_resource


class SceneGraphComponent(ABC):
    """
    Interface for scene graph instance components.
    Each object contains minimal information and require the knowledge graph to be fully understood/validated.
    """

    SCHEMA_REGISTRY = Registry().with_resource(ID_SCHEMA_NAME, schema_to_resource(id_schema()))

    def __init_subclass__(cls, **kwargs):
        """This code automatically registers the schema of any subclass."""
        super().__init_subclass__(**kwargs)
        SceneGraphComponent.SCHEMA_REGISTRY = SceneGraphComponent.SCHEMA_REGISTRY.with_resource(
            cls.schema_name(), schema_to_resource(cls.schema())
        )

    @classmethod
    @abstractmethod
    def from_json(cls, json_dict: dict) -> Self:
        """
        Function used to recursively load components.
        The only check that should be performed here is whether each expected class attribute is present
        and of the right type.
        However, if the component could not be constructed, returns None
        :param json_dict: the dict read from file
        :return: the current component being loaded if loading was successful, else None
        """
        raise NotImplementedError

    @abstractmethod
    def to_json(self) -> dict:
        """
        Converts the component back to JSON format.
        :return: json dict
        """
        raise NotImplementedError

    @abstractmethod
    def __repr__(self):
        raise NotImplementedError

    @classmethod
    def schema_name(cls) -> str:
        """Schema name for JSON structure validation."""
        return cls.__name__

    @classmethod
    @abstractmethod
    def schema(cls) -> dict:
        """JSON schema for structure validation."""
        # raise NotImplementedError
        return {}
