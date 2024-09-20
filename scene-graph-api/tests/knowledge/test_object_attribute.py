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

import logging
from unittest import TestCase

from scene_graph_api.knowledge.KnowledgeComponent import KnowledgeComponent
from scene_graph_api.knowledge.ObjectAttribute import *
from scene_graph_api.logging_handlers import TestingHandler
from scene_graph_api.utils.parsing import get_validator


class TestObjectAttribute(TestCase):
    logger = logging.getLogger("knowledge/ObjectAttribute")
    handler = TestingHandler()
    validator = get_validator(ObjectAttribute.schema(), registry=KnowledgeComponent.SCHEMA_REGISTRY)

    @classmethod
    def setUpClass(cls):
        cls.logger.addHandler(cls.handler)

    def setUp(self):
        self.handler.purge()

    def test_from_json_valid(self):
        json_dict = {ObjectAttribute._id_key: 2, ObjectAttribute._name_key: "name", ObjectAttribute._type_key: "str"}
        self.assertEqual(0, len(list(self.validator.iter_errors(json_dict))))
        res = ObjectAttribute.from_json(json_dict)
        self.assertEqual(2, res.id)
        self.assertEqual("name", res.name)

    def test_from_json_no_type(self):
        json_dict = {ObjectAttribute._id_key: 2, ObjectAttribute._name_key: "name"}
        # Note: somehow when the type is missing, the values key becomes required
        self.assertEqual(2, len(list(self.validator.iter_errors(json_dict))))

    def _test_from_json_success(self, json_dict: dict, expected_type: type):
        self.assertEqual(0, len(list(self.validator.iter_errors(json_dict))))
        res = ObjectAttribute.from_json(json_dict)
        self.assertEqual(expected_type, type(res))

    def test_from_json_with_str_type(self):
        json_dict = {
            ObjectAttribute._id_key: 2,
            ObjectAttribute._name_key: "name",
            ObjectAttribute._type_key: StrAttribute.get_attribute_type()
        }
        self._test_from_json_success(json_dict, StrAttribute)

    def test_from_json_with_int_type(self):
        json_dict = {
            ObjectAttribute._id_key: 2,
            ObjectAttribute._name_key: "name",
            ObjectAttribute._type_key: IntAttribute.get_attribute_type()
        }
        self._test_from_json_success(json_dict, IntAttribute)

    def test_from_json_with_float_type(self):
        json_dict = {
            ObjectAttribute._id_key: 2,
            ObjectAttribute._name_key: "name",
            ObjectAttribute._type_key: FloatAttribute.get_attribute_type()
        }
        self._test_from_json_success(json_dict, FloatAttribute)

    def test_from_json_with_bool_type(self):
        json_dict = {
            ObjectAttribute._id_key: 2,
            ObjectAttribute._name_key: "name",
            ObjectAttribute._type_key: BoolAttribute.get_attribute_type()
        }
        self._test_from_json_success(json_dict, BoolAttribute)

    def test_from_json_with_enum_type_no_values(self):
        json_dict = {
            ObjectAttribute._id_key: 2,
            ObjectAttribute._name_key: "name",
            ObjectAttribute._type_key: EnumAttribute.get_attribute_type()
        }
        self.assertEqual(1, len(list(self.validator.iter_errors(json_dict))))

    def test_from_json_with_enum_type_empty_values(self):
        json_dict = {
            ObjectAttribute._id_key: 2,
            ObjectAttribute._name_key: "name",
            ObjectAttribute._type_key: EnumAttribute.get_attribute_type(),
            ObjectAttribute._values_key: []
        }
        self.assertEqual(1, len(list(self.validator.iter_errors(json_dict))))

    def test_from_json_unknown_type(self):
        json_dict = {
            ObjectAttribute._id_key: 2,
            ObjectAttribute._name_key: "name",
            ObjectAttribute._type_key: "def not a type"
        }
        self.assertEqual(1, len(list(self.validator.iter_errors(json_dict))))

    def test_from_json_with_enum_type_values_not_a_list(self):
        json_dict = {
            ObjectAttribute._id_key: 2,
            ObjectAttribute._name_key: "name",
            ObjectAttribute._type_key: EnumAttribute.get_attribute_type(),
            EnumAttribute._values_key: 1
        }
        self.assertEqual(1, len(list(self.validator.iter_errors(json_dict))))

    def test_from_json_with_enum_type_values_not_unique(self):
        json_dict = {
            ObjectAttribute._id_key: 2,
            ObjectAttribute._name_key: "name",
            ObjectAttribute._type_key: EnumAttribute.get_attribute_type(),
            EnumAttribute._values_key: [1, 1]
        }
        self.assertEqual(1, len(list(self.validator.iter_errors(json_dict))))

    def test_from_json_with_enum_type_with_values(self):
        # Empty value lists should raise an error during the validate() call
        json_dict = {
            ObjectAttribute._id_key: 2,
            ObjectAttribute._name_key: "name",
            ObjectAttribute._type_key: EnumAttribute.get_attribute_type(),
            EnumAttribute._values_key: [1, "a", 0., None]
        }
        self.assertEqual(0, len(list(self.validator.iter_errors(json_dict))))
        res = ObjectAttribute.from_json(json_dict)
        self.assertIsNotNone(res)

    def test_to_json(self):
        res = StrAttribute(1, "STR")
        obj_dict = res.to_json()
        self.assertEqual(0, len(list(self.validator.iter_errors(obj_dict))))
        self.assertEqual(res.get_attribute_type(), obj_dict[res._type_key])

    def test_to_json_enum_attribute(self):
        res = EnumAttribute(1, "Enum", [1, "a", .0, None])
        obj_dict = res.to_json()
        self.assertEqual(0, len(list(self.validator.iter_errors(obj_dict))))
        self.assertEqual(res.get_attribute_type(), obj_dict[res._type_key])
        self.assertEqual(res.values, obj_dict[res._values_key])

    def test_validate_success(self):
        res = EnumAttribute(1, "Bad Enum", ["a", "n"])
        obj_dict = res.to_json()
        self.assertEqual(0, len(list(self.validator.iter_errors(obj_dict))))
