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
from typing import Any
from unittest import TestCase

from scene_graph_api.knowledge import ObjectClass, StrAttribute, IntAttribute, \
    FloatAttribute, EnumAttribute, NaturalImageKG
from scene_graph_api.logging_handlers import TestingHandler
from scene_graph_api.scene import *
from scene_graph_api.scene.SceneGraphComponent import SceneGraphComponent
from scene_graph_api.utils.parsing import get_validator


# noinspection DuplicatedCode
class TestObjectAttribute(TestCase):
    logger = logging.getLogger("scene/ObjectAttribute")
    handler = TestingHandler()
    # Define knowledge for some tests
    parent_object_class_id = 1
    template_str_attr = StrAttribute(1, "")
    template_int_attr = IntAttribute(2, "")
    template_float_attr = FloatAttribute(3, "")
    template_enum_attr = EnumAttribute(4, "", [1, 2])
    template_obj = ObjectClass(
        parent_object_class_id,
        attributes=[template_str_attr,
                    template_int_attr,
                    template_float_attr,
                    template_enum_attr]
    )
    template = NaturalImageKG([template_obj])
    validator = get_validator(ObjectAttribute.schema(), registry=SceneGraphComponent.SCHEMA_REGISTRY)

    @classmethod
    def setUpClass(cls):
        cls.logger.addHandler(cls.handler)

    def setUp(self):
        self.handler.purge()

    def test_from_json_valid(self):
        json_dict = {ObjectAttribute._id_key: 2, ObjectAttribute._value_key: "val"}
        self.assertEqual(0, len(list(self.validator.iter_errors(json_dict))))
        res = ObjectAttribute.from_json(json_dict)
        self.assertIsNotNone(res)
        self.assertEqual(2, res.id)
        self.assertEqual("val", res.value)

    def test_from_json_no_id(self):
        json_dict = {ObjectAttribute._value_key: "val"}
        self.assertEqual(1, len(list(self.validator.iter_errors(json_dict))))

    def test_from_json_no_value(self):
        json_dict = {ObjectAttribute._id_key: 2}
        self.assertEqual(1, len(list(self.validator.iter_errors(json_dict))))

    def test_from_json_empty(self):
        json_dict = {}
        self.assertEqual(2, len(list(self.validator.iter_errors(json_dict))))

    def _test_validate_type_and_value_type_found(self, attr_id: int, value: Any):
        attr = ObjectAttribute(attr_id, value)
        success = attr.validate_type_and_value(self.template_obj, self.logger)
        self.assertTrue(success)
        self.assertEqual(0, self.handler.get_warning_message_count())
        self.assertEqual(0, self.handler.get_error_message_count())

    def test_validate_type_and_value_type_str_found(self):
        self._test_validate_type_and_value_type_found(self.template_str_attr.id, "valid")

    def test_validate_type_and_value_type_int_found(self):
        self._test_validate_type_and_value_type_found(self.template_int_attr.id, 1)

    def test_validate_type_and_value_type_float_found(self):
        self._test_validate_type_and_value_type_found(self.template_float_attr.id, 1.)

    def test_validate_type_and_value_type_unknown_type(self):
        attr = ObjectAttribute(42, "")
        success = attr.validate_type_and_value(self.template_obj, self.logger)
        self.assertFalse(success)
        self.assertEqual(0, self.handler.get_warning_message_count())
        self.assertEqual(1, self.handler.get_error_message_count())

    def test_validate_type_and_value_type_enum_found(self):
        self._test_validate_type_and_value_type_found(self.template_str_attr.id, self.template_enum_attr.values[0])

    def test_validate_type_and_value_type_str_value_not_str(self):
        # Cast to str never fails
        attr = ObjectAttribute(self.template_str_attr.id, 1)
        success = attr.validate_type_and_value(self.template_obj, self.logger)
        self.assertTrue(success)
        self.assertEqual(0, self.handler.get_warning_message_count())
        self.assertEqual(0, self.handler.get_error_message_count())

    def test_validate_type_and_value_type_int_value_not_int_invalid(self):
        attr = ObjectAttribute(self.template_int_attr.id, "sdf")
        success = attr.validate_type_and_value(self.template_obj, self.logger)
        self.assertFalse(success)
        self.assertEqual(1, self.handler.get_warning_message_count())
        self.assertEqual(1, self.handler.get_error_message_count())

    def test_validate_type_and_value_type_int_value_not_int_valid(self):
        # Cast to str never fails
        attr = ObjectAttribute(self.template_int_attr.id, 1.)
        success = attr.validate_type_and_value(self.template_obj, self.logger)
        self.assertTrue(success)
        self.assertEqual(1, self.handler.get_warning_message_count())
        self.assertEqual(0, self.handler.get_error_message_count())

    def test_validate_type_and_value_type_float_value_not_float_invalid(self):
        attr = ObjectAttribute(self.template_int_attr.id, "sdf")
        success = attr.validate_type_and_value(self.template_obj, self.logger)
        self.assertFalse(success)
        self.assertEqual(1, self.handler.get_warning_message_count())
        self.assertEqual(1, self.handler.get_error_message_count())

    def test_validate_type_and_value_type_float_value_not_float_valid(self):
        attr = ObjectAttribute(self.template_float_attr.id, "1.")
        success = attr.validate_type_and_value(self.template_obj, self.logger)
        self.assertTrue(success)
        self.assertEqual(0, self.handler.get_warning_message_count())
        self.assertEqual(0, self.handler.get_error_message_count())

    def test_validate_type_and_value_type_enum_value_invalid(self):
        attr = ObjectAttribute(self.template_enum_attr.id, "sdf")
        success = attr.validate_type_and_value(self.template_obj, self.logger)
        self.assertFalse(success)
        self.assertEqual(0, self.handler.get_warning_message_count())
        self.assertEqual(1, self.handler.get_error_message_count())

    def test_validate_type_and_value_type_enum_value_valid(self):
        attr = ObjectAttribute(self.template_enum_attr.id, self.template_enum_attr.values[0])
        success = attr.validate_type_and_value(self.template_obj, self.logger)
        self.assertTrue(success)
        self.assertEqual(0, self.handler.get_warning_message_count())
        self.assertEqual(0, self.handler.get_error_message_count())

    def test_to_json(self):
        attr = ObjectAttribute(42, "val")
        json_dict = attr.to_json()
        self.assertEqual(attr.id, json_dict[ObjectAttribute._id_key])
        self.assertEqual(attr.value, json_dict[ObjectAttribute._value_key])
