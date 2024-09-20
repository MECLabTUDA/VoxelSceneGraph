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

from scene_graph_api.knowledge import ObjectClass
from scene_graph_api.knowledge.ObjectAttribute import *
from scene_graph_api.logging_handlers import TestingHandler
from scene_graph_api.utils.parsing import get_validator


# noinspection DuplicatedCode
class TestObjectClass(TestCase):
    logger = logging.getLogger("knowledge/ObjectClass")
    handler = TestingHandler()
    validator = get_validator(ObjectClass.schema(), registry=KnowledgeComponent.SCHEMA_REGISTRY)

    @classmethod
    def setUpClass(cls):
        cls.logger.addHandler(cls.handler)

    def setUp(self):
        self.handler.purge()

    def test_from_json_valid_id_and_name(self):
        json_dict = {ObjectClass._id_key: 2, ObjectClass._name_key: "name"}
        self.assertEqual(0, len(list(self.validator.iter_errors(json_dict))))
        res = ObjectClass.from_json(json_dict)
        self.assertIsNotNone(res)
        self.assertEqual(2, res.id)
        self.assertEqual("name", res.name)
        self.assertFalse(res.has_mask)
        self.assertFalse(res.is_unique)
        self.assertFalse(res.is_ignored)
        # Test default color
        self.assertEqual("#ff6362", res.color)

    def test_from_json_valid_has_mask(self):
        json_dict = {ObjectClass._id_key: 2, ObjectClass._name_key: "name", ObjectClass._has_mask_key: True}
        self.assertEqual(0, len(list(self.validator.iter_errors(json_dict))))
        res = ObjectClass.from_json(json_dict)
        self.assertIsNotNone(res)
        self.assertEqual(2, res.id)
        self.assertEqual("name", res.name)
        self.assertTrue(res.has_mask)

    def test_from_json_valid_is_unique(self):
        json_dict = {ObjectClass._id_key: 2, ObjectClass._name_key: "name", ObjectClass._is_unique_key: True}
        self.assertEqual(0, len(list(self.validator.iter_errors(json_dict))))
        res = ObjectClass.from_json(json_dict)
        self.assertIsNotNone(res)
        self.assertEqual(2, res.id)
        self.assertEqual("name", res.name)
        self.assertTrue(res.is_unique)

    def test_from_json_valid_is_ignored(self):
        json_dict = {ObjectClass._id_key: 2, ObjectClass._name_key: "name", ObjectClass._is_ignored_key: True}
        self.assertEqual(0, len(list(self.validator.iter_errors(json_dict))))
        res = ObjectClass.from_json(json_dict)
        self.assertIsNotNone(res)
        self.assertEqual(2, res.id)
        self.assertEqual("name", res.name)
        self.assertTrue(res.is_ignored)

    def test_from_json_castable_has_mask(self):
        json_dict = {ObjectClass._id_key: 2, ObjectClass._name_key: "name", ObjectClass._has_mask_key: 1}
        # We're strict now...
        self.assertEqual(1, len(list(self.validator.iter_errors(json_dict))))

    def test_from_json_empty_attributes(self):
        json_dict = {ObjectClass._id_key: 2, ObjectClass._name_key: "name", ObjectClass._attributes_key: []}
        self.assertEqual(0, len(list(self.validator.iter_errors(json_dict))))
        res = ObjectClass.from_json(json_dict)
        self.assertIsNotNone(res)
        self.assertEqual(2, res.id)
        self.assertEqual("name", res.name)
        self.assertEqual([], res.attributes)

    def test_from_json_invalid_attribute_list(self):
        json_dict = {ObjectClass._id_key: 2, ObjectClass._name_key: "name", ObjectClass._attributes_key: "dummy_key"}
        self.assertEqual(1, len(list(self.validator.iter_errors(json_dict))))

    def test_from_json_invalid_attribute(self):
        json_dict = {
            ObjectClass._id_key: 2,
            ObjectClass._name_key: "name",
            # Need to have a value that is not iterable to check that the code validates that we have a dict
            ObjectClass._attributes_key: [1]
        }
        self.assertEqual(1, len(list(self.validator.iter_errors(json_dict))))

    def test_from_json_valid_attribute(self):
        # noinspection PyUnresolvedReferences
        json_dict = {
            ObjectClass._id_key: 2,
            ObjectClass._name_key: "name",
            ObjectClass._attributes_key: [{
                StrAttribute._id_key: 2,
                StrAttribute._name_key: "attr name",
                StrAttribute._type_key: StrAttribute.get_attribute_type()
            }]
        }
        self.assertEqual(0, len(list(self.validator.iter_errors(json_dict))))
        res = ObjectClass.from_json(json_dict)
        self.assertEqual(1, len(res.attributes))

    def test_from_json_valid_color(self):
        json_dict = {ObjectClass._id_key: 2, ObjectClass._name_key: "name", ObjectClass._color_key: "#fffff0"}
        self.assertEqual(0, len(list(self.validator.iter_errors(json_dict))))
        res = ObjectClass.from_json(json_dict)
        self.assertIsNotNone(res)
        self.assertEqual(2, res.id)
        self.assertEqual("name", res.name)
        self.assertEqual("#fffff0", res.color)

    def test_to_json(self):
        obj_class = ObjectClass(1, "name", [StrAttribute(2, "test")], True, True, True, "#ffffff")
        obj_dict = obj_class.to_json()
        self.assertEqual(0, len(list(self.validator.iter_errors(obj_dict))))
        self.assertEqual(obj_class.id, obj_dict[ObjectClass._id_key])
        self.assertEqual(obj_class.name, obj_dict[ObjectClass._name_key])
        self.assertEqual(obj_class.has_mask, obj_dict[ObjectClass._has_mask_key])
        self.assertEqual(obj_class.is_unique, obj_dict[ObjectClass._is_unique_key])
        self.assertEqual(obj_class.is_ignored, obj_dict[ObjectClass._is_ignored_key])
        self.assertEqual(len(obj_class.attributes), len(obj_dict[ObjectClass._attributes_key]))
        self.assertEqual(obj_class.attributes[0].to_json(), obj_dict[ObjectClass._attributes_key][0])
        self.assertEqual(obj_class.color, obj_dict[ObjectClass._color_key])

    def test_validate_success(self):
        obj_class = ObjectClass(1, "name", [], color="#ffffff")
        obj_dict = obj_class.to_json()
        self.assertEqual(0, len(list(self.validator.iter_errors(obj_dict))))

    def test_validate_id_zero(self):
        obj_class = ObjectClass(0, "name", [], color="#ffffff")
        obj_dict = obj_class.to_json()
        self.assertEqual(1, len(list(self.validator.iter_errors(obj_dict))))

    def test_validate_attributes_success(self):
        obj_class = ObjectClass(1, "name", [StrAttribute(2, "test"), StrAttribute(3, "test2")], color="#ffffff")
        success = obj_class.validate_attributes(self.logger)
        self.assertTrue(success)
        self.assertEqual(0, self.handler.get_warning_message_count())
        self.assertEqual(0, self.handler.get_error_message_count())

    def test_validate_attributes_ids_not_unique(self):
        obj_class = ObjectClass(1, "name", [StrAttribute(2, "test"), StrAttribute(2, "test2")])
        success = obj_class.validate_attributes(self.logger)
        self.assertFalse(success)
        self.assertEqual(0, self.handler.get_warning_message_count())
        self.assertEqual(1, self.handler.get_error_message_count())

    def test_validate_attributes_names_not_unique(self):
        obj_class = ObjectClass(1, "name", [StrAttribute(1, "test"), StrAttribute(2, "test")])
        success = obj_class.validate_attributes(self.logger)
        self.assertTrue(success)
        self.assertEqual(1, self.handler.get_warning_message_count())
        self.assertEqual(0, self.handler.get_error_message_count())

    def test_validate_attributes_ids_and_names_not_unique(self):
        obj_class = ObjectClass(1, "name", [StrAttribute(1, "test"), StrAttribute(1, "test")])
        success = obj_class.validate_attributes(self.logger)
        self.assertFalse(success)
        self.assertEqual(1, self.handler.get_warning_message_count())
        self.assertEqual(1, self.handler.get_error_message_count())

    def test_validate_attributes_invalid_color(self):
        obj_class = ObjectClass(1, "name", [StrAttribute(2, "test"), StrAttribute(3, "test2")], color="dummy_color")
        success = obj_class.validate_attributes(self.logger)
        self.assertFalse(success)
        self.assertEqual(0, self.handler.get_warning_message_count())
        self.assertEqual(1, self.handler.get_error_message_count())
