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

from scene_graph_api.knowledge.ClassFilter import *
from scene_graph_api.knowledge.KnowledgeComponent import KnowledgeComponent
from scene_graph_api.logging_handlers import TestingHandler
from scene_graph_api.utils.parsing import get_validator


class TestClassFilter(TestCase):
    logger = logging.getLogger("knowledge/TestClassFilter")
    handler = TestingHandler()
    validator = get_validator(ClassFilter.schema(), registry=KnowledgeComponent.SCHEMA_REGISTRY)

    @classmethod
    def setUpClass(cls):
        cls.logger.addHandler(cls.handler)

    def setUp(self):
        self.handler.purge()

    def test_from_json_no_type(self):
        json_dict = {ClassFilter._objects_classes_key: []}
        self.assertEqual(1, len(list(self.validator.iter_errors(json_dict))))

    def test_from_json_with_whitelist_type(self):
        json_dict = {ClassFilter._type_key: WhitelistFilter.get_filter_type(), ClassFilter._objects_classes_key: [1]}
        self.assertEqual(0, len(list(self.validator.iter_errors(json_dict))))

    def test_from_json_with_blacklist_type(self):
        json_dict = {ClassFilter._type_key: BlacklistFilter.get_filter_type(), ClassFilter._objects_classes_key: []}
        self.assertEqual(0, len(list(self.validator.iter_errors(json_dict))))

    def test_from_json_with_incorrect_type(self):
        json_dict = {ClassFilter._type_key: "definitely incorrect", ClassFilter._objects_classes_key: []}
        self.assertEqual(1, len(list(self.validator.iter_errors(json_dict))))

    def test_from_json_no_type_and_no_object_classes(self):
        # Important to check that multiple errors get logged at once
        json_dict = {}
        self.assertEqual(2, len(list(self.validator.iter_errors(json_dict))))

    def test_from_json_object_classes_not_a_list(self):
        # Important to check that multiple errors get logged at once
        json_dict = {ClassFilter._type_key: WhitelistFilter.get_filter_type(), ClassFilter._objects_classes_key: 1}
        self.assertEqual(1, len(list(self.validator.iter_errors(json_dict))))

    def test_from_json_with_not_empty_list(self):
        json_dict = {ClassFilter._type_key: WhitelistFilter.get_filter_type(), ClassFilter._objects_classes_key: [1, 2]}
        self.assertEqual(0, len(list(self.validator.iter_errors(json_dict))))
        res = ClassFilter.from_json(json_dict)
        self.assertEqual(2, len(res.object_class_ids))

    def test_from_json_with_not_int_list(self):
        json_dict = {ClassFilter._type_key: WhitelistFilter.get_filter_type(), ClassFilter._objects_classes_key: ["1"]}
        self.assertEqual(1, len(list(self.validator.iter_errors(json_dict))))

    def test_to_json(self):
        res = WhitelistFilter([1, 2])
        obj_dict = res.to_json()
        self.assertEqual(0, len(list(self.validator.iter_errors(obj_dict))))
        self.assertEqual(res.get_filter_type(), obj_dict[res._type_key])
        self.assertEqual(res.object_class_ids, obj_dict[res._objects_classes_key])

    def test_validate_success(self):
        wl_filter = WhitelistFilter([1, 2])
        success = wl_filter.validate_references(1, [1, 2, 3], self.logger)
        self.assertTrue(success)
        self.assertEqual(0, self.handler.get_warning_message_count())
        self.assertEqual(0, self.handler.get_error_message_count())

    def test_validate_unknown_ids(self):
        wl_filter = WhitelistFilter([1, 2, 4])
        success = wl_filter.validate_references(1, [1, 2, 3], self.logger)
        self.assertFalse(success)
        self.assertEqual(0, self.handler.get_warning_message_count())
        self.assertEqual(1, self.handler.get_error_message_count())

    def test_validate_whitelist_empty_ids(self):
        wl_filter = WhitelistFilter([])
        success = wl_filter.validate_references(1, [1, 2, 3], self.logger)
        self.assertTrue(success)
        self.assertEqual(1, self.handler.get_warning_message_count())
        self.assertEqual(0, self.handler.get_error_message_count())

    def test_validate_whitelist_not_unique_ids(self):
        wl_filter = WhitelistFilter([1, 1])
        success = wl_filter.validate_references(1, [1, 2, 3], self.logger)
        self.assertTrue(success)
        self.assertEqual(1, len(wl_filter.object_class_ids))
        self.assertEqual(0, self.handler.get_warning_message_count())
        self.assertEqual(0, self.handler.get_error_message_count())

    def test_get_authorized_classes_whitelist(self):
        wl_filter = WhitelistFilter([1, 2])
        res = wl_filter.get_authorized_classes([1, 2, 3])
        self.assertEqual([1, 2], res)
        self.assertEqual(0, self.handler.get_warning_message_count())
        self.assertEqual(0, self.handler.get_error_message_count())

    def test_get_authorized_classes_blacklist(self):
        bl_filter = BlacklistFilter([1, 2])
        res = bl_filter.get_authorized_classes([1, 2, 3])
        self.assertEqual([3], res)
        self.assertEqual(0, self.handler.get_warning_message_count())
        self.assertEqual(0, self.handler.get_error_message_count())
