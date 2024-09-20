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

from scene_graph_api.knowledge import RelationRule, WhitelistFilter, BlacklistFilter
from scene_graph_api.knowledge.KnowledgeComponent import KnowledgeComponent
from scene_graph_api.logging_handlers import TestingHandler
from scene_graph_api.utils.parsing import get_validator


class TestRule(TestCase):
    logger = logging.getLogger("knowledge/Rule")
    handler = TestingHandler()
    validator = get_validator(RelationRule.schema(), registry=KnowledgeComponent.SCHEMA_REGISTRY)

    @classmethod
    def setUpClass(cls):
        cls.logger.addHandler(cls.handler)

    def setUp(self):
        self.handler.purge()

    def test_from_json_valid(self):
        json_dict = {
            RelationRule._id_key: 2, RelationRule._name_key: "name",
            RelationRule._subject_key: WhitelistFilter([1]).to_json(),
            RelationRule._object_key: BlacklistFilter([1]).to_json(),
        }
        self.assertEqual(0, len(list(self.validator.iter_errors(json_dict))))
        res = RelationRule.from_json(json_dict)
        self.assertIsNotNone(res)
        self.assertEqual(2, res.id)
        self.assertEqual("name", res.name)

    def test_from_json_no_subject(self):
        json_dict = {
            RelationRule._id_key: 2, RelationRule._name_key: "name",
            RelationRule._object_key: WhitelistFilter([1]).to_json(),
        }
        self.assertEqual(1, len(list(self.validator.iter_errors(json_dict))))

    def test_from_json_no_object(self):
        json_dict = {
            RelationRule._id_key: 2, RelationRule._name_key: "name",
            RelationRule._subject_key: WhitelistFilter([1]).to_json(),
        }
        self.assertEqual(1, len(list(self.validator.iter_errors(json_dict))))

    def test_from_json_subject_not_dict(self):
        json_dict = {
            RelationRule._id_key: 2, RelationRule._name_key: "name",
            RelationRule._subject_key: 1,
            RelationRule._object_key: WhitelistFilter([1]).to_json(),
        }
        self.assertEqual(1, len(list(self.validator.iter_errors(json_dict))))

    def test_from_json_object_not_dict(self):
        json_dict = {
            RelationRule._id_key: 2, RelationRule._name_key: "name",
            RelationRule._subject_key: WhitelistFilter([1]).to_json(),
            RelationRule._object_key: 1,
        }
        self.assertEqual(1, len(list(self.validator.iter_errors(json_dict))))

    def test_validate_references_success(self):
        rule = RelationRule(1, "named", WhitelistFilter([1]), WhitelistFilter([2]))
        success = rule.validate_references([1, 2, 3], self.logger)
        self.assertTrue(success)
        self.assertEqual(0, self.handler.get_warning_message_count())
        self.assertEqual(0, self.handler.get_error_message_count())

    def test_validate_references_unknown_ids_in_filters(self):
        rule = RelationRule(1, "named", WhitelistFilter([1]), WhitelistFilter([2]))
        success = rule.validate_references([3], self.logger)
        self.assertFalse(success)
        self.assertEqual(0, self.handler.get_warning_message_count())
        self.assertEqual(2, self.handler.get_error_message_count())

    def test_validate_id_zero(self):
        rule = RelationRule(0, "named", WhitelistFilter([1]), WhitelistFilter([1]))
        obj_dict = rule.to_json()
        self.assertEqual(1, len(list(self.validator.iter_errors(obj_dict))))

    def test_validate_success(self):
        rule = RelationRule(1, "named", WhitelistFilter([1]), WhitelistFilter([1]))
        obj_dict = rule.to_json()
        self.assertEqual(0, len(list(self.validator.iter_errors(obj_dict))))

    def test_to_json(self):
        rule = RelationRule(1, "named", WhitelistFilter([1]), WhitelistFilter([2]))
        obj_dict = rule.to_json()
        self.assertEqual(rule.id, obj_dict[RelationRule._id_key])
        self.assertEqual(rule.name, obj_dict[RelationRule._name_key])
        self.assertEqual(rule.subject_filter.to_json(), obj_dict[rule._subject_key])
        self.assertEqual(rule.object_filter.to_json(), obj_dict[rule._object_key])
