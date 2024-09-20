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
from scene_graph_api.knowledge import RelationRule, WhitelistFilter, NaturalImageKG
from scene_graph_api.logging_handlers import TestingHandler
from scene_graph_api.scene import *
from scene_graph_api.scene.SceneGraphComponent import SceneGraphComponent
from scene_graph_api.utils.parsing import get_validator


class TestRelation(TestCase):
    logger = logging.getLogger("scene/Relation")
    handler = TestingHandler()
    # Define knowledge for some tests
    object_class_id1 = 1
    object_class_id2 = 2
    rel_id = 1
    unknown_rel_id = rel_id + 1
    template = NaturalImageKG(
        [ObjectClass(object_class_id1), ObjectClass(object_class_id2)],
        [RelationRule(rel_id, "rule", WhitelistFilter([object_class_id1]), WhitelistFilter([object_class_id2]))]
    )
    # Define object instances
    obj1_cls1 = BoundingBox(object_class_id1, 1, "", [], [[], []])
    obj2_cls1 = BoundingBox(object_class_id1, 2, "", [], [[], []])
    obj3_cls2 = BoundingBox(object_class_id2, 3, "", [], [[], []])
    known_objs = [obj1_cls1, obj2_cls1, obj3_cls2]
    unknown_obj = BoundingBox(object_class_id1, 4, "", [], [[], []])
    validator = get_validator(Relation.schema(), registry=SceneGraphComponent.SCHEMA_REGISTRY)

    @classmethod
    def setUpClass(cls):
        cls.logger.addHandler(cls.handler)

    def setUp(self):
        self.handler.purge()

    def test_from_json_valid(self):
        json_dict = {Relation._rule_id_key: 0, Relation._subject_id_key: 1, Relation._object_id_key: 2}
        self.assertEqual(0, len(list(self.validator.iter_errors(json_dict))))
        res = Relation.from_json(json_dict)
        self.assertIsNotNone(res)
        self.assertEqual(0, res.rule_id)
        self.assertEqual(1, res.subject_id)
        self.assertEqual(2, res.object_id)

    def test_from_json_missing_rule_id(self):
        json_dict = {Relation._subject_id_key: 1, Relation._object_id_key: 2}
        self.assertEqual(1, len(list(self.validator.iter_errors(json_dict))))

    def test_from_json_missing_subject_id(self):
        json_dict = {Relation._rule_id_key: 0, Relation._object_id_key: 2}
        self.assertEqual(1, len(list(self.validator.iter_errors(json_dict))))

    def test_from_json_missing_object_id(self):
        json_dict = {Relation._rule_id_key: 0, Relation._subject_id_key: 1}
        self.assertEqual(1, len(list(self.validator.iter_errors(json_dict))))

    def test_from_json_empty(self):
        self.assertEqual(3, len(list(self.validator.iter_errors({}))))

    def test_validate_success(self):
        rel = Relation(self.rel_id, self.obj1_cls1.id, self.obj3_cls2.id)
        success = rel.validate_references(self.template, self.known_objs, self.logger)
        self.assertTrue(success)
        self.assertEqual(0, self.handler.get_warning_message_count())
        self.assertEqual(0, self.handler.get_error_message_count())

    def _validate_fail(self, rel: Relation):
        success = rel.validate_references(self.template, self.known_objs, self.logger)
        self.assertFalse(success)
        self.assertEqual(0, self.handler.get_warning_message_count())
        self.assertEqual(1, self.handler.get_error_message_count())

    def test_validate_unknown_rel_id(self):
        rel = Relation(self.unknown_rel_id, self.obj1_cls1.id, self.obj3_cls2.id)
        self._validate_fail(rel)

    def test_validate_unknown_subject_id(self):
        rel = Relation(self.rel_id, self.unknown_obj.id, self.obj3_cls2.id)
        self._validate_fail(rel)

    def test_validate_unknown_object_id(self):
        rel = Relation(self.rel_id, self.obj1_cls1.id, self.unknown_obj.id)
        self._validate_fail(rel)

    def test_validate_not_authorized_subject_type(self):
        rel = Relation(self.rel_id, self.obj3_cls2.id, self.obj3_cls2.id)
        self._validate_fail(rel)

    def test_validate_not_authorized_object_type(self):
        rel = Relation(self.rel_id, self.obj1_cls1.id, self.obj2_cls1.id)
        self._validate_fail(rel)

    def test_to_json(self):
        rel = Relation(1, 2, 3)
        json_dict = rel.to_json()
        self.assertEqual(rel.rule_id, json_dict[Relation._rule_id_key])
        self.assertEqual(rel.subject_id, json_dict[Relation._subject_id_key])
        self.assertEqual(rel.object_id, json_dict[Relation._object_id_key])
