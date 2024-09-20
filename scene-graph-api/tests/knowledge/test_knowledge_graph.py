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
import os
import shutil
import tempfile
from pathlib import Path
from unittest import TestCase

from scene_graph_api.knowledge import *
from scene_graph_api.knowledge.KnowledgeComponent import KnowledgeComponent
from scene_graph_api.logging_handlers import TestingHandler
from scene_graph_api.utils.parsing import get_validator


# noinspection DuplicatedCode
class TestKnowledgeGraph(TestCase):
    logger = logging.getLogger("knowledge/SceneGraphTemplate")
    handler = TestingHandler()
    temp_folder = tempfile.mkdtemp()
    validator = get_validator(KnowledgeGraph.schema(), registry=KnowledgeComponent.SCHEMA_REGISTRY)

    @classmethod
    def setUpClass(cls):
        cls.logger.addHandler(cls.handler)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_folder)

    def setUp(self):
        self.handler.purge()

    def test_natural_image_type_from_json_valid_no_rules_no_image_level(self):
        json_dict = {
            KnowledgeGraph._type_key: NaturalImageKG.get_graph_type(),
            KnowledgeGraph._object_classes_key: [ObjectClass(1, "").to_json()]
        }
        self.assertEqual(0, len(list(self.validator.iter_errors(json_dict))))
        res = KnowledgeGraph.from_json(json_dict)
        self.assertEqual(hash(frozenset(json_dict)), res.hash)
        self.assertEqual(1, len(res.classes))

    def test_natural_image_type_from_json_valid_no_rules(self):
        json_dict = {
            KnowledgeGraph._type_key: NaturalImageKG.get_graph_type(),
            KnowledgeGraph._object_classes_key: [ObjectClass(1, "").to_json()],
            KnowledgeGraph._image_level_attributes_key: [IntAttribute(1, "a").to_json()]
        }
        self.assertEqual(0, len(list(self.validator.iter_errors(json_dict))))
        res = KnowledgeGraph.from_json(json_dict)
        self.assertEqual(hash(frozenset(json_dict)), res.hash)
        self.assertEqual(1, len(res.classes))

    def test_natural_image_type_from_json_valid_invalid_image_level_attribute(self):
        json_dict = {
            KnowledgeGraph._type_key: NaturalImageKG.get_graph_type(),
            KnowledgeGraph._object_classes_key: [ObjectClass(1, "").to_json()],
            KnowledgeGraph._image_level_attributes_key: [IntAttribute(0, "a").to_json()]
        }
        self.assertEqual(1, len(list(self.validator.iter_errors(json_dict))))

    def test_radiology_image_type_from_json_valid_no_rules_no_image_level(self):
        json_dict = {
            KnowledgeGraph._type_key: RadiologyImageKG.get_graph_type(),
            KnowledgeGraph._object_classes_key: [ObjectClass(1, "").to_json()],
            RadiologyImageKG._window_center_key: 90,
            RadiologyImageKG._window_width_key: 99,
            RadiologyImageKG._default_axis_key: RadiologyImageKG.ImageAxis.sagittal.value
        }

        self.assertEqual(0, len(list(self.validator.iter_errors(json_dict))))
        # noinspection PyTypeChecker
        res: RadiologyImageKG = KnowledgeGraph.from_json(json_dict)
        self.assertEqual(90, res.window_center)
        self.assertEqual(99, res.window_width)
        self.assertEqual(RadiologyImageKG.ImageAxis.sagittal, res.default_axis)

    def test_radiology_image_type_from_json_missing_additional_fields(self):
        json_dict = {
            KnowledgeGraph._type_key: RadiologyImageKG.get_graph_type(),
            KnowledgeGraph._object_classes_key: [ObjectClass(1, "").to_json()],
        }
        self.assertEqual(2, len(list(self.validator.iter_errors(json_dict))))

    def test_radiology_image_type_from_json_invalid_default_axis(self):
        json_dict = {
            KnowledgeGraph._type_key: RadiologyImageKG.get_graph_type(),
            KnowledgeGraph._object_classes_key: [ObjectClass(1, "").to_json()],
            RadiologyImageKG._window_center_key: 90,
            RadiologyImageKG._window_width_key: 99,
            RadiologyImageKG._default_axis_key: "test"
        }

        self.assertEqual(1, len(list(self.validator.iter_errors(json_dict))))

    def test_from_json_no_classes(self):
        # We allow empty knowledge graphs
        json_dict = {KnowledgeGraph._type_key: NaturalImageKG.get_graph_type()}
        self.assertEqual(0, len(list(self.validator.iter_errors(json_dict))))

    def test_from_json_invalid_classes(self):
        # Important to make the bad value not iterable to check tha the code expects a dict
        json_dict = {
            KnowledgeGraph._type_key: NaturalImageKG.get_graph_type(),
            KnowledgeGraph._object_classes_key: [1]
        }
        self.assertEqual(1, len(list(self.validator.iter_errors(json_dict))))

    def test_from_json_no_classes_but_rules(self):
        json_dict = {
            KnowledgeGraph._type_key: NaturalImageKG.get_graph_type(),
            KnowledgeGraph._rules_key: [
                RelationRule(1, "test", BlacklistFilter([]), BlacklistFilter([])).to_json()
            ]}
        # This should fail during the validation
        self.assertEqual(0, len(list(self.validator.iter_errors(json_dict))))

    def test_from_json_invalid_rules(self):
        # Important to make the bad value not iterable to check tha the code expects a dict
        json_dict = {
            KnowledgeGraph._type_key: NaturalImageKG.get_graph_type(),
            KnowledgeGraph._object_classes_key: [],
            KnowledgeGraph._rules_key: [1]
        }
        self.assertEqual(1, len(list(self.validator.iter_errors(json_dict))))

    def test_validate_success(self):
        tmp = NaturalImageKG([ObjectClass(1, "name"), ObjectClass(2, "name2")])
        success = tmp.validate(self.logger)
        self.assertTrue(success)
        self.assertEqual(0, self.handler.get_warning_message_count())
        self.assertEqual(0, self.handler.get_error_message_count())

    def test_validate_obj_class_ids_not_unique(self):
        tmp = NaturalImageKG([ObjectClass(1, "name"), ObjectClass(1, "name2")])
        success = tmp.validate(self.logger)
        self.assertFalse(success)
        self.assertEqual(0, self.handler.get_warning_message_count())
        self.assertEqual(1, self.handler.get_error_message_count())

    def test_validate_obj_class_names_not_unique(self):
        tmp = NaturalImageKG([ObjectClass(1, "name"), ObjectClass(2, "name")])
        success = tmp.validate(self.logger)
        self.assertTrue(success)
        self.assertEqual(1, self.handler.get_warning_message_count())
        self.assertEqual(0, self.handler.get_error_message_count())

    def test_validate_rule_ids_not_unique(self):
        tmp = NaturalImageKG(
            [ObjectClass(1, name="1"), ObjectClass(2, name="2")],
            [
                RelationRule(1, "rule", BlacklistFilter([]), BlacklistFilter([])),
                RelationRule(1, "rule2", BlacklistFilter([]), BlacklistFilter([]))
            ]
        )
        success = tmp.validate(self.logger)
        self.assertFalse(success)
        self.assertEqual(0, self.handler.get_warning_message_count())
        self.assertEqual(1, self.handler.get_error_message_count())

    def test_validate_rule_names_not_unique(self):
        tmp = NaturalImageKG(
            [ObjectClass(1, name="1"), ObjectClass(2, name="2")],
            [
                RelationRule(1, "rule", BlacklistFilter([]), BlacklistFilter([])),
                RelationRule(2, "rule", BlacklistFilter([]), BlacklistFilter([]))
            ]
        )
        success = tmp.validate(self.logger)
        self.assertTrue(success)
        self.assertEqual(1, self.handler.get_warning_message_count())
        self.assertEqual(0, self.handler.get_error_message_count())

    def test_validate_unknown_ids(self):
        tmp = NaturalImageKG([], [RelationRule(1, "rule", BlacklistFilter([1]), BlacklistFilter([2]))])
        success = tmp.validate(self.logger)
        self.assertFalse(success)
        self.assertEqual(0, self.handler.get_warning_message_count())
        self.assertEqual(2, self.handler.get_error_message_count())

    def test_natural_image_type_from_json_valid_no_rules_invalid_image_level_attribute(self):
        tmp = NaturalImageKG(
            [ObjectClass(1, "name"), ObjectClass(2, "name2")],
            [],
            [IntAttribute(1, "a"), IntAttribute(1, "b")]
        )
        success = tmp.validate(self.logger)
        self.assertFalse(success)
        self.assertEqual(0, self.handler.get_warning_message_count())
        self.assertEqual(1, self.handler.get_error_message_count())

    def test_to_json(self):
        tmp = NaturalImageKG(
            [ObjectClass(1, "name")],
            [RelationRule(2, "rule", BlacklistFilter([]), BlacklistFilter([]))]
        )
        obj_dict = tmp.to_json()
        self.assertEqual(1, len(obj_dict[KnowledgeGraph._object_classes_key]))
        self.assertEqual(1, len(obj_dict[KnowledgeGraph._rules_key]))

    def test_to_json_radiology_image(self):
        tmp = RadiologyImageKG(
            [ObjectClass(1, "name")],
            [RelationRule(2, "rule", BlacklistFilter([]), BlacklistFilter([]))],
            window_center=200,
            window_width=400,
            default_axis=RadiologyImageKG.ImageAxis.sagittal
        )
        obj_dict = tmp.to_json()
        self.assertEqual(1, len(obj_dict[KnowledgeGraph._object_classes_key]))
        self.assertEqual(1, len(obj_dict[KnowledgeGraph._rules_key]))
        self.assertEqual(200, obj_dict[RadiologyImageKG._window_center_key])
        self.assertEqual(400, obj_dict[RadiologyImageKG._window_width_key])
        self.assertEqual(RadiologyImageKG.ImageAxis.sagittal.value, obj_dict[RadiologyImageKG._default_axis_key])

    def test_load_valid_json(self):
        tmp = NaturalImageKG(
            [ObjectClass(1, "1"), ObjectClass(2, "2")],
            [RelationRule(1, "rule", BlacklistFilter([1]), BlacklistFilter([2]))]
        )
        path = os.path.join(self.temp_folder, "test_load_valid_json.json")
        tmp.save(path)
        tmp2 = KnowledgeGraph.load(path, self.logger)
        self.assertIsNotNone(tmp2)
        self.assertEqual(0, self.handler.get_warning_message_count())
        self.assertEqual(0, self.handler.get_error_message_count())

    def test_load_pathlike_valid_json(self):
        tmp = NaturalImageKG(
            [ObjectClass(1, "1"), ObjectClass(2, "2")],
            [RelationRule(1, "rule", BlacklistFilter([1]), BlacklistFilter([2]))]
        )
        path = Path(self.temp_folder) / "test_load_valid_json.json"
        tmp.save(path)
        tmp2 = KnowledgeGraph.load(path, self.logger)
        self.assertIsNotNone(tmp2)
        self.assertEqual(0, self.handler.get_warning_message_count())
        self.assertEqual(0, self.handler.get_error_message_count())

    def test_load_missing_file(self):
        path = os.path.join(self.temp_folder, "test_load_valid_missing_file.json")
        self.assertFalse(os.path.isfile(path))
        tmp2 = KnowledgeGraph.load(path, self.logger)
        self.assertIsNone(tmp2)
        self.assertEqual(0, self.handler.get_warning_message_count())
        self.assertEqual(1, self.handler.get_error_message_count())

    def test_load_pathlike_missing_file(self):
        path = Path(self.temp_folder) / "test_load_valid_missing_file.json"
        self.assertFalse(os.path.isfile(path))
        tmp2 = KnowledgeGraph.load(path, self.logger)
        self.assertIsNone(tmp2)
        self.assertEqual(0, self.handler.get_warning_message_count())
        self.assertEqual(1, self.handler.get_error_message_count())

    def test_load_invalid_content(self):
        path = os.path.join(self.temp_folder, "test_load_valid_invalid_content.json")
        with open(path, "w") as f:
            f.write("Invalid content")

        tmp2 = KnowledgeGraph.load(path, self.logger)
        self.assertIsNone(tmp2)
        self.assertEqual(0, self.handler.get_warning_message_count())
        self.assertEqual(1, self.handler.get_error_message_count())

    def test_load_pathlike_invalid_content(self):
        path = Path(self.temp_folder) / "test_load_valid_invalid_content.json"
        with open(path, "w") as f:
            f.write("Invalid content")

        tmp2 = KnowledgeGraph.load(path, self.logger)
        self.assertIsNone(tmp2)
        self.assertEqual(0, self.handler.get_warning_message_count())
        self.assertEqual(1, self.handler.get_error_message_count())

    def test_load_pathlike_bad_json_schema(self):
        path = Path(self.temp_folder) / "test_load_valid_invalid_content.json"
        with open(path, "w") as f:
            f.write("{}")

        tmp2 = KnowledgeGraph.load(path, self.logger)
        self.assertIsNone(tmp2)
        self.assertEqual(0, self.handler.get_warning_message_count())
        self.assertEqual(3, self.handler.get_error_message_count())

    def test_hash_update_when_saving(self):
        template = NaturalImageKG([], [], [], 0)
        json_dict = template.to_json()
        new_hash = json_dict[template._hash_key]
        del json_dict[template._hash_key]
        self.assertNotEqual(0, new_hash)
        self.assertEqual(hash(frozenset(json_dict)), new_hash)

    def test_remap_ids_as_contiguous_no_op(self):
        tmp = NaturalImageKG(
            [ObjectClass(1, "1"), ObjectClass(2, "2")],
            [RelationRule(1, "rule", BlacklistFilter([1]), BlacklistFilter([2]))]
        )
        tmp.remap_ids_as_contiguous()
        self.assertEqual(1, tmp.classes[0].id)
        self.assertEqual(2, tmp.classes[1].id)
        self.assertEqual([1], tmp.rules[0].subject_filter.object_class_ids)
        self.assertEqual([2], tmp.rules[0].object_filter.object_class_ids)

    def test_remap_ids_as_contiguous_remap(self):
        tmp = NaturalImageKG(
            [ObjectClass(4, "1"), ObjectClass(2, "2")],
            [RelationRule(1, "rule", BlacklistFilter([4]), BlacklistFilter([2]))]
        )
        tmp.remap_ids_as_contiguous()
        self.assertEqual(1, tmp.classes[0].id)
        self.assertEqual(2, tmp.classes[1].id)
        self.assertEqual([1], tmp.rules[0].subject_filter.object_class_ids)
        self.assertEqual([2], tmp.rules[0].object_filter.object_class_ids)

    def test_remap_ids_as_contiguous_remap_attributes(self):
        tmp = NaturalImageKG(
            [ObjectClass(1, "1", attributes=[IntAttribute(4, "4"), IntAttribute(2, "2")])],
            []
        )
        tmp.remap_ids_as_contiguous()
        self.assertEqual(2, len(tmp.classes[0].attributes))
        self.assertEqual(1, tmp.classes[0].attributes[0].id)
        self.assertEqual("4", tmp.classes[0].attributes[0].name)
        self.assertEqual(2, tmp.classes[0].attributes[1].id)
        self.assertEqual("2", tmp.classes[0].attributes[1].name)

    def test_remap_ids_as_contiguous_remap_image_attributes(self):
        tmp = NaturalImageKG(
            [],
            [],
            [IntAttribute(4, "4"), IntAttribute(2, "2")]
        )
        tmp.remap_ids_as_contiguous()
        self.assertEqual(2, len(tmp.image.attributes))
        self.assertEqual(1, tmp.image.attributes[0].id)
        self.assertEqual("4", tmp.image.attributes[0].name)
        self.assertEqual(2, tmp.image.attributes[1].id)
        self.assertEqual("2", tmp.image.attributes[1].name)

    def test_remap_ids_as_contiguous_ignored_put_last(self):
        tmp = NaturalImageKG(
            [
                ObjectClass(1, "1", is_ignored=True),
                ObjectClass(3, "3", is_ignored=True),
                ObjectClass(4, "4")
            ],
            []
        )
        tmp.remap_ids_as_contiguous()
        self.assertEqual(1, tmp.classes[0].id)
        self.assertEqual(2, tmp.classes[1].id)
        self.assertEqual(3, tmp.classes[2].id)
        self.assertEqual("4", tmp.classes[0].name)
        self.assertEqual("1", tmp.classes[1].name)
        self.assertEqual("3", tmp.classes[2].name)

    def test_repr(self):
        attrs = [
            StrAttribute(1, "str"),
            IntAttribute(2, "int"),
            FloatAttribute(3, "float"),
            BoolAttribute(4, "bool"),
            EnumAttribute(5, "enum", [1, "dfg", None, 1., logging]),
        ]
        tmp = NaturalImageKG(
            [ObjectClass(4, "1", attributes=attrs)],
            [RelationRule(1, "rule", BlacklistFilter([4]), WhitelistFilter([2]))]
        )
        # Just test that no exception is raised
        self.assertTrue(isinstance(repr(tmp), str))

    def test_to_coco(self):
        # Check that ids are remapped as contiguous
        tmp = NaturalImageKG([ObjectClass(3, "Obj1"), ObjectClass(4, "Bb2")], [RelationRule(5, "rule")])
        coco = tmp.to_coco()
        # TODO ids as contiguous when?
        # Check keys
        self.assertTrue("info" in coco)
        self.assertTrue("categories" in coco)
        self.assertTrue("images" in coco)
        self.assertTrue("annotations" in coco)
        self.assertTrue("predicates" in coco)
        self.assertTrue("relations" in coco)
        # Check categories
        self.assertEqual([{"id": 3, "name": "Obj1"}, {"id": 4, "name": "Bb2"}], coco["categories"])
        self.assertEqual([{"id": 5, "name": "rule"}], coco["predicates"])
