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

from scene_graph_api.knowledge import ObjectClass, StrAttribute, NaturalImageKG
from scene_graph_api.logging_handlers import TestingHandler
from scene_graph_api.scene import *
from scene_graph_api.scene.SceneGraphComponent import SceneGraphComponent
from scene_graph_api.utils.parsing import get_validator


# noinspection DuplicatedCode
class TestObject(TestCase):
    logger = logging.getLogger("scene/Object")
    handler = TestingHandler()
    # Define knowledge for some tests
    object_class_id = 1
    template_str_attr = StrAttribute(1, "")
    template = NaturalImageKG([
        ObjectClass(object_class_id, attributes=[template_str_attr])
    ])
    validator = get_validator(BoundingBox.schema(), registry=SceneGraphComponent.SCHEMA_REGISTRY)

    @classmethod
    def setUpClass(cls):
        cls.logger.addHandler(cls.handler)

    def setUp(self):
        self.handler.purge()

    # <------------------------------------- Objects tests ------------------------------------->
    def test_from_json_valid(self):
        json_dict = {
            BoundingBox._class_id_key: self.object_class_id,
            BoundingBox._id_key: 2,
            BoundingBox._name_key: "name",
            BoundingBox._attributes_key: [ObjectAttribute(3, "val").to_json()],
            BoundingBox._bb_key: [[1], [2]]
        }
        self.assertEqual(0, len(list(self.validator.iter_errors(json_dict))))
        res = BoundingBox.from_json(json_dict)
        self.assertEqual(self.object_class_id, res.class_id)
        self.assertEqual(2, res.id)
        self.assertEqual("name", res.name)
        self.assertEqual(1, len(res.attributes))

    def test_from_json_no_class_id(self):
        json_dict = {
            BoundingBox._id_key: 2,
            BoundingBox._name_key: "name",
            BoundingBox._attributes_key: [],
            BoundingBox._bb_key: [[1], [2]]
        }
        self.assertEqual(1, len(list(self.validator.iter_errors(json_dict))))

    def test_from_json_no_id(self):
        json_dict = {
            BoundingBox._class_id_key: self.object_class_id,
            BoundingBox._name_key: "name",
            BoundingBox._attributes_key: [],
            BoundingBox._bb_key: [[1], [2]]
        }
        self.assertEqual(1, len(list(self.validator.iter_errors(json_dict))))

    def test_from_json_no_name(self):
        json_dict = {
            BoundingBox._class_id_key: self.object_class_id,
            BoundingBox._id_key: 2,
            BoundingBox._attributes_key: [],
            BoundingBox._bb_key: [[1], [2]]
        }
        self.assertEqual(1, len(list(self.validator.iter_errors(json_dict))))

    def test_from_json_no_attributes(self):
        json_dict = {
            BoundingBox._class_id_key: self.object_class_id,
            BoundingBox._id_key: 2,
            BoundingBox._name_key: "name",
            BoundingBox._bb_key: [[1], [2]]
        }
        self.assertEqual(0, len(list(self.validator.iter_errors(json_dict))))

    def test_from_json_all_missing_key(self):
        self.assertEqual(4, len(list(self.validator.iter_errors({}))))

    def test_validate_success(self):
        obj = Object(self.object_class_id, 2, "name", [ObjectAttribute(self.template_str_attr.id, "attr")])
        success = obj.validate(self.template, self.logger)
        self.assertTrue(success)
        self.assertEqual(0, self.handler.get_warning_message_count())
        self.assertEqual(0, self.handler.get_error_message_count())

    def test_validate_valid_unknown_class_id(self):
        obj = Object(42, 2, "name", [])
        success = obj.validate(self.template, self.logger)
        self.assertFalse(success)
        self.assertEqual(0, self.handler.get_warning_message_count())
        self.assertEqual(1, self.handler.get_error_message_count())

    def test_validate_valid_missing_attr(self):
        obj = Object(self.object_class_id, 2, "name", [])
        success = obj.validate(self.template, self.logger)
        self.assertFalse(success)
        self.assertEqual(0, self.handler.get_warning_message_count())
        self.assertEqual(1, self.handler.get_error_message_count())

    def test_validate_valid_unknown_attr(self):
        obj = Object(self.object_class_id, 2, "name", [ObjectAttribute(42, "attr"), ObjectAttribute(1, "attr templ")])
        success = obj.validate(self.template, self.logger)
        self.assertFalse(success)
        self.assertEqual(0, self.handler.get_warning_message_count())
        self.assertEqual(1, self.handler.get_error_message_count())

    def test_to_json(self):
        obj = BoundingBox(1, 2, "name", [ObjectAttribute(3, 4)], ((1,), (1,)))
        json_dict = obj.to_json()
        self.assertEqual(obj.class_id, json_dict[BoundingBox._class_id_key])
        self.assertEqual(obj.id, json_dict[BoundingBox._id_key])
        self.assertEqual(obj.name, json_dict[BoundingBox._name_key])
        self.assertEqual(len(obj.attributes), len(json_dict[BoundingBox._attributes_key]))
        self.assertEqual(obj.attributes[0].to_json(), json_dict[BoundingBox._attributes_key][0])

    # <------------------------------------- Bounding box tests ------------------------------------->
    def test_bounding_box_from_json_valid(self):
        json_dict = {
            BoundingBox._class_id_key: self.object_class_id,
            BoundingBox._id_key: 2,
            BoundingBox._name_key: "name",
            BoundingBox._attributes_key: [ObjectAttribute(3, "val").to_json()],
            BoundingBox._bb_key: [[1, 2, 3], [4, 5, 6]],
        }
        self.assertEqual(0, len(list(self.validator.iter_errors(json_dict))))
        bb = BoundingBox.from_json(json_dict)
        self.assertIsNotNone(bb)
        self.assertEqual(self.object_class_id, bb.class_id)
        self.assertEqual(2, bb.id)
        self.assertEqual("name", bb.name)
        self.assertEqual(1, len(bb.attributes))
        list_bb = json_dict[BoundingBox._bb_key]
        self.assertEqual((tuple(list_bb[0]), tuple(list_bb[1])), bb.bounding_box)

    def test_bounding_box_from_json_missing_bb(self):
        json_dict = {
            BoundingBox._class_id_key: self.object_class_id,
            BoundingBox._id_key: 2,
            BoundingBox._name_key: "name",
            BoundingBox._attributes_key: [ObjectAttribute(3, "val").to_json()]
        }
        self.assertEqual(1, len(list(self.validator.iter_errors(json_dict))))

    def test_bounding_box_from_json_missing_coordinates(self):
        json_dict = {
            BoundingBox._class_id_key: self.object_class_id,
            BoundingBox._id_key: 2,
            BoundingBox._name_key: "name",
            BoundingBox._attributes_key: [ObjectAttribute(3, "val").to_json()],
            BoundingBox._bb_key: [[1, 2, 3]],
        }
        self.assertEqual(1, len(list(self.validator.iter_errors(json_dict))))

    def test_bounding_box_validate_success(self):
        bb = BoundingBox(
            self.object_class_id,
            2,
            "name",
            [ObjectAttribute(self.template_str_attr.id, "attr")],
            [[1, 2, 3], [4, 5, 6]],
        )
        success = bb.validate(self.template, self.logger)
        self.assertTrue(success)
        self.assertEqual(0, self.handler.get_warning_message_count())
        self.assertEqual(0, self.handler.get_error_message_count())

    def test_bounding_box_to_json_bb_as_tuple(self):
        # Tuple bb needs to be converted to a list for serialization
        obj = BoundingBox(1, 2, "name", [ObjectAttribute(3, 4)], ((1, 2), (3, 4)))
        json_dict = obj.to_json()
        self.assertEqual(obj.class_id, json_dict[BoundingBox._class_id_key])
        self.assertEqual(obj.id, json_dict[BoundingBox._id_key])
        self.assertEqual(obj.name, json_dict[BoundingBox._name_key])
        self.assertEqual(len(obj.attributes), len(json_dict[BoundingBox._attributes_key]))
        self.assertEqual(obj.attributes[0].to_json(), json_dict[BoundingBox._attributes_key][0])
        self.assertEqual([[1, 2], [3, 4]], json_dict[BoundingBox._bb_key])

    def test_validate_bbox_length_mismatch(self):
        obj = BoundingBox(1, 2, "name", [ObjectAttribute(1, 4)], ((1, 2), (3,)))
        success = obj.validate(self.template, self.logger)
        self.assertFalse(success)
        self.assertEqual(0, self.handler.get_warning_message_count())
        self.assertEqual(1, self.handler.get_error_message_count())
