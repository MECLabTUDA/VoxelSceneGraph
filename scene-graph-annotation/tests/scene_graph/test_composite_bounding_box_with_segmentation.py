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

import nibabel as nib
import numpy as np

from scene_graph_annotation.knowledge import ObjectClass, StrAttribute, RelationRule, NaturalImageKG
from scene_graph_annotation.logging_handlers import TestingHandler
from scene_graph_annotation.scene import SceneGraph, Relation, Attribute, BoundingBox
from scene_graph_annotation.scene.object_merging import merge_with_mask
from scene_graph_annotation.scene.object_splitting import split_with_mask


# noinspection DuplicatedCode
class TestCompositeBoundingBoxWithSegmentation(TestCase):
    logger = logging.getLogger("scene_graph/TestCompositeBoundingBoxWithSegmentation")
    handler = TestingHandler()
    # Define template for some tests
    object_class_id = 1
    str_attr_id = 2
    knowledge_graph = NaturalImageKG(
        [ObjectClass(object_class_id, attributes=[StrAttribute(str_attr_id, "attr")])],
        [RelationRule(1)]
    )
    # noinspection PyTypeChecker
    seg1 = BoundingBox(object_class_id, 1, "Seg1", [[1], [1]], attributes=[Attribute(str_attr_id, "Test Attr")])
    # noinspection PyTypeChecker
    seg2 = BoundingBox(object_class_id, 2, "Seg2", [[1], [1]], attributes=[Attribute(str_attr_id, "Test Attr2")])
    object_labelmap = np.array([0, 0, 1, 1, 0, 0, 2, 2], dtype=np.uint8)
    object_labelmap_orig = object_labelmap.copy()
    scene_graph = SceneGraph(
        knowledge_graph,
        image_affine=np.eye(4),
        image_header=nib.Nifti1Header(),
        bounding_box_objects=[seg1, seg2],
        relations=[Relation(1, seg1.id, seg2.id)],
        object_labelmap=object_labelmap
    )

    @classmethod
    def setUpClass(cls):
        cls.logger.addHandler(cls.handler)

    def setUp(self):
        self.handler.purge()

    def test_merge_segmentations_fail_same_seg(self):
        with self.assertRaises(AssertionError):
            merge_with_mask(self.scene_graph, self.seg1, self.seg1)

    def test_merge_segmentations_fail_diff_obj_classes(self):
        bad_seg = BoundingBox(self.object_class_id + 1, 3, "Name", [[], []])
        with self.assertRaises(AssertionError):
            merge_with_mask(self.scene_graph, self.seg1, bad_seg)

    def test_merge_segmentations_and_split_success(self):
        object_hitboxes = self.scene_graph.object_hitboxes
        object_labelmap = self.scene_graph.object_labelmap  # Do this because the cast to uint8 changes ref

        # Validate that the scene graph is correct at the beginning
        success = self.scene_graph.validate(self.logger)
        self.assertTrue(success)
        rel = self.scene_graph.relations_by_rule_id[1][0]
        self.assertEqual(self.seg1.id, rel.subject_id)
        self.assertEqual(self.seg2.id, rel.object_id)

        # Merge the two segmentations
        composite = merge_with_mask(self.scene_graph, self.seg1, self.seg2)
        self.assertEqual(self.seg1.id, composite.class_id)
        # self.assertEqual(self.seg1.name, composite.name)
        self.assertEqual(1, len(composite.attributes))

        # Check that the segmentation list has been updated
        self.assertTrue(composite in self.scene_graph.bounding_boxes_by_class_id[self.object_class_id])
        self.assertFalse(self.seg1 in self.scene_graph.bounding_boxes_by_class_id[self.object_class_id])
        self.assertFalse(self.seg2 in self.scene_graph.bounding_boxes_by_class_id[self.object_class_id])

        # Check that the relations have been updated
        self.assertEqual(composite.id, rel.subject_id)
        self.assertEqual(composite.id, rel.object_id)

        # Check that the labelmap has been updated
        self.assertTrue(np.alltrue((self.object_labelmap_orig == 0) == (object_labelmap == 0)))
        self.assertTrue(np.alltrue(object_labelmap[self.object_labelmap_orig == self.seg1.id] == composite.id))
        self.assertTrue(np.alltrue(object_labelmap[self.object_labelmap_orig == self.seg2.id] == composite.id))

        # Check that the object hitboxes have been updated
        self.assertTrue(np.alltrue((self.object_labelmap_orig == 0) == (object_hitboxes == 0)))
        self.assertTrue(np.alltrue(object_hitboxes[self.object_labelmap_orig == self.seg1.id] == composite.id))
        self.assertTrue(np.alltrue(object_hitboxes[self.object_labelmap_orig == self.seg2.id] == composite.id))

        # Validate that the scene graph is still correct
        success = self.scene_graph.validate(self.logger)
        self.assertTrue(success)

        # Update the attribute value in the composite
        composite.attributes[0].value += " changed"
        self.assertNotEqual(composite.attributes[0].value, self.seg1.attributes[0].value)
        self.assertNotEqual(composite.attributes[0].value, self.seg2.attributes[0].value)

        # Split back the segmentation
        seg1, seg2 = split_with_mask(composite, self.scene_graph)

        # Check that the segmentation list has been updated
        self.assertFalse(composite in self.scene_graph.bounding_boxes_by_class_id[self.object_class_id])
        self.assertTrue(seg1 in self.scene_graph.bounding_boxes_by_class_id[self.object_class_id])
        self.assertTrue(seg2 in self.scene_graph.bounding_boxes_by_class_id[self.object_class_id])

        # Check that the relations have been updated
        self.assertEqual(2, len(self.scene_graph.relations_by_rule_id[1]))

        # Check that the labelmap has been updated
        self.assertTrue(np.alltrue((self.object_labelmap_orig == 0) == (self.object_labelmap == 0)))
        self.assertTrue(np.alltrue(object_labelmap[self.object_labelmap_orig == seg1.id] == seg1.id))
        self.assertTrue(np.alltrue(object_labelmap[self.object_labelmap_orig == seg2.id] == seg2.id))

        # Check that the object hitboxes have been updated
        self.assertTrue(np.alltrue((self.object_labelmap_orig == 0) == (object_hitboxes == 0)))
        self.assertTrue(np.alltrue(object_hitboxes[self.object_labelmap_orig == self.seg1.id] == seg1.id))
        self.assertTrue(np.alltrue(object_hitboxes[self.object_labelmap_orig == self.seg2.id] == seg2.id))

        # Validate that the scene graph is still correct
        success = self.scene_graph.validate(self.logger)
        self.assertTrue(success)
        self.assertEqual(0, self.handler.get_warning_message_count())
        self.assertEqual(0, self.handler.get_error_message_count())

    def test_merge_split_merge_split_duplicate_relation_removal(self):
        # Note: this test depends on the previous one to succeed (such that it's a no-op)

        # Validate that the scene graph is correct at the beginning
        success = self.scene_graph.validate(self.logger)
        self.assertTrue(success)

        # 1st merge
        composite = merge_with_mask(self.scene_graph, self.seg1, self.seg2)
        # Check that the relations have been updated
        self.assertEqual(1, len(self.scene_graph.relations_by_rule_id[1]))

        # Validate after 1st merge
        success = self.scene_graph.validate(self.logger)
        self.assertTrue(success)

        # 1st split
        split_with_mask(composite, self.scene_graph)
        # Check that the relations have been updated
        self.assertEqual(2, len(self.scene_graph.relations_by_rule_id[1]))

        # Validate after 1st split
        success = self.scene_graph.validate(self.logger)
        self.assertTrue(success)

        # 2nd merge
        composite = merge_with_mask(self.scene_graph, self.seg1, self.seg2)
        # Check that the relations have been updated
        self.assertEqual(1, len(self.scene_graph.relations_by_rule_id[1]))

        # Validate after 2nd merge
        success = self.scene_graph.validate(self.logger)
        self.assertTrue(success)

        # 2nd split
        split_with_mask(composite, self.scene_graph)
        # Check that the relations have been updated
        self.assertEqual(2, len(self.scene_graph.relations_by_rule_id[1]))

        # Validate after 2nd split
        success = self.scene_graph.validate(self.logger)
        self.assertTrue(success)

        self.assertEqual(0, self.handler.get_warning_message_count())
        self.assertEqual(0, self.handler.get_error_message_count())
