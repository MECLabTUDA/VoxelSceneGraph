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
import shutil
import tempfile
from pathlib import Path
from unittest import TestCase

import nibabel as nib
import numpy as np

from scene_graph_api.knowledge import ObjectClass, RelationRule, WhitelistFilter, IntAttribute, NaturalImageKG
from scene_graph_api.logging_handlers import TestingHandler
from scene_graph_api.scene import Relation, SceneGraph, BoundingBox, ObjectAttribute
from scene_graph_api.utils.nifti_io import NiftiImageWrapper


# noinspection DuplicatedCode
class TestSceneGraph(TestCase):
    logger = logging.getLogger("scene/SceneGraph")
    handler = TestingHandler()
    # Define knowledge for some tests
    bb_class_id = 1
    seg_class_id = 2
    unique_class_id = 3
    ignored_class_id = 4
    rel_id = 1
    unknown_id = 42
    template_hash = 0
    template = NaturalImageKG(
        [
            ObjectClass(bb_class_id, attributes=[IntAttribute(1, "Attr")], has_mask=False),
            ObjectClass(seg_class_id, has_mask=True),
            ObjectClass(unique_class_id, has_mask=True, is_unique=True),
            ObjectClass(ignored_class_id, has_mask=True, is_ignored=True),
        ],
        [RelationRule(rel_id, "rule", WhitelistFilter([bb_class_id]), WhitelistFilter([bb_class_id]))],
        [],
        template_hash
    )

    template_with_il_attr = NaturalImageKG(
        [
            ObjectClass(bb_class_id, attributes=[IntAttribute(1, "Attr")], has_mask=False),
            ObjectClass(seg_class_id, has_mask=True),
            ObjectClass(unique_class_id, has_mask=True, is_unique=True),
            ObjectClass(ignored_class_id, has_mask=True, is_ignored=True),
        ],
        [RelationRule(rel_id, "rule", WhitelistFilter([bb_class_id]), WhitelistFilter([bb_class_id]))],
        [IntAttribute(1, "a")],
        template_hash
    )

    # Define object instances
    bb = BoundingBox(bb_class_id, 2, "", [ObjectAttribute(1, "Attr")], [[1, 2], [3, 4]])
    seg = BoundingBox(seg_class_id, 3, "", [], [[1, 2], [3, 4]])
    rel = Relation(rel_id, bb.id, bb.id)
    labelmap = np.array([[0, seg.id]])
    affine = np.eye(4)
    header = nib.Nifti1Header()

    @classmethod
    def setUpClass(cls):
        cls.logger.addHandler(cls.handler)

    def setUp(self):
        self.handler.purge()

    def test_create_from_labelmap_not_contiguous(self):
        # Check that attributes are init as the ObjectAttribute objects and not just the value
        # noinspection DuplicatedCode
        arr = np.array(
            [[[1, 0, 3],
              [1, 0, 3]]],
            dtype=np.uint8
        ).reshape((1, 2, 3))
        ids_to_class = {1: self.seg_class_id, 3: self.seg_class_id}
        labelmap = NiftiImageWrapper(nib.Nifti1Image(arr, np.eye(4)), True)
        res = SceneGraph.create_fom_labelmap(self.template, labelmap, ids_to_class, self.logger)
        self.assertIsNotNone(res)
        self.assertEqual(2, len(list(res.iter_bounding_boxes())))
        self.assertIsNotNone(res.get_bounding_box_by_id(1))
        self.assertIsNone(res.get_bounding_box_by_id(2))
        self.assertIsNotNone(res.get_bounding_box_by_id(3))
        self.assertTrue(res.validate(self.logger))

    def test_create_from_segmentation_check_attributes_init(self):
        # Check that attributes are init as the ObjectAttribute objects and not just the value
        # noinspection DuplicatedCode
        arr = np.array(
            [[[self.bb_class_id, 0, self.bb_class_id],
              [self.seg_class_id, 0, self.seg_class_id]]],
            dtype=np.uint8
        ).reshape((1, 2, 3))
        seg = NiftiImageWrapper(nib.Nifti1Image(arr, np.eye(4)), True)
        res = SceneGraph.create_from_segmentation(self.template, seg, self.logger)
        self.assertIsNotNone(res)
        for bb in res.bounding_boxes_by_class_id[self.bb_class_id]:
            self.assertEqual(1, len(bb.attributes))
            self.assertTrue(isinstance(bb.attributes[0], ObjectAttribute))
        self.assertEqual(4, len(list(res.iter_bounding_boxes())))
        self.assertTrue(res.validate(self.logger))

    def test_create_from_segmentation_check_image_level_attributes_init(self):
        # Check that attributes are init as the ObjectAttribute objects and not just the value
        # noinspection DuplicatedCode
        arr = np.array(
            [[[self.bb_class_id, 0, self.bb_class_id],
              [self.seg_class_id, 0, self.seg_class_id]]],
            dtype=np.uint8
        ).reshape((1, 2, 3))
        seg = NiftiImageWrapper(nib.Nifti1Image(arr, np.eye(4)), True)
        res = SceneGraph.create_from_segmentation(self.template_with_il_attr, seg, self.logger)
        self.assertIsNotNone(res)
        self.assertTrue(res.validate(self.logger))

    def test_create_from_segmentation_check_unique_class(self):
        # Check that attributes are init as the ObjectAttribute objects and not just the value
        arr = np.array(
            [[[self.unique_class_id, 0, self.unique_class_id],
              [0, 0, 0]]],
            dtype=np.uint8
        ).reshape((1, 2, 3))
        seg = NiftiImageWrapper(nib.Nifti1Image(arr, np.eye(4)), True)
        res = SceneGraph.create_from_segmentation(self.template, seg, self.logger)
        self.assertIsNotNone(res)
        self.assertEqual(1, len(list(res.iter_bounding_boxes())))
        self.assertTrue(res.validate(self.logger))

    def test_create_from_segmentation_check_ignored_class(self):
        # Check that attributes are init as the ObjectAttribute objects and not just the value
        arr = np.array(
            [[[self.bb_class_id, self.ignored_class_id, self.bb_class_id],
              [self.seg_class_id, self.ignored_class_id, self.seg_class_id]]],
            dtype=np.uint8
        ).reshape((1, 2, 3))
        seg = NiftiImageWrapper(nib.Nifti1Image(arr, np.eye(4)), True)
        res = SceneGraph.create_from_segmentation(self.template, seg, self.logger)
        self.assertIsNotNone(res)
        self.assertEqual(4, len(list(res.iter_bounding_boxes())))
        self.assertTrue(res.validate(self.logger))

    def test_create_from_segmentation_save_load(self):
        # Check that attributes are init as the ObjectAttribute objects and not just the value
        # noinspection DuplicatedCode
        arr = np.array([[[self.bb_class_id, 0, self.bb_class_id],
                         [self.seg_class_id, 0, self.seg_class_id]]], dtype=np.uint8).reshape((1, 2, 3))
        seg = NiftiImageWrapper(nib.Nifti1Image(arr, np.eye(4)), True)
        res = SceneGraph.create_from_segmentation(self.template, seg, self.logger)

        target_folder = Path(tempfile.mkdtemp())
        try:
            target = target_folder / "test.json"
            self.assertFalse(target.is_file())
            # Check that no attribute is a np-type (e.g. np.int32) that would cause the serialization to fail
            res.save(target.as_posix())
            self.assertTrue(target.is_file())
            res = SceneGraph.load(target.as_posix(), self.template, self.logger)
            self.assertIsNotNone(res)
            success = res.validate(self.logger)
            self.assertTrue(success)
            self.assertEqual(0, self.handler.get_warning_message_count())
            self.assertEqual(0, self.handler.get_error_message_count())
        finally:
            shutil.rmtree(target_folder)
