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

import unittest

import numpy as np
import torch
from nibabel import Nifti1Image

from scene_graph_api.knowledge import ObjectClass, RelationRule, NaturalImageKG, IntAttribute, FloatAttribute, \
    StrAttribute, BoolAttribute, EnumAttribute
from scene_graph_api.scene import SceneGraph, BoundingBox, Relation, ObjectAttribute
from scene_graph_api.tensor_structures import BoxList, BinaryMaskList, FieldExtractor, BoxListConverter


# noinspection DuplicatedCode
class TestBoxList(unittest.TestCase):
    def setUp(self):
        boxes = torch.tensor([
            [0, 0, 100, 99],
            [5, 5, 40, 40]
        ], dtype=torch.float32)
        self.boxlist = BoxList(boxes, (150, 100), mode=BoxList.Mode.zyxzyx)
        mask = torch.zeros((150, 100))
        mask[1, 2] = 1
        self.boxlist_with_mask = BoxList(boxes, (150, 100), mode=BoxList.Mode.zyxzyx)
        self.boxlist_with_mask.add_field(BoxList.AnnotationField.MASKS, BinaryMaskList(mask, (150, 100)))
        self.labelmap = torch.tensor([1, 2, 3, 4]).view((2, 2))
        self.boxlist_with_labelmap = self.boxlist[:]
        self.boxlist_with_labelmap.add_field(BoxList.AnnotationField.LABELMAP, self.labelmap, indexing_power=0)

    def test_add_field(self):
        boxlist = BoxList(torch.tensor([[1, 1]]), (1,), )
        for i in range(3):
            boxlist.add_field(str(i), i * 2, indexing_power=i)
            self.assertTrue(str(i) in boxlist.extra_fields)
            self.assertEqual(boxlist.extra_fields[str(i)], i * 2)
            self.assertTrue(str(i) in boxlist.fields_indexing_power)
            self.assertEqual(boxlist.fields_indexing_power[str(i)], i)

    def test_add_field_shorthand(self):
        boxlist = BoxList(torch.tensor([[1, 1]]), (1,), )
        # Annotation field
        boxlist.AFFINE_MATRIX = torch.eye(4)
        self.assertTrue(BoxList.AnnotationField.AFFINE_MATRIX in boxlist.extra_fields)
        self.assertEqual(0, boxlist.fields_indexing_power[BoxList.AnnotationField.AFFINE_MATRIX])
        # Prediction field
        boxlist.PRED_SEGMENTATION = torch.eye(4)
        self.assertTrue(BoxList.PredictionField.PRED_SEGMENTATION in boxlist.extra_fields)
        self.assertEqual(0, boxlist.fields_indexing_power[BoxList.PredictionField.PRED_SEGMENTATION])

    def test_get_field_shorthand(self):
        boxlist = BoxList(torch.tensor([[1, 1]]), (1,), )
        # Annotation field
        boxlist.AFFINE_MATRIX = torch.eye(4)
        self.assertIsNotNone(boxlist.AFFINE_MATRIX)
        # Prediction field
        boxlist.PRED_SEGMENTATION = torch.eye(4)
        self.assertIsNotNone(boxlist.PRED_SEGMENTATION)

    def test_add_field_negative_indexing_power(self):
        with self.assertRaises(ValueError):
            self.boxlist.add_field(1, 1, -1)

    def test_get_item_slice(self):
        boxlist = BoxList(torch.tensor([[1, 1]]), (1,), )
        boxlist.add_field("0", 0, indexing_power=0)
        boxlist.add_field("1", torch.tensor([1]), indexing_power=1)
        boxlist.add_field("2", torch.tensor([[2]]), indexing_power=2)
        boxlist = boxlist[:]
        # Test that the indexing power is correctly registered after indexing
        for i in range(3):
            self.assertTrue(str(i) in boxlist.fields_indexing_power)
            self.assertEqual(boxlist.fields_indexing_power[str(i)], i)

    def test_get_item_int_index(self):
        # Note: this is not supported because it changes the dimensionality of the boxes
        # and this should be done on .boxes directly
        with self.assertRaises(ValueError):
            # noinspection PyTypeChecker
            _ = self.boxlist[0]
        # indexed = self.boxlist[0]
        # expected_boxes = torch.tensor([
        #     [0, 0, 101, 101],
        # ], dtype=torch.float32)
        # self.assertEqual(1, len(indexed))
        # torch.testing.assert_close(expected_boxes, indexed.boxes)

    def test_get_item_with_mask_int_index(self):
        # Note: this is not supported because it changes the dimensionality of the boxes
        # and this should be done on .boxes directly
        with self.assertRaises(ValueError):
            # noinspection PyTypeChecker
            _ = self.boxlist_with_mask[0]
        # indexed = self.boxlist_with_mask[0]
        # self.assertTrue(indexed.has_field(BoxList.AnnotationField.MASKS))
        # self.assertEqual(1, len(indexed.get_field(BoxList.AnnotationField.MASKS)))

    def test_get_item_int_tensor_index(self):
        indexed = self.boxlist[torch.tensor([0, 1, 1, 0])]
        expected_boxes = torch.tensor([
            [0, 0, 100, 99],
            [5, 5, 40, 40],
            [5, 5, 40, 40],
            [0, 0, 100, 99]
        ], dtype=torch.float32)
        self.assertEqual(4, len(indexed))
        torch.testing.assert_close(expected_boxes, indexed.boxes)

    def test_get_item_bool_tensor_index(self):
        indexed = self.boxlist[torch.tensor([False, True])]
        expected_boxes = torch.tensor([
            [5, 5, 40, 40],
        ], dtype=torch.float32)
        self.assertEqual(1, len(indexed))
        torch.testing.assert_close(expected_boxes, indexed.boxes)

    def test_get_item_with_pred_rel(self):
        boxlist = BoxList(torch.tensor([[1, 1], [1, 1], [1, 1], [1, 1]]), (1,), )
        boxlist.REL_PAIR_IDXS = torch.tensor([[0, 0], [0, 1], [2, 3], [3, 3]])
        boxlist.PRED_REL_LABELS = torch.tensor([1, 2, 3, 4])
        boxlist.PRED_REL_CLS_SCORES = torch.tensor([[1, 1], [2, 2], [3, 3], [4, 4]])

        with self.subTest("slice"):
            new_boxlist = boxlist[1:]
            torch.testing.assert_close(new_boxlist.REL_PAIR_IDXS, torch.tensor([[1, 2], [2, 2]]))
            torch.testing.assert_close(new_boxlist.PRED_REL_LABELS, torch.tensor([3, 4]))
            torch.testing.assert_close(new_boxlist.PRED_REL_CLS_SCORES, torch.tensor([[3, 3], [4, 4]]))

        with self.subTest("int tensor"):
            new_boxlist = boxlist[torch.tensor([2, 3, 2])]
            # Check that one relation has been sampled twice from box 2 being sampled twice
            torch.testing.assert_close(new_boxlist.REL_PAIR_IDXS, torch.tensor([[0, 1], [2, 1], [1, 1]]))
            torch.testing.assert_close(new_boxlist.PRED_REL_LABELS, torch.tensor([3, 3, 4]))
            torch.testing.assert_close(new_boxlist.PRED_REL_CLS_SCORES, torch.tensor([[3, 3], [3, 3], [4, 4]]))

        with self.subTest("bool tensor"):
            new_boxlist = boxlist[torch.tensor([False, True, True, True])]
            torch.testing.assert_close(new_boxlist.REL_PAIR_IDXS, torch.tensor([[1, 2], [2, 2]]))
            torch.testing.assert_close(new_boxlist.PRED_REL_LABELS, torch.tensor([3, 4]))
            torch.testing.assert_close(new_boxlist.PRED_REL_CLS_SCORES, torch.tensor([[3, 3], [4, 4]]))

    def test_get_item_tensor_index_preserve_labelmap(self):
        indexed: BoxList = self.boxlist_with_labelmap[torch.tensor([0, 1, 1, 0])]
        torch.testing.assert_close(indexed.get_field(BoxList.AnnotationField.LABELMAP), self.labelmap)

    def test_get_item_with_mask_tensor_index(self):
        indexed = self.boxlist_with_mask[torch.tensor([0, 0])]
        self.assertTrue(indexed.has_field(BoxList.AnnotationField.MASKS))
        self.assertEqual(2, len(indexed.get_field(BoxList.AnnotationField.MASKS)))

    def test_get_item_with_indexing_power2(self):
        boxes = torch.tensor([5, 15] * 10, dtype=torch.float32).reshape(10, 2)
        boxlist = BoxList(boxes, (20,), mode=BoxList.Mode.zyxzyx)
        relations = torch.zeros((10, 10), dtype=torch.uint8)
        relations[0, 3] = 1
        boxlist.add_field(BoxList.AnnotationField.RELATIONS, relations, indexing_power=2)

        indexed = boxlist[torch.tensor([0, 3])]
        self.assertTrue(indexed.has_field(BoxList.AnnotationField.RELATIONS))
        expected_relations = torch.tensor([
            [0, 1],
            [0, 0]
        ], dtype=torch.uint8)
        torch.testing.assert_close(expected_relations, indexed.get_field(BoxList.AnnotationField.RELATIONS))

    def test_convert(self):
        converted = self.boxlist.convert(BoxList.Mode.zyxdhw)
        # Maximums are included
        expected_boxes = torch.tensor([
            [0, 0, 101, 100],
            [5, 5, 36, 36]
        ], dtype=torch.float32)
        torch.testing.assert_close(expected_boxes, converted.boxes)
        # Convert back
        torch.testing.assert_close(converted.convert(BoxList.Mode.zyxzyx).boxes, self.boxlist.boxes)

    def test_convert_with_mask(self):
        converted = self.boxlist_with_mask.convert(BoxList.Mode.zyxdhw)
        self.assertTrue(converted.has_field(BoxList.AnnotationField.MASKS))

    @staticmethod
    def _prepare_graph() -> tuple[SceneGraph, np.ndarray]:
        # Prepare knowledge and graph
        template = NaturalImageKG(
            classes=[ObjectClass(1, "", [], has_mask=True)],
            rules=[RelationRule(1, "")]
        )

        labelmap = np.array([
            [[1, 1, 0],
             [2, 1, 0]],
            [[0, 0, 0],
             [2, 0, 0]]
        ], dtype=np.uint8)
        img = Nifti1Image(labelmap, np.eye(4))

        objects = [
            BoundingBox(1, 1, "", [], [[0, 0, 0], [0, 1, 1]]),
            BoundingBox(1, 2, "", [], [[0, 1, 0], [1, 1, 0]]),
        ]
        relations = [Relation(1, 1, 2)]

        graph = SceneGraph(
            template,
            img.affine,
            img.header,
            objects,
            relations,
            labelmap
        )

        return graph, labelmap

    def test_from_scene_graph(self):
        graph, labelmap = self._prepare_graph()

        # Test content of BoxList
        # Size
        target = BoxListConverter(BoxList).from_scene_graph(graph)
        self.assertEqual((2, 2, 3), target.size)
        self.assertEqual(3, target.n_dim)

        # Affine matrix
        self.assertTrue(target.has_field(target.AnnotationField.AFFINE_MATRIX))
        torch.testing.assert_close(
            torch.tensor(graph.image_affine),
            target.get_field(target.AnnotationField.AFFINE_MATRIX)
        )
        self.assertEqual(0, target.fields_indexing_power[target.AnnotationField.AFFINE_MATRIX])

        # Boxes
        expected_boxes = torch.tensor([
            [0, 0, 0, 0, 1, 1],
            [0, 1, 0, 1, 1, 0]
        ], dtype=torch.float32)
        torch.testing.assert_close(expected_boxes, target.boxes)

        # Masks
        mask1 = torch.tensor(labelmap == 1, dtype=torch.uint8)
        mask2 = torch.tensor(labelmap == 2, dtype=torch.uint8)
        masks: BinaryMaskList = FieldExtractor.masks(target)
        torch.testing.assert_close(mask1, masks.masks[0])
        torch.testing.assert_close(mask2, masks.masks[1])

        # Relations
        self.assertTrue(target.has_field(BoxList.AnnotationField.RELATIONS))
        expected_relations = torch.tensor(
            [
                [0, 1],
                [0, 0]
            ],
            dtype=torch.uint8
        )
        torch.testing.assert_close(expected_relations, target.get_field(BoxList.AnnotationField.RELATIONS))

        # Check that attributes were not added
        self.assertFalse(target.has_field(BoxList.AnnotationField.ATTRIBUTES))
        self.assertFalse(target.has_field(BoxList.AnnotationField.IMAGE_ATTRIBUTES))

    def test_from_scene_graph_unordered_objects(self):
        graph, labelmap = self._prepare_graph()
        # Swap the ordering of the boxes in the internal representation
        # The output should remain exactly the same
        graph._bb_by_id = {2: graph._bb_by_id[2], 1: graph._bb_by_id[1]}

        # Test content of BoxList
        # Size
        target = BoxListConverter(BoxList).from_scene_graph(graph)
        self.assertEqual((2, 2, 3), target.size)
        self.assertEqual(3, target.n_dim)

        # Boxes
        expected_boxes = torch.tensor([
            [0, 0, 0, 0, 1, 1],
            [0, 1, 0, 1, 1, 0]
        ], dtype=torch.float32)
        torch.testing.assert_close(expected_boxes, target.boxes)

        # Masks
        mask1 = torch.tensor(labelmap == 1, dtype=torch.uint8)
        mask2 = torch.tensor(labelmap == 2, dtype=torch.uint8)
        masks: BinaryMaskList = FieldExtractor.masks(target)
        torch.testing.assert_close(mask1, masks.masks[0])
        torch.testing.assert_close(mask2, masks.masks[1])

        # Relations
        self.assertTrue(target.has_field(BoxList.AnnotationField.RELATIONS))
        expected_relations = torch.tensor(
            [
                [0, 1],
                [0, 0]
            ],
            dtype=torch.uint8
        )
        torch.testing.assert_close(expected_relations, target.get_field(BoxList.AnnotationField.RELATIONS))

        # Check that attributes were not added
        self.assertFalse(target.has_field(BoxList.AnnotationField.ATTRIBUTES))
        self.assertFalse(target.has_field(BoxList.AnnotationField.IMAGE_ATTRIBUTES))

    def test_labelmap_from_scene_graph(self):
        graph, labelmap = self._prepare_graph()

        # Test content of BoxList
        # Size
        target = BoxListConverter(BoxList).from_scene_graph(graph)
        self.assertEqual((2, 2, 3), target.size)
        self.assertEqual(3, target.n_dim)

        # Boxes
        expected_boxes = torch.tensor([
            [0, 0, 0, 0, 1, 1],
            [0, 1, 0, 1, 1, 0]
        ], dtype=torch.float32)
        torch.testing.assert_close(expected_boxes, target.boxes)

        # Affine matrix
        self.assertTrue(target.has_field(target.AnnotationField.AFFINE_MATRIX))
        torch.testing.assert_close(
            torch.tensor(graph.image_affine),
            target.get_field(target.AnnotationField.AFFINE_MATRIX)
        )
        self.assertEqual(0, target.fields_indexing_power[target.AnnotationField.AFFINE_MATRIX])

        # Masks
        self.assertTrue(target.has_field(BoxList.AnnotationField.LABELMAP))
        continuous_graph = graph.remap_ids_as_contiguous()
        torch.testing.assert_close(target.get_field(BoxList.AnnotationField.LABELMAP),
                                   torch.from_numpy(continuous_graph.object_labelmap))

        # Relations
        self.assertTrue(target.has_field(BoxList.AnnotationField.RELATIONS))
        expected_relations = torch.tensor(
            [
                [0, 1],
                [0, 0]
            ],
            dtype=torch.uint8
        )
        torch.testing.assert_close(expected_relations, target.get_field(BoxList.AnnotationField.RELATIONS))

    def test_labelmap_from_masks_from_labelmap(self):
        graph, labelmap = self._prepare_graph()
        # Get compressed masks from graph, expand masks and recompress
        compressed_target = BoxListConverter(BoxList).from_scene_graph(graph)
        orig_labelmap = compressed_target.LABELMAP
        compressed_target.MASKS = FieldExtractor.masks(compressed_target)
        compressed_target.del_field(BoxList.AnnotationField.LABELMAP)
        labelmap = FieldExtractor.labelmap(compressed_target)

        torch.testing.assert_close(orig_labelmap, labelmap)

    @staticmethod
    def _prepare_graph_with_attributes() -> SceneGraph:
        # Prepare knowledge and graph
        template = NaturalImageKG(
            classes=[
                ObjectClass(1, "", [
                    FloatAttribute(2, "2"),
                    StrAttribute(3, "3"),
                    IntAttribute(1, "1"),

                ], has_mask=True),
                ObjectClass(2, "", [
                    BoolAttribute(1, "1"),
                    EnumAttribute(2, "2", ["a", "b", 3]),
                ], has_mask=True),
            ],
            image_level_attributes=[]
        )

        # Check that we can handle object lists that are not ordered and with alternating object classes
        objects = [
            BoundingBox(1, 1, "", [
                ObjectAttribute(attr_id=2, value=1.),
                ObjectAttribute(attr_id=3, value="1."),
                ObjectAttribute(attr_id=1, value=1),
            ], [[0, 0, 0], [0, 1, 1]]),
            BoundingBox(1, 3, "", [
                ObjectAttribute(attr_id=2, value=2.),
                ObjectAttribute(attr_id=3, value="2."),
                ObjectAttribute(attr_id=1, value=2),
            ], [[0, 0, 0], [0, 1, 2]]),
            BoundingBox(2, 2, "", [
                ObjectAttribute(attr_id=1, value=True),
                ObjectAttribute(attr_id=2, value="b"),
            ], [[0, 1, 0], [1, 1, 0]]),
            BoundingBox(2, 4, "", [
                ObjectAttribute(attr_id=1, value=False),
                ObjectAttribute(attr_id=2, value=3),
            ], [[0, 1, 0], [1, 2, 0]]),
        ]

        img = Nifti1Image(np.zeros((5, 5, 5), dtype=np.uint8), np.eye(4))
        graph = SceneGraph(
            template,
            img.affine,
            img.header,
            objects,
            [],
            np.zeros((5, 5, 5))
        )

        return graph

    def test_from_scene_graph_attributes(self):
        graph = self._prepare_graph_with_attributes()

        # Test content of BoxList
        # Size
        target = BoxListConverter(BoxList).from_scene_graph(graph)

        # Boxes: this will check that the box ordering is preserved
        expected_boxes = torch.tensor([
            [0, 0, 0, 0, 1, 1],
            [0, 1, 0, 1, 1, 0],
            [0, 0, 0, 0, 1, 2],
            [0, 1, 0, 1, 2, 0],
        ], dtype=torch.float32)
        torch.testing.assert_close(expected_boxes, target.boxes)

        # Check that attributes were added
        self.assertTrue(target.has_field(BoxList.AnnotationField.ATTRIBUTES))
        expected_attributes = torch.tensor([
            [1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1],
            [2, 0, 0, 0, 0],
            [0, 0, 0, 0, 2],
        ], dtype=torch.long)
        torch.testing.assert_close(expected_attributes, target.ATTRIBUTES)
