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

import torch

from scene_graph_api.tensor_structures import BoxList, BinaryMaskList, BoxListOps


class TestBoxListOps(unittest.TestCase):
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

    def test_crop(self):
        cropped = BoxListOps.crop(self.boxlist, (25, 25, 44, 39))
        self.assertEqual((20, 15), cropped.size)
        expected_boxes = torch.tensor([
            [0, 0, 19, 14],
            [0, 0, 15, 14]
        ], dtype=torch.float32)
        torch.testing.assert_close(expected_boxes, cropped.boxes)

    def test_crop_with_mask(self):
        cropped = BoxListOps.crop(self.boxlist_with_mask, (25, 25, 44, 39))
        self.assertTrue(cropped.has_field(BoxList.AnnotationField.MASKS))
        mask_list = cropped.get_field(BoxList.AnnotationField.MASKS)
        self.assertEqual((20, 15), mask_list.size)
        self.assertEqual((20, 15), mask_list.get_mask_tensor().size())

    def test_crop_with_labelmap_segmentation(self):
        bbox = self.boxlist_with_mask[:]
        labelmap = torch.zeros((150, 100))
        bbox.add_field(bbox.AnnotationField.LABELMAP, labelmap, indexing_power=0)
        segmentation = torch.zeros((150, 100))
        bbox.add_field(bbox.AnnotationField.SEGMENTATION, segmentation, indexing_power=0)

        cropped = BoxListOps.crop(bbox, (25, 25, 44, 39))
        self.assertTrue(cropped.has_field(bbox.AnnotationField.LABELMAP))
        tensor = cropped.get_field(bbox.AnnotationField.LABELMAP)
        self.assertEqual((20, 15), tensor.size())
        self.assertTrue(cropped.has_field(bbox.AnnotationField.SEGMENTATION))
        tensor = cropped.get_field(bbox.AnnotationField.SEGMENTATION)
        self.assertEqual((20, 15), tensor.size())

    def test_resize(self):
        resized = BoxListOps.resize(self.boxlist, (50, 20))
        self.assertEqual((50, 20), resized.size)
        expected_boxes = torch.tensor([
            [0, 0, 101 / 3, 100 / 5],
            [5 / 3, 5 / 5, 36 / 3, 36 / 5]
        ], dtype=torch.float32)
        torch.testing.assert_close(expected_boxes, resized.convert(BoxList.Mode.zyxdhw).boxes)

    def test_resize_same_ratio(self):
        resized = BoxListOps.resize(self.boxlist, (30, 20))
        self.assertEqual((30, 20), resized.size)
        expected_boxes = torch.tensor([
            [0, 0, 20, 19.8],
            [1, 1, 8, 8]
        ], dtype=torch.float32)
        torch.testing.assert_close(expected_boxes, resized.boxes)

    def test_resize_with_mask(self):
        resized = BoxListOps.resize(self.boxlist_with_mask, (50, 20))
        self.assertTrue(resized.has_field(BoxList.AnnotationField.MASKS))
        mask_list = resized.get_field(BoxList.AnnotationField.MASKS)
        self.assertEqual((50, 20), mask_list.size)
        self.assertEqual((50, 20), mask_list.get_mask_tensor().size())

    def test_hflip(self):
        flipped = BoxListOps.flip(self.boxlist, BoxList.FlipDim.WIDTH)
        expected_boxes = torch.tensor([
            [0, 0, 100, 99],
            [5, 59, 40, 94]
        ], dtype=torch.float32)
        torch.testing.assert_close(expected_boxes, flipped.boxes)

    def test_hflip_with_mask(self):
        flipped = BoxListOps.flip(self.boxlist_with_mask, BoxList.FlipDim.WIDTH)
        self.assertTrue(flipped.has_field(BoxList.AnnotationField.MASKS))
        mask = flipped.get_field(BoxList.AnnotationField.MASKS).get_mask_tensor()
        self.assertEqual(0, mask[1, 2].item())
        self.assertEqual(1, mask[1, -3].item())

    def test_hflip_with_labelmap_segmentation(self):
        bbox = self.boxlist_with_mask[:]
        labelmap = torch.zeros((150, 100))
        labelmap[2, 1] = 1
        bbox.add_field(bbox.AnnotationField.LABELMAP, labelmap, indexing_power=0)
        segmentation = torch.zeros((150, 100))
        segmentation[2, 1] = 2
        bbox.add_field(bbox.AnnotationField.SEGMENTATION, segmentation, indexing_power=0)

        flipped = BoxListOps.flip(bbox, BoxList.FlipDim.WIDTH)
        self.assertTrue(flipped.has_field(bbox.AnnotationField.LABELMAP))
        self.assertTrue(flipped.has_field(bbox.AnnotationField.SEGMENTATION))
        labelmap = flipped.get_field(bbox.AnnotationField.LABELMAP)
        self.assertEqual(1, labelmap[2, -2].item())
        segmentation = flipped.get_field(bbox.AnnotationField.SEGMENTATION)
        self.assertEqual(2, segmentation[2, -2].item())

    def test_centers(self):
        boxes = torch.tensor([
            [0, 0, 0, 0],
            [1, 0, 1, 1]
        ]).float()
        centers = BoxListOps.centers(BoxList(boxes, (50, 50)))
        expected_centers = torch.tensor([
            [0, 0],
            [1, .5]
        ]).float()
        torch.testing.assert_close(centers, expected_centers)

    @unittest.skipIf(not torch.cuda.is_available(), "no CUDA detected")
    def test_centers_cuda(self):
        boxes = torch.tensor([
            [0, 0, 0, 0],
            [1, 0, 1, 1]
        ]).float().cuda()
        centers = BoxListOps.centers(BoxList(boxes, (50, 50)))
        expected_centers = torch.tensor([
            [0, 0],
            [1, .5]
        ]).float().cuda()
        torch.testing.assert_close(centers, expected_centers)

    def test_clip_to_image(self):
        boxes = torch.tensor([
            [1, 1, 50, 10],  # Full
            [51, 1, 101, 10],  # Outside Y
            [51, 11, 101, 20],  # Fully outside
            [1, 2, 50, 2]  # Slice
        ])
        boxlist = BoxList(boxes, (51, 11), BoxList.Mode.zyxzyx)

        boxlist = BoxListOps.clip_to_image(boxlist, remove_empty=False)
        expected_box = torch.tensor([
            [1, 1, 50, 10],  # Full
            [50, 1, 50, 10],  # Now slice at the edge
            [50, 10, 50, 10],  # 1x1
            [1, 2, 50, 2]  # Slice
        ], dtype=torch.float32)
        torch.testing.assert_close(expected_box, boxlist.boxes)

        self.assertEqual(2, len(BoxListOps.clip_to_image(boxlist, remove_empty=True)))

    def test_area_zyxzyx(self):
        boxes = torch.tensor([
            [1, 1, 1, 1],  # 1x1
            [1, 1, 1, 10],  # Slice 1x10
            [1, 1, 10, 1],  # Slice 10x1
            [1, 1, 10, 10]  # Cube
        ], dtype=torch.float32)
        boxlist = BoxList(boxes, (20, 20), BoxList.Mode.zyxzyx)

        expected_area = torch.tensor([1, 10, 10, 100], dtype=torch.float32)
        torch.testing.assert_close(expected_area, BoxListOps.volume(boxlist))

    def test_area_zyxdhw(self):
        boxes = torch.tensor([
            [10, 10, 1, 1],  # 1x1
            [10, 10, 1, 10],  # Slice 1x10
            [10, 10, 1, 10],  # Slice 10x1
            [10, 10, 10, 10]  # Cube
        ])
        boxlist = BoxList(boxes, (20, 20), BoxList.Mode.zyxdhw)

        expected_area = torch.tensor([1, 10, 10, 100], dtype=torch.float32)
        torch.testing.assert_close(expected_area, BoxListOps.volume(boxlist))

    def test_area_empty_boxes(self):
        boxlist = BoxList(torch.tensor([]).reshape((0, 2)), (0,), BoxList.Mode.zyxzyx)
        torch.testing.assert_close(torch.tensor([]), BoxListOps.volume(boxlist))

    def test_remove_small_boxes(self):
        boxes = torch.tensor([
            [1, 1, 1, 1],  # 1x1
            [1, 1, 1, 10],  # Slice 1x10
            [1, 1, 10, 1],  # Slice 10x1
            [1, 1, 10, 10]  # Cube
        ])
        boxlist = BoxList(boxes, (20, 20), BoxList.Mode.zyxzyx)

        self.assertEqual(4, len(BoxListOps.remove_small_boxes(boxlist, 0)))
        self.assertEqual(4, len(BoxListOps.remove_small_boxes(boxlist, 1)))
        self.assertEqual(1, len(BoxListOps.remove_small_boxes(boxlist, 2)))
        self.assertEqual(1, len(BoxListOps.remove_small_boxes(boxlist, 10)))
        self.assertEqual(0, len(BoxListOps.remove_small_boxes(boxlist, 11)))
        self.assertEqual(4, len(BoxListOps.remove_small_boxes(boxlist, (0, 0))))
        self.assertEqual(2, len(BoxListOps.remove_small_boxes(boxlist, (1, 2))))
        self.assertEqual(2, len(BoxListOps.remove_small_boxes(boxlist, (2, 1))))
        self.assertEqual(1, len(BoxListOps.remove_small_boxes(boxlist, (2, 2))))

    def test_remove_small_boxes_empty_boxes(self):
        boxlist = BoxList(torch.tensor([]).reshape((0, 2)), (0,), BoxList.Mode.zyxzyx)
        torch.testing.assert_close(torch.tensor([]).reshape((0, 2)), BoxListOps.remove_small_boxes(boxlist, 0).boxes)

    def test_remove_small_boxes_with_labelmap_segmentation(self):
        boxes = torch.tensor([
            [1, 1, 1, 1],
            [1, 1, 6, 6],
            [1, 1, 5, 5],
            [1, 1, 10, 10]  # Cube
        ])
        boxlist = BoxList(boxes, (20, 20), BoxList.Mode.zyxzyx)
        boxlist.LABELMAP = torch.tensor([0, 1, 2, 3, 4]).int()
        boxlist.SEGMENTATION = torch.tensor([0, 1, 2, 1, 2]).int()

        new_boxlist = BoxListOps.remove_small_boxes(boxlist, 6)
        self.assertTrue(new_boxlist.has_field(boxlist.AnnotationField.LABELMAP))
        expected_labelmap = torch.tensor([0, 0, 1, 0, 2]).int()
        torch.testing.assert_close(new_boxlist.LABELMAP, expected_labelmap)
        self.assertTrue(new_boxlist.has_field(boxlist.AnnotationField.SEGMENTATION))
        expected_segmentation = torch.tensor([0, 0, 2, 0, 2]).int()
        torch.testing.assert_close(new_boxlist.SEGMENTATION, expected_segmentation)

    def test_iou(self):
        boxes = torch.tensor([
            [1, 1, 1, 1],
            [2, 2, 2, 2],
            [1, 1, 20, 10],
            [1, 1, 10, 20]
        ])
        boxlist = BoxList(boxes, (20, 20), BoxList.Mode.zyxzyx)

        expected_iou = torch.tensor([
            [1, 0, 1 / 200, 1 / 200],
            [0, 1, 1 / 200, 1 / 200],
            [1 / 200, 1 / 200, 1, 100 / 300],
            [1 / 200, 1 / 200, 100 / 300, 1]
        ], dtype=torch.float32)
        torch.testing.assert_close(expected_iou, BoxListOps.iou(boxlist, boxlist))

    def test_union(self):
        boxes1 = torch.tensor([
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 10, 20],
            [1, 1, 10, 10]
        ])
        boxlist1 = BoxList(boxes1, (20, 20), BoxList.Mode.zyxzyx)
        boxes2 = torch.tensor([
            [1, 1, 1, 1],
            [2, 2, 2, 2],
            [1, 1, 20, 10],
            [10, 10, 20, 20]
        ])
        boxlist2 = BoxList(boxes2, (20, 20), BoxList.Mode.zyxzyx)

        expected_union = torch.tensor([
            [1, 1, 1, 1],
            [1, 1, 2, 2],
            [1, 1, 20, 20],
            [1, 1, 20, 20]
        ], dtype=torch.float32)
        torch.testing.assert_close(expected_union, BoxListOps.union(boxlist1, boxlist2).boxes)

    def test_intersection(self):
        boxes1 = torch.tensor([
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 10, 20],
            [1, 1, 10, 10],
            [1, 1, 20, 20],
        ])
        boxlist1 = BoxList(boxes1, (20, 20), BoxList.Mode.zyxzyx)
        boxes2 = torch.tensor([
            [1, 1, 1, 1],
            [2, 2, 2, 2],
            [1, 1, 20, 10],
            [10, 10, 20, 20],
            [5, 5, 15, 15],
        ])
        boxlist2 = BoxList(boxes2, (20, 20), BoxList.Mode.zyxzyx)

        inter = BoxListOps.intersection(boxlist1, boxlist2, filter_empty=True)
        self.assertEqual(4, len(inter))  # Check that the invalid box has been removed
        expected_inter = torch.tensor([
            [1, 1, 1, 1],  # Self
            # Invalid was removed
            [1, 1, 10, 10],  # Corner of 2 rectangles
            [10, 10, 10, 10],  # Touching by the corner
            [5, 5, 15, 15],  # 2 in 1
        ], dtype=torch.float32)
        torch.testing.assert_close(expected_inter, inter.boxes)

    def test_cat(self):
        boxlists = [
            BoxList(torch.tensor([]).reshape((0, 2)), (1,), BoxList.Mode.zyxzyx),
            BoxList(torch.tensor([[1, 1]]), (1,), BoxList.Mode.zyxzyx),
            BoxList(torch.tensor([[2, 3]]), (1,), BoxList.Mode.zyxzyx),
        ]

        boxlist = BoxListOps.cat(boxlists)
        torch.testing.assert_close(torch.tensor([[1, 1], [2, 3]], dtype=torch.float32), boxlist.boxes)

    def test_cat_indexing_power0(self):
        boxlists = [
            BoxList(torch.tensor([]).reshape((0, 2)), (1,), BoxList.Mode.zyxzyx),
            BoxList(torch.tensor([[1, 1]]), (1,), BoxList.Mode.zyxzyx),
            BoxList(torch.tensor([[2, 3]]), (1,), BoxList.Mode.zyxzyx),
        ]

        for bb in boxlists:
            bb.add_field("test", 0, indexing_power=0)

        boxlist = BoxListOps.cat(boxlists)
        self.assertTrue(boxlist.has_field("test"))
        self.assertEqual(boxlist.get_field("test"), 0)

    def test_cat_indexing_power1(self):
        boxlists = [
            BoxList(torch.tensor([]).reshape((0, 2)), (1,), BoxList.Mode.zyxzyx),
            BoxList(torch.tensor([[1, 1]]), (1,), BoxList.Mode.zyxzyx),
            BoxList(torch.tensor([[2, 3]]), (1,), BoxList.Mode.zyxzyx),
        ]

        for bb in boxlists:
            bb.add_field("test", torch.tensor([0, 1]), indexing_power=1)

        boxlist = BoxListOps.cat(boxlists)
        self.assertTrue(boxlist.has_field("test"))
        torch.testing.assert_close(boxlist.get_field("test"), torch.tensor([0, 1] * len(boxlists)))

    def test_cat_indexing_power2(self):
        boxlists = [
            BoxList(torch.tensor([]).reshape((0, 2)), (1,), BoxList.Mode.zyxzyx),
            BoxList(torch.tensor([[1, 1]]), (1,), BoxList.Mode.zyxzyx),
            BoxList(torch.tensor([[2, 3]]), (1,), BoxList.Mode.zyxzyx),
        ]

        for bb in boxlists:
            bb.add_field("test", torch.tensor([[1.]]), indexing_power=2)

        boxlist = BoxListOps.cat(boxlists)
        self.assertTrue(boxlist.has_field("test"))
        torch.testing.assert_close(boxlist.get_field("test"), torch.eye(3))

    def test_cat_invalid_size(self):
        boxlists = [
            BoxList(torch.tensor([[1, 1]]), (1,), BoxList.Mode.zyxzyx),
            BoxList(torch.tensor([[1, 2]]), (2,), BoxList.Mode.zyxzyx),
        ]
        with self.assertRaises(AssertionError):
            BoxListOps.cat(boxlists)

    def test_cat_invalid_mode(self):
        boxlists = [
            BoxList(torch.tensor([[1, 3]]), (1,), BoxList.Mode.zyxzyx),
            BoxList(torch.tensor([[1, 4]]), (1,), BoxList.Mode.zyxdhw),
        ]
        with self.assertRaises(AssertionError):
            BoxListOps.cat(boxlists)

    def test_cat_different_fields(self):
        boxlists = [
            BoxList(torch.tensor([[1, 3]]), (1,), BoxList.Mode.zyxzyx),
            BoxList(torch.tensor([[1, 4]]), (1,), BoxList.Mode.zyxzyx),
        ]
        boxlists[0].add_field("f1", torch.tensor([1]), 0)
        boxlists[1].add_field("f2", torch.tensor([1]), 0)
        with self.assertRaises(AssertionError):
            BoxListOps.cat(boxlists)

    def test_affine_transformation_no_transform_with_labelmap(self):
        boxlist = BoxList(
            torch.tensor([
                [0, 0, 0, 0],
                [1, 0, 1, 1]
            ]),
            image_size=(2, 2),
            mode=BoxList.Mode.zyxzyx
        )
        labelmap = torch.tensor([
            [1, 0],
            [2, 2]
        ], dtype=torch.uint8)
        boxlist.LABELMAP = labelmap
        transformed = BoxListOps.affine_transformation(boxlist, output_labelmap=True)
        torch.testing.assert_close(transformed.boxes, boxlist.boxes)
        self.assertTrue(transformed.has_field(BoxList.AnnotationField.LABELMAP))
        self.assertFalse(transformed.has_field(BoxList.AnnotationField.MASKS))
        torch.testing.assert_close(transformed.LABELMAP, labelmap)
        self.assertTrue(boxlist.LABELMAP.dtype, torch.uint8)

    def test_affine_transformation_no_transform_with_labelmap_zyxdhw_boxes(self):
        boxlist = BoxList(
            torch.tensor([
                [0, 0, 0, 0],
                [1, 0, 1, 1]
            ]),
            image_size=(2, 2),
            mode=BoxList.Mode.zyxzyx
        ).convert(BoxList.Mode.zyxdhw)
        labelmap = torch.tensor([
            [1, 0],
            [2, 2]
        ], dtype=torch.uint8)
        boxlist.LABELMAP = labelmap
        transformed = BoxListOps.affine_transformation(boxlist, output_labelmap=True)
        torch.testing.assert_close(transformed.boxes, boxlist.boxes)
        self.assertTrue(transformed.has_field(BoxList.AnnotationField.LABELMAP))
        self.assertFalse(transformed.has_field(BoxList.AnnotationField.MASKS))
        torch.testing.assert_close(transformed.LABELMAP, labelmap)
        self.assertTrue(boxlist.LABELMAP.dtype, torch.uint8)

    def test_affine_transformation_translate_with_labelmap(self):
        boxlist = BoxList(
            torch.tensor([
                [0, 0, 0, 0],
                [1, 0, 1, 1]
            ]),
            image_size=(2, 2),
            mode=BoxList.Mode.zyxzyx
        )
        labelmap = torch.tensor([
            [1, 0],
            [2, 2]
        ], dtype=torch.uint8)
        boxlist.LABELMAP = labelmap
        transformed = BoxListOps.affine_transformation(boxlist, translate=(0, 1), output_labelmap=True)

        expected_boxes = torch.tensor([
            [0., 1., 0., 1.],
            [1., 1., 1., 1.]
        ])
        expected_labelmap = torch.tensor([
            [0, 1],
            [0, 2]
        ], dtype=torch.uint8)

        torch.testing.assert_close(transformed.boxes, expected_boxes)
        self.assertTrue(transformed.has_field(BoxList.AnnotationField.LABELMAP))
        torch.testing.assert_close(transformed.LABELMAP, expected_labelmap)

    def test_affine_transformation_rotate_with_labelmap(self):
        boxlist = BoxList(
            torch.tensor([
                [0, 0, 0, 0],
                [1, 0, 1, 1]
            ]),
            image_size=(2, 2),
            mode=BoxList.Mode.zyxzyx
        )
        labelmap = torch.tensor([
            [1, 0],
            [2, 2]
        ], dtype=torch.uint8)
        boxlist.LABELMAP = labelmap
        transformed = BoxListOps.affine_transformation(boxlist, rotate=(90,), output_labelmap=True)

        expected_boxes = torch.tensor([
            [1., 0., 1., 0.],
            [0., 1., 1., 1.]
        ])
        expected_labelmap = torch.tensor([
            [0, 2],
            [1, 2]
        ], dtype=torch.uint8)

        torch.testing.assert_close(transformed.boxes, expected_boxes)
        self.assertTrue(transformed.has_field(BoxList.AnnotationField.LABELMAP))
        torch.testing.assert_close(transformed.LABELMAP, expected_labelmap)

    def test_affine_transformation_scale_with_labelmap(self):
        boxlist = BoxList(
            torch.tensor([
                [1, 1, 2, 2]
            ]),
            image_size=(4, 4),
            mode=BoxList.Mode.zyxzyx
        )
        labelmap = torch.tensor([
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0]
        ], dtype=torch.uint8)
        boxlist.LABELMAP = labelmap
        transformed = BoxListOps.affine_transformation(boxlist, scale=(1, 2), output_labelmap=True)

        expected_boxes = torch.tensor([
            [1., 0., 2., 3.]
        ])
        expected_labelmap = torch.tensor([
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [0, 0, 0, 0]
        ], dtype=torch.uint8)

        torch.testing.assert_close(transformed.boxes, expected_boxes)
        self.assertTrue(transformed.has_field(BoxList.AnnotationField.LABELMAP))
        torch.testing.assert_close(transformed.LABELMAP, expected_labelmap)

    def test_affine_transformation_objects_out_of_frame(self):
        boxlist = BoxList(
            torch.tensor([
                [0, 0, 0, 0],
                [1, 0, 1, 1]
            ]),
            image_size=(2, 2),
            mode=BoxList.Mode.zyxzyx
        )
        labelmap = torch.tensor([
            [1, 0],
            [2, 2]
        ], dtype=torch.uint8)
        boxlist.LABELMAP = labelmap
        transformed = BoxListOps.affine_transformation(boxlist, translate=(0, -1), output_labelmap=True)

        expected_boxes = torch.tensor([
            [1., 0., 1., 0.]
        ])
        expected_labelmap = torch.tensor([
            [0, 0],
            [1, 0]
        ], dtype=torch.uint8)

        torch.testing.assert_close(transformed.boxes, expected_boxes)
        self.assertTrue(transformed.has_field(BoxList.AnnotationField.LABELMAP))
        torch.testing.assert_close(transformed.LABELMAP, expected_labelmap)

    def test_affine_transformation_objects_out_of_frame_with_attribute(self):
        boxlist = BoxList(
            torch.tensor([
                [0, 0, 0, 0],
                [1, 0, 1, 1]
            ]),
            image_size=(2, 2),
            mode=BoxList.Mode.zyxzyx
        )
        labelmap = torch.tensor([
            [1, 0],
            [2, 2]
        ], dtype=torch.uint8)
        boxlist.LABELMAP = labelmap
        boxlist.ATTRIBUTES = torch.tensor([1, 2]).long()
        transformed = BoxListOps.affine_transformation(boxlist, translate=(0, -1), output_labelmap=True)

        self.assertTrue(transformed.has_field(BoxList.AnnotationField.ATTRIBUTES))
        torch.testing.assert_close(transformed.ATTRIBUTES, torch.tensor([2]).long())

    def test_affine_transformation_objects_out_of_frame_with_relation(self):
        boxlist = BoxList(
            torch.tensor([
                [0, 0, 0, 0],
                [1, 0, 1, 1]
            ]),
            image_size=(2, 2),
            mode=BoxList.Mode.zyxzyx
        )
        labelmap = torch.tensor([
            [1, 0],
            [2, 2]
        ], dtype=torch.uint8)
        boxlist.LABELMAP = labelmap
        boxlist.RELATIONS = torch.tensor([[0, 2], [0, 1]]).long()
        transformed = BoxListOps.affine_transformation(boxlist, translate=(0, -1), output_labelmap=True)

        self.assertTrue(transformed.has_field(BoxList.AnnotationField.RELATIONS))
        torch.testing.assert_close(transformed.RELATIONS, torch.tensor([[1]]).long())

    def test_affine_transformation_objects_out_of_frame_raise(self):
        boxlist = BoxList(
            torch.tensor([
                [0, 0, 0, 0],
                [1, 0, 1, 1]
            ]),
            image_size=(2, 2),
            mode=BoxList.Mode.zyxzyx
        )
        labelmap = torch.tensor([
            [1, 0],
            [2, 2]
        ], dtype=torch.uint8)
        boxlist.LABELMAP = labelmap
        with self.assertRaises(RuntimeError):
            BoxListOps.affine_transformation(
                boxlist, translate=(0, -1), output_labelmap=True, raise_on_missing_bbox=True
            )

    def test_affine_transformation_no_transform_with_uint8_expanded_masks(self):
        boxlist = BoxList(
            torch.tensor([
                [0, 0, 0, 0],
                [1, 0, 1, 1]
            ]),
            image_size=(2, 2),
            mode=BoxList.Mode.zyxzyx
        )
        masks = BinaryMaskList(torch.tensor([
            [[1, 0],
             [0, 0]],
            [[0, 0],
             [1, 1]],
        ], dtype=torch.uint8), (2, 2))
        boxlist.MASKS = masks
        transformed = BoxListOps.affine_transformation(boxlist, output_labelmap=False, output_masks=True)
        torch.testing.assert_close(transformed.boxes, boxlist.boxes)
        self.assertTrue(transformed.has_field(BoxList.AnnotationField.MASKS))
        self.assertFalse(transformed.has_field(BoxList.AnnotationField.LABELMAP))
        torch.testing.assert_close(transformed.MASKS.masks, masks.masks)
        self.assertEqual(transformed.get_field(BoxList.AnnotationField.MASKS).masks.dtype, masks.masks.dtype)

    def test_affine_transformation_no_transform_with_float32_expanded_masks(self):
        boxlist = BoxList(
            torch.tensor([
                [0, 0, 0, 0],
                [1, 0, 1, 1]
            ]),
            image_size=(2, 2),
            mode=BoxList.Mode.zyxzyx
        )
        masks = BinaryMaskList(torch.tensor([
            [[1, 0],
             [0, 0]],
            [[0, 0],
             [1, 1]],
        ], dtype=torch.float32), (2, 2))
        boxlist.add_field(BoxList.AnnotationField.MASKS, masks)
        transformed = BoxListOps.affine_transformation(boxlist)
        self.assertEqual(transformed.MASKS.masks.dtype, masks.masks.dtype)

    @unittest.skipIf(not torch.cuda.is_available(), "no CUDA detected")
    def test_affine_transformation_no_transform_with_labelmap_cuda(self):
        boxlist = BoxList(
            torch.tensor([
                [0, 0, 0, 0],
                [1, 0, 1, 1]
            ]),
            image_size=(2, 2),
            mode=BoxList.Mode.zyxzyx
        )
        labelmap = torch.tensor([
            [1, 0],
            [2, 2]
        ], dtype=torch.uint8).cuda()
        boxlist.LABELMAP = labelmap
        boxlist = boxlist.to(device="cuda:0")
        transformed = BoxListOps.affine_transformation(boxlist)
        # Note: also tests the device
        torch.testing.assert_close(transformed.boxes, boxlist.boxes)
        torch.testing.assert_close(transformed.LABELMAP, labelmap)

    @unittest.skipIf(not torch.cuda.is_available(), "no CUDA detected")
    def test_affine_transformation_no_transform_with_semantic_segmentation_cuda(self):
        boxlist = BoxList(
            torch.tensor([
                [0, 0, 0, 0],
                [1, 0, 1, 1]
            ]),
            image_size=(2, 2),
            mode=BoxList.Mode.zyxzyx
        )
        labelmap = torch.tensor([
            [1, 0],
            [2, 2]
        ], dtype=torch.uint8)
        boxlist.LABELMAP = labelmap
        seg = torch.tensor([
            [2, 0],
            [1, 1]
        ], dtype=torch.uint8)
        labels = torch.tensor([2, 1], dtype=torch.uint8)
        boxlist.LABELS = labels
        boxlist = boxlist.to("cuda:0")
        transformed = BoxListOps.affine_transformation(boxlist, output_labelmap=False, output_segmentation=True)
        self.assertFalse(transformed.has_field(BoxList.AnnotationField.LABELMAP))
        # Note: also tests the device
        torch.testing.assert_close(transformed.boxes, boxlist.boxes)
        torch.testing.assert_close(transformed.SEGMENTATION, seg.cuda())

    @unittest.skipIf(not torch.cuda.is_available(), "no CUDA detected")
    def test_affine_transformation_no_transform_with_masks_cuda(self):
        boxlist = BoxList(
            torch.tensor([
                [0, 0, 0, 0],
                [1, 0, 1, 1]
            ]),
            image_size=(2, 2),
            mode=BoxList.Mode.zyxzyx
        )
        masks = BinaryMaskList(torch.tensor([
            [[1, 0],
             [0, 0]],
            [[0, 0],
             [1, 1]],
        ], dtype=torch.uint8), (2, 2)).to(device="cuda:0")
        boxlist.MASKS = masks
        boxlist = boxlist.to(device="cuda:0")
        transformed = BoxListOps.affine_transformation(boxlist)
        # Note: also tests the device
        torch.testing.assert_close(transformed.boxes, boxlist.boxes)
        torch.testing.assert_close(transformed.MASKS.masks, masks.masks)

    def test_affine_transformation_no_mask(self):
        with self.assertRaises(ValueError):
            BoxListOps.affine_transformation(BoxList(torch.tensor([[1, 3]]), (1,), BoxList.Mode.zyxzyx))
