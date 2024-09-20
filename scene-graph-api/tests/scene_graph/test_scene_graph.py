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

import gzip
import logging
import shutil
import tempfile
from pathlib import Path
from unittest import TestCase

import nibabel as nib
import numpy as np
import numpy.testing

from scene_graph_api.knowledge import ObjectClass, RelationRule, WhitelistFilter, StrAttribute, \
    NaturalImageKG, IntAttribute
from scene_graph_api.logging_handlers import TestingHandler
from scene_graph_api.scene import Relation, SceneGraph, BoundingBox, ObjectAttribute
# noinspection PyProtectedMember
from scene_graph_api.utils.nifti_io import NiftiImageWrapper
from scene_graph_api.utils.parsing import get_validator


# noinspection DuplicatedCode
class TestSceneGraph(TestCase):
    logger = logging.getLogger("scene/SceneGraph")
    handler = TestingHandler()
    # Define knowledge for some tests
    bb_class_id = 1
    seg_class_id = 2
    rel_id = 1
    unknown_id = 42
    template_hash = 0
    template = NaturalImageKG(
        [ObjectClass(bb_class_id, attributes=[StrAttribute(1, "Attr")], has_mask=False),
         ObjectClass(seg_class_id, name="Mask", has_mask=True)],
        [RelationRule(rel_id, "rule", WhitelistFilter([bb_class_id]), WhitelistFilter([bb_class_id]))],
        [],
        template_hash
    )

    template_with_il_attr = NaturalImageKG(
        [ObjectClass(bb_class_id, attributes=[StrAttribute(1, "Attr")], has_mask=False),
         ObjectClass(seg_class_id, name="Mask", has_mask=True)],
        [RelationRule(rel_id, "rule", WhitelistFilter([bb_class_id]), WhitelistFilter([bb_class_id]))],
        [IntAttribute(1, "a")],
        template_hash
    )

    # Define object instances
    bb = BoundingBox(bb_class_id, 2, "", [ObjectAttribute(1, "Attr")], [[1, 2], [3, 4]])
    seg = BoundingBox(seg_class_id, 3, "", [], [[1, 2], [3, 4]])
    rel = Relation(rel_id, bb.id, bb.id)
    labelmap = np.array([[0, bb.id, seg.id]])
    affine = np.eye(4)
    header = nib.Nifti1Header()
    validator = get_validator(SceneGraph.schema(), registry=SceneGraph.SCHEMA_REGISTRY)

    @classmethod
    def setUpClass(cls):
        cls.logger.addHandler(cls.handler)

    def setUp(self):
        self.handler.purge()

    def test_from_json_valid_no_image_level_attribute(self):
        arr = np.zeros((3, 3))
        labelmap_str = NiftiImageWrapper(nib.Nifti1Image(arr, np.eye(4)), True).to_str()
        json_dict = {
            SceneGraph._bb_key: [
                BoundingBox(self.bb_class_id, 1, "", [ObjectAttribute(1, "dss")], [[1, 2], [2, 3]]).to_json(),
                BoundingBox(self.seg_class_id, 2, "", [], [[0, 1], [0, 1]]).to_json()],
            SceneGraph._rel_key: [Relation(1, 1, 3).to_json()],
            SceneGraph._labelmap_key: labelmap_str,
            SceneGraph._image_level_attributes_key: [],
            SceneGraph._knowledge_graph_hash_key: self.template_hash
        }
        self.assertEqual(0, len(list(self.validator.iter_errors(json_dict))))
        res = SceneGraph.from_json(self.template, json_dict, self.logger)
        self.assertIsNotNone(res)
        self.assertTrue(np.allclose(arr, res.object_labelmap))
        self.assertEqual(1, len(res.bounding_boxes_by_class_id[self.bb_class_id]))
        self.assertEqual(1, len(res.bounding_boxes_by_class_id[self.seg_class_id]))
        self.assertEqual(1, len(res.relations_by_rule_id[1]))
        self.assertTrue(isinstance(res.relations_by_rule_id[1][0], Relation))

    def test_from_json_valid(self):
        arr = np.zeros((3, 3))
        labelmap_str = NiftiImageWrapper(nib.Nifti1Image(arr, np.eye(4)), True).to_str()
        json_dict = {
            SceneGraph._bb_key: [
                BoundingBox(self.bb_class_id, 1, "", [ObjectAttribute(1, "dss")], [[1, 2], [2, 3]]).to_json(),
                BoundingBox(self.seg_class_id, 2, "", [], [[0, 1], [0, 1]]).to_json()],
            SceneGraph._rel_key: [Relation(1, 1, 3).to_json()],
            SceneGraph._labelmap_key: labelmap_str,
            SceneGraph._image_level_attributes_key: [ObjectAttribute(1, 2).to_json()],
            SceneGraph._knowledge_graph_hash_key: self.template_hash
        }
        self.assertEqual(0, len(list(self.validator.iter_errors(json_dict))))
        res = SceneGraph.from_json(self.template_with_il_attr, json_dict, self.logger)
        self.assertIsNotNone(res)
        self.assertTrue(np.allclose(arr, res.object_labelmap))
        self.assertEqual(1, len(res.bounding_boxes_by_class_id[self.bb_class_id]))
        self.assertEqual(1, len(res.bounding_boxes_by_class_id[self.seg_class_id]))
        self.assertEqual(1, len(res.relations_by_rule_id[1]))
        self.assertEqual(1, len(res.image.attributes))
        self.assertTrue(isinstance(res.image.attributes[0], ObjectAttribute))
        self.assertTrue(isinstance(res.relations_by_rule_id[1][0], Relation))

    def test_from_json_valid_but_labelmap_precision_loss(self):
        """Sometimes the labelmap loses precision and a value like 5.0 is stored as 4.999999999."""
        labelmap_str = NiftiImageWrapper(nib.Nifti1Image(np.array(1.9997).reshape((1, 1)), np.eye(4)), True).to_str()
        json_dict = {
            SceneGraph._bb_key: [BoundingBox(self.seg_class_id, 2, "", [], [[0, 1], [0, 1]]).to_json()],
            SceneGraph._rel_key: [],
            SceneGraph._labelmap_key: labelmap_str,
            SceneGraph._knowledge_graph_hash_key: self.template_hash
        }
        self.assertEqual(0, len(list(self.validator.iter_errors(json_dict))))
        res = SceneGraph.from_json(self.template, json_dict, self.logger)
        self.assertIsNotNone(res)
        self.assertEqual(res.object_labelmap.dtype, np.uint8)
        numpy.testing.assert_equal(res.object_labelmap, np.array(2))

    def test_from_json_different_template_hash(self):
        arr = np.zeros((3, 3))
        labelmap_str = NiftiImageWrapper(nib.Nifti1Image(arr, np.eye(4)), True).to_str()
        json_dict = {
            SceneGraph._bb_key: [],
            SceneGraph._rel_key: [],
            SceneGraph._labelmap_key: labelmap_str,
            SceneGraph._knowledge_graph_hash_key: self.template_hash + 1
        }
        # Does not get checked anymore
        self.assertEqual(0, len(list(self.validator.iter_errors(json_dict))))

    def test_from_json_missing_bb(self):
        json_dict = {
            SceneGraph._rel_key: [Relation(1, 1, 3).to_json()],
            SceneGraph._knowledge_graph_hash_key: self.template_hash,
            SceneGraph._labelmap_key: ""
        }
        self.assertEqual(1, len(list(self.validator.iter_errors(json_dict))))

    def test_from_json_missing_rel(self):
        json_dict = {
            SceneGraph._bb_key: [BoundingBox(1, 1, "", [ObjectAttribute(1, "sf")], [[1, 2], [2, 3]]).to_json()],
            SceneGraph._knowledge_graph_hash_key: self.template_hash,
            SceneGraph._labelmap_key: ""
        }
        self.assertEqual(1, len(list(self.validator.iter_errors(json_dict))))

    def test_from_json_missing_labelmap(self):
        json_dict = {
            SceneGraph._bb_key: [BoundingBox(1, 1, "", [ObjectAttribute(1, "sf")], [[1, 2], [2, 3]]).to_json()],
            SceneGraph._rel_key: [Relation(1, 1, 3).to_json()],
            SceneGraph._knowledge_graph_hash_key: self.template_hash,
        }
        self.assertEqual(1, len(list(self.validator.iter_errors(json_dict))))

    def test_from_json_bad_bb(self):
        json_dict = {
            SceneGraph._bb_key: [1],
            SceneGraph._rel_key: [],
            SceneGraph._labelmap_key: "",
            SceneGraph._knowledge_graph_hash_key: self.template_hash,
        }
        self.assertEqual(1, len(list(self.validator.iter_errors(json_dict))))

    def test_from_json_bad_rel(self):
        json_dict = {
            SceneGraph._bb_key: [],
            SceneGraph._rel_key: [1],
            SceneGraph._labelmap_key: "",
            SceneGraph._knowledge_graph_hash_key: self.template_hash,
        }
        self.assertEqual(1, len(list(self.validator.iter_errors(json_dict))))

    def test_from_json_bad_image_level_attribute(self):
        json_dict = {
            SceneGraph._bb_key: [],
            SceneGraph._rel_key: [],
            SceneGraph._image_level_attributes_key: [1],
            SceneGraph._labelmap_key: "",
            SceneGraph._knowledge_graph_hash_key: self.template_hash,
        }
        self.assertEqual(1, len(list(self.validator.iter_errors(json_dict))))

    def test_from_json_labelmap_not_gzipped(self):
        json_dict = {
            SceneGraph._bb_key: [],
            SceneGraph._rel_key: [],
            SceneGraph._labelmap_key: f"random_content",
            SceneGraph._knowledge_graph_hash_key: self.template_hash,
        }
        self.assertEqual(0, len(list(self.validator.iter_errors(json_dict))))
        SceneGraph.from_json(self.template, json_dict, self.logger)
        self.assertEqual(0, self.handler.get_warning_message_count())
        self.assertEqual(1, self.handler.get_error_message_count())

    def test_from_json_labelmap_gzipped_bad_content(self):
        json_dict = {
            SceneGraph._bb_key: [],
            SceneGraph._rel_key: [],
            SceneGraph._labelmap_key: gzip.compress(b"random_content").decode(NiftiImageWrapper._encoding),
            SceneGraph._knowledge_graph_hash_key: self.template_hash,
        }
        self.assertEqual(0, len(list(self.validator.iter_errors(json_dict))))
        SceneGraph.from_json(self.template, json_dict, self.logger)
        self.assertEqual(0, self.handler.get_warning_message_count())
        self.assertEqual(1, self.handler.get_error_message_count())

    def test_from_json_empty(self):
        self.assertEqual(3, len(list(self.validator.iter_errors({}))))

    def test_validate_success_no_image_level_attribute(self):
        graph = SceneGraph(
            graph=self.template,
            image_affine=self.affine,
            image_header=self.header,
            bounding_box_objects=[self.bb, self.seg],
            relations=[self.rel],
            object_labelmap=self.labelmap
        )
        success = graph.validate(self.logger)
        self.assertTrue(success)
        self.assertEqual(0, self.handler.get_warning_message_count())
        self.assertEqual(0, self.handler.get_error_message_count())

    def test_validate_success(self):
        graph = SceneGraph(
            graph=self.template_with_il_attr,
            image_affine=self.affine,
            image_header=self.header,
            bounding_box_objects=[self.bb, self.seg],
            relations=[self.rel],
            object_labelmap=self.labelmap,
            image_level_attributes=[ObjectAttribute(1, 1)]
        )
        success = graph.validate(self.logger)
        self.assertTrue(success)
        self.assertEqual(0, self.handler.get_warning_message_count())
        self.assertEqual(0, self.handler.get_error_message_count())

    def _validate_fail(self, graph: SceneGraph, errors: int = 2):
        success = graph.validate(self.logger)
        self.assertFalse(success)
        self.assertEqual(0, self.handler.get_warning_message_count())
        self.assertEqual(errors, self.handler.get_error_message_count())

    def test_validate_unknown_bb_class_id(self):
        graph = SceneGraph(
            graph=self.template,
            image_affine=self.affine,
            image_header=self.header,
            bounding_box_objects=[BoundingBox(0, 1, "", [ObjectAttribute(1, "Attr")], [[1, 2], [2, 3]])],
            relations=[],
            object_labelmap=np.array([1])
        )
        self._validate_fail(graph)

    def test_validate_unknown_image_level_attribute_class_id(self):
        graph = SceneGraph(
            graph=self.template,
            image_affine=self.affine,
            image_header=self.header,
            bounding_box_objects=[],
            relations=[],
            object_labelmap=np.array([0]),
            image_level_attributes=[ObjectAttribute(2, 2)]
        )
        self._validate_fail(graph, errors=1)

    def test_validate_unknown_bad_coordinate_length_first(self):
        graph = SceneGraph(
            graph=self.template,
            image_affine=self.affine,
            image_header=self.header,
            bounding_box_objects=[
                BoundingBox(self.bb_class_id, 1, "", [ObjectAttribute(1, "Attr")], [[1], [2, 3]])
            ],
            relations=[],
            object_labelmap=np.array([1])
        )
        self._validate_fail(graph, errors=2)

    def test_validate_unknown_bad_labelmap(self):
        graph = SceneGraph(
            graph=self.template,
            image_affine=self.affine,
            image_header=self.header,
            bounding_box_objects=[
                BoundingBox(self.bb_class_id, 1, "", [ObjectAttribute(1, "Attr")], [[1, 1], [2, 2]])
            ],
            relations=[],
            object_labelmap=np.empty((0, 0))
        )
        self._validate_fail(graph, errors=1)

    def test_validate_unknown_bad_coordinate_length_second(self):
        graph = SceneGraph(
            graph=self.template,
            image_affine=self.affine,
            image_header=self.header,
            bounding_box_objects=[
                BoundingBox(self.bb_class_id, 1, "", [ObjectAttribute(1, "Attr")], [[1, 2], [2]])
            ],
            relations=[],
            object_labelmap=np.array([1])
        )
        self._validate_fail(graph)

    def test_validate_unknown_rel_class_id(self):
        graph = SceneGraph(
            graph=self.template,
            image_affine=self.affine,
            image_header=self.header,
            bounding_box_objects=[],
            relations=[Relation(42, self.bb.id, self.bb.id)],
            object_labelmap=np.array([[0]])
        )
        success = graph.validate(self.logger)
        self.assertTrue(success)
        self.assertFalse(42 in graph.relations_by_rule_id)  # Relation should be filtered out
        self.assertEqual(0, self.handler.get_warning_message_count())
        self.assertEqual(0, self.handler.get_error_message_count())

    def test_validate_invalid_rel(self):
        graph = SceneGraph(
            graph=self.template,
            image_affine=self.affine,
            image_header=self.header,
            bounding_box_objects=[self.bb],
            relations=[Relation(self.rel_id, self.bb.id, self.seg.id)],
            object_labelmap=self.labelmap
        )
        self._validate_fail(graph, errors=2)

    def test_validate_two_bb_same_id(self):
        graph = SceneGraph(
            graph=self.template,
            image_affine=self.affine,
            image_header=self.header,
            bounding_box_objects=[self.bb, self.bb],
            relations=[],
            object_labelmap=self.labelmap
        )
        self._validate_fail(graph)

    def test_to_json(self):
        graph = SceneGraph(
            self.template,
            np.eye(4),
            nib.Nifti1Header(),
            [BoundingBox(self.bb_class_id, 1, "fff", [ObjectAttribute(1, "sf")], [[1, 2], [2, 3]]),
             BoundingBox(self.seg_class_id, 1, "sdf", [], [[0, 1], [0, 1]])],
            [Relation(1, 2, 3)],
            np.eye(3)
        )
        json_dict = graph.to_json()
        self.assertEqual(2, len(json_dict[SceneGraph._bb_key]))
        self.assertEqual(graph.bounding_boxes_by_class_id[self.bb_class_id][0].to_json(),
                         json_dict[SceneGraph._bb_key][0])
        self.assertEqual(graph.bounding_boxes_by_class_id[self.seg_class_id][0].to_json(),
                         json_dict[SceneGraph._bb_key][1])
        self.assertEqual(1, len(json_dict[SceneGraph._rel_key]))
        self.assertEqual(graph.relations_by_rule_id[1][0].to_json(), json_dict[SceneGraph._rel_key][0])
        self.assertNotEqual("", json_dict[SceneGraph._labelmap_key])

    def test_load_empty_file_deleted(self):
        target_folder = Path(tempfile.mkdtemp())
        try:
            target = target_folder / "test.json"
            # Create an empty file
            with open(target, "w+"):
                pass
            self.assertTrue(target.is_file())
            res = SceneGraph.load(target.as_posix(), self.template, self.logger)
            self.assertIsNone(res)
            self.assertFalse(target.is_file())
            self.assertEqual(0, self.handler.get_warning_message_count())
            self.assertEqual(1, self.handler.get_error_message_count())
        finally:
            shutil.rmtree(target_folder)

    def test_load_pathlike_empty_file_deleted(self):
        target_folder = Path(tempfile.mkdtemp())
        try:
            target = target_folder / "test.json"
            # Create an empty file
            with open(target, "w+"):
                pass
            self.assertTrue(target.is_file())
            res = SceneGraph.load(target, self.template, self.logger)
            self.assertIsNone(res)
            self.assertFalse(target.is_file())
            self.assertEqual(0, self.handler.get_warning_message_count())
            self.assertEqual(1, self.handler.get_error_message_count())
        finally:
            shutil.rmtree(target_folder)

    def test_remap_ids_as_contiguous_no_op(self):
        graph = SceneGraph(
            self.template,
            np.eye(4),
            nib.Nifti1Header(),
            [BoundingBox(self.bb_class_id, 1, "fff", [ObjectAttribute(1, "sf")], [[0, 0], [0, 0]]),
             BoundingBox(self.seg_class_id, 2, "sdf", [], [[0, 0], [0, 0]])],
            [Relation(1, 1, 2)],
            np.array([[2]])
        )
        graph.remap_ids_as_contiguous()
        # Check bb ids and shorthands
        self.assertIsNotNone(graph.get_bounding_box_by_id(1))
        self.assertEqual(1, graph.get_bounding_box_by_id(1).id)
        self.assertIsNotNone(graph.get_bounding_box_by_id(2))
        self.assertEqual(2, graph.get_bounding_box_by_id(2).id)
        self.assertTrue(1 in graph._bb_by_id)
        self.assertTrue(2 in graph._bb_by_id)
        # Check relation id and bb reference in relation
        self.assertTrue(graph.relations_by_rule_id[1])
        self.assertEqual(1, graph.relations_by_rule_id[1][0].subject_id)
        self.assertEqual(2, graph.relations_by_rule_id[1][0].object_id)
        # Check labelmap
        self.assertTrue(graph.object_labelmap[0, 0] == 2)

    def test_remap_ids_as_contiguous_remap(self):
        bb_to_be_renamed = BoundingBox(
            self.seg_class_id, 3,
            BoundingBox.default_name(self.template, self.seg_class_id, 3),
            [], [[0, 0], [0, 0]]
        )
        graph = SceneGraph(
            self.template,
            np.eye(4),
            nib.Nifti1Header(),
            [BoundingBox(self.bb_class_id, 4, "fff", [ObjectAttribute(1, "sf")], [[0, 0], [0, 0]]),
             bb_to_be_renamed],
            [Relation(1, 4, 3)],
            np.array([[3]])
        )
        graph.remap_ids_as_contiguous()
        # Check bb ids and shorthands
        self.assertIsNotNone(graph.get_bounding_box_by_id(1))
        self.assertEqual(1, graph.get_bounding_box_by_id(1).id)
        self.assertIsNotNone(graph.get_bounding_box_by_id(2))
        self.assertEqual(2, graph.get_bounding_box_by_id(2).id)
        self.assertTrue(1 in graph._bb_by_id)
        self.assertTrue(2 in graph._bb_by_id)
        # Check that object with orig id 3 has been renamed
        self.assertEqual(bb_to_be_renamed.name, BoundingBox.default_name(self.template, self.seg_class_id, 2))
        # Check relation id and bb reference in relation
        self.assertTrue(graph.relations_by_rule_id[1])
        self.assertEqual(1, graph.relations_by_rule_id[1][0].subject_id)
        self.assertEqual(2, graph.relations_by_rule_id[1][0].object_id)
        # Check labelmap
        self.assertTrue(graph.object_labelmap[0, 0] == 2)
        # Check that previous references are gone
        self.assertIsNone(graph.get_bounding_box_by_id(4))
        self.assertIsNone(graph.get_bounding_box_by_id(3))
        self.assertFalse(4 in graph._bb_by_id)
        self.assertFalse(3 in graph._bb_by_id)

    def test_repr(self):
        graph = SceneGraph(
            self.template,
            np.eye(4),
            nib.Nifti1Header(),
            [BoundingBox(self.bb_class_id, 4, "fff", [ObjectAttribute(1, "sf")], [[0, 0], [0, 0]])],
            [Relation(1, 4, 3)],
            np.array([[3]])
        )
        # Just test that no exception is raised
        self.assertTrue(isinstance(repr(graph), str))

    def test_add_coco_annotations(self):
        import pycocotools3d.mask as mask_utils
        tmp = NaturalImageKG([ObjectClass(1, "Obj1"), ObjectClass(2, "Bb2", has_mask=True)], [RelationRule(1, "rule")])
        graph = SceneGraph(
            tmp,
            np.eye(4),
            nib.Nifti1Header(),
            # Check that the graph is remapped as contiguous
            [BoundingBox(1, 2, "fff", None, [[1, 2], [3, 4]]),
             BoundingBox(1, 3, "fff", None, [[5, 6], [7, 8]])],
            [Relation(1, 2, 3)],
            np.array([[0, 3]])
        )

        coco = tmp.to_coco()
        image_id = 4
        graph.add_coco_annotations(coco, image_id)

        # Check bounding boxes
        annotations = coco["annotations"]
        self.assertEqual(2, len(annotations))
        annot1 = {
            "id": 1,
            "image_id": image_id,
            "category_id": 1,
            "bbox": [1, 2, 2, 2],
            "area": 4.,
            "iscrowd": 0
            # No mask, so no segmentation
        }
        self.assertEqual(annot1, annotations[0])
        annot2 = {
            "id": 2,
            "image_id": image_id,
            "category_id": 1,
            "bbox": [5, 6, 2, 2],
            "area": 4.,
            "iscrowd": 0,
            "segmentation": mask_utils.encode(np.array([[0, 1]]))
        }
        self.assertEqual(annot2, annotations[1])

        # Check relation
        relations = coco["relations"]
        self.assertEqual(1, len(relations))
        self.assertEqual((image_id, 1, 2, 1), relations[0])
