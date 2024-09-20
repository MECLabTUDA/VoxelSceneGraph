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
import json
from collections import defaultdict
from os import PathLike
from typing import TypeVar

import nibabel as nib
import numpy as np
import torch

try:
    from pycocotools3d.coco.abstractions.common import Image
    from pycocotools3d.coco.abstractions.relation_detection import SSGDataset
except ImportError:
    Image = dict
    SSGDataset = dict

from scene_graph_api.scene import BoundingBox, Relation
from scene_graph_api.utils.nifti_io import NiftiImageWrapper
from .box_list_field_extractor import FieldExtractor
from .box_list_fields import PredictionField
from ..scene import SceneGraph
from ..knowledge import KnowledgeGraph

BoxListBase = TypeVar("BoxListBase", bound="BoxList")


class BoxListConverter:
    """Helper class used to convert BoxList to/from other formats."""

    def __init__(self, boxlist_type: type[BoxListBase]):
        self.boxlist_type = boxlist_type

    # ------------------------------------------------------------------------------------------------------------------
    # Format conversion: FROM
    # ------------------------------------------------------------------------------------------------------------------
    def from_scene_graph(self, graph: SceneGraph) -> BoxListBase:
        """
        Convert a SceneGraph to a BoxList. The graph is automatically remapped with contiguous ids.
        The BoxList will have LABELS, LABELMAP, and RELATIONS fields.
        WARNING: this method should only be used with SceneGraphs that have been remapped to contiguous ids.
        Note: all attributes are added as block_diag tensors, i.e. only one tensor for all object classes/instances.
        Note: currently no support for keypoints.
        """
        dhw_shape = graph.object_labelmap.shape

        box_coordinates_xyxy = []
        obj_classes = []
        for obj in sorted(graph.iter_bounding_boxes(), key=lambda bb: bb.id):
            zyx_start, zyx_end = obj.bounding_box
            box_coordinates_xyxy.append(list(zyx_start) + list(zyx_end))
            obj_classes.append(obj.class_id)

        # Cast box to tensor
        box = torch.tensor(box_coordinates_xyxy, dtype=torch.float32).reshape(-1, 2 * len(dhw_shape))
        target = self.boxlist_type(box, tuple(dhw_shape), self.boxlist_type.Mode.zyxzyx)

        # Add affine matrix
        target.AFFINE_MATRIX = torch.tensor(graph.image_affine)

        # Add labels
        target.LABELS = torch.tensor(obj_classes, dtype=torch.long)

        # Add attributes
        # Note: to construct the tensor, it's easier to use torch.block_diag after computing arrays per object class
        #       However, this can also change the ordering of objects, so we also keep track of the original ordering
        attr_per_obj_class = defaultdict(list)
        mapping = {}
        knowledge = graph.knowledge_graph
        for block_diag_idx, obj in enumerate(graph.iter_bounding_boxes()):
            mapping[obj.id - 1] = block_diag_idx
            obj_class = knowledge.get_object_class_by_id(obj.class_id)
            attr_values = [
                obj_class.get_attribute_by_id(attr.id).value_instance_to_long(attr.value)
                for attr in sorted(obj.attributes, key=lambda attr: attr.id)
            ]
            attr_per_obj_class[obj.class_id].append(attr_values)
        block_diag_attrs = torch.block_diag(
            *[torch.tensor(attrs, dtype=torch.long) for attrs in attr_per_obj_class.values()]
        )
        if block_diag_attrs.nelement() > 0:
            inverse_mapping = torch.tensor([mapping[idx] for idx in range(len(mapping))], dtype=torch.long)
            target.ATTRIBUTES = block_diag_attrs[inverse_mapping]

        # Add image-level attribute
        # Note: incompatible attributes are just replaced by a default value
        img_attr_values = [
            knowledge.image.get_attribute_by_id(attr.id).value_instance_to_long(attr.value)
            for attr in sorted(graph.image.attributes, key=lambda attr: attr.id)
        ]
        if img_attr_values:
            target.IMAGE_ATTRIBUTES = torch.tensor(img_attr_values, dtype=torch.long)

        # Add relation to target
        num_box = len(target)
        relation_map = torch.zeros((num_box, num_box), dtype=torch.uint8)
        for rel_class_id, rels in graph.relations_by_rule_id.items():
            for rel in rels:
                if relation_map[rel.subject_id - 1, rel.object_id - 1] > 0:
                    raise RuntimeError(
                        f"Subject {rel.subject_id} and object {rel.object_id} pair cannot have multiple relations."
                    )
                relation_map[rel.subject_id - 1, rel.object_id - 1] = rel_class_id  # Object ids start at 1
        target.RELATIONS = relation_map

        # Add the labelmap
        target.LABELMAP = torch.from_numpy(graph.object_labelmap).to(torch.uint8)

        return target

    # ------------------------------------------------------------------------------------------------------------------
    # Format conversion: TO
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def to_scene_graph(boxlist: BoxListBase, graph: KnowledgeGraph) -> SceneGraph:
        """Convert the BoxList to a SceneGraph given a SceneGraphTemplate."""
        # FIXME we currently have no way to save prediction scores to a SceneGraph
        #  If we had that, we could also convert the Scene Graph to a BoxList with PredictionFields

        # Create BoundingBox objects
        n_dim = boxlist.n_dim
        bounding_boxes = []
        labels = FieldExtractor.labels(boxlist)
        for idx, (box, label) in enumerate(zip(boxlist.boxes, labels), start=1):
            bounding_boxes.append(
                BoundingBox(
                    label.item(),
                    idx,
                    BoundingBox.default_name(graph, label, idx),
                    [],  # TODO recover attributes once we support them
                    [box[:n_dim].tolist(), box[n_dim:].tolist()],
                )
            )

        # Create Relation objects if any
        relations = []
        try:
            relation_matrix = FieldExtractor.relation_matrix(boxlist).cpu()
            for subject_id, object_id in torch.nonzero(relation_matrix):
                relations.append(
                    Relation(
                        relation_matrix[subject_id, object_id].item(),
                        subject_id.item() + 1,
                        object_id.item() + 1
                    )
                )
        except ValueError:
            # No relations in this BoxList
            pass

        if boxlist.has_field(boxlist.AnnotationField.AFFINE_MATRIX):
            affine = boxlist.AFFINE_MATRIX.cpu().numpy()
        else:
            affine = np.eye(4)

        labelmap = FieldExtractor.labelmap(boxlist).detach().cpu().numpy()
        tmp_image = nib.Nifti1Image(labelmap, affine)  # Recreate Nifti1Image to automatically create a proper header
        # noinspection PyTypeChecker
        return SceneGraph(
            graph,
            tmp_image.affine,
            tmp_image.header,
            bounding_boxes,
            relations,
            labelmap
        )

    @staticmethod
    def to_labelmap(boxlist: BoxListBase) -> tuple[NiftiImageWrapper, dict[int, int]]:
        """
        Convert the BoxList's labelmap to a Nifti image.
        Note: can be used with both groundtruth annotations and predictions.
        Note: if this BoxList has no field AFFINE_MATRIX, a default np.eye(4) will be used.
        """
        # Prepare the affine matrix
        if boxlist.has_field(boxlist.AnnotationField.AFFINE_MATRIX):
            affine = boxlist.AFFINE_MATRIX.cpu().numpy()
        else:
            affine = np.eye(4)

        labelmap: torch.LongTensor = FieldExtractor.labelmap(boxlist).cpu().numpy()
        # noinspection PyUnresolvedReferences
        ids_to_class = {idx: lbl.item() for idx, lbl in enumerate(FieldExtractor.labels(boxlist), start=1)}

        return NiftiImageWrapper(nib.Nifti1Image(labelmap, affine), True), ids_to_class

    @staticmethod
    def to_semantic_segmentation(boxlist: BoxListBase) -> NiftiImageWrapper:
        """
        Convert the BoxList's semantic segmentation to a Nifti image.
        Note: if this BoxList has no field AFFINE_MATRIX, a default np.eye(4) will be used.
        Note: can be used with both groundtruth annotations and predictions.
        """
        # Prepare the affine matrix
        if boxlist.has_field(boxlist.AnnotationField.AFFINE_MATRIX):
            affine = boxlist.AFFINE_MATRIX.cpu().numpy()
        else:
            affine = np.eye(4)

        segmentation = FieldExtractor.segmentation(boxlist).cpu().numpy()
        return NiftiImageWrapper(nib.Nifti1Image(segmentation, affine), True)

    @staticmethod
    def save_boxes_as_3d_slicer_markups(
            boxlist: BoxListBase,
            path: PathLike | str,
            graph: KnowledgeGraph
    ):
        """
        Save boxes as a 3D slicer markups JSON file. The knowledge graph is used to color-code the boxes.
        Note: can be used with both groundtruth annotations and predictions.
        Note: if this BoxList has no field AFFINE_MATRIX, a default np.eye(4) will be used.
        """
        from PIL import ImageColor

        # Load fields
        aff_matrix = boxlist.get_field(boxlist.AnnotationField.AFFINE_MATRIX)
        if aff_matrix is None:
            aff_matrix = torch.eye(4).to(boxlist.boxes)
        rot = aff_matrix[:3, :3]
        offset = aff_matrix[:3, -1]

        if boxlist.has_field(PredictionField.PRED_SCORES):
            scores = boxlist.PRED_SCORES.detach().cpu()
        else:
            scores = torch.ones(len(boxlist))
        labels = FieldExtractor.labels(boxlist)

        markups = []
        boxlist = boxlist.convert(boxlist.mode.zyxdhw)
        orientation = rot.cpu().view(-1).tolist()  # The orientation is very important for the ROI to be image-aligned
        for idx, (box, score, label) in enumerate(zip(boxlist.boxes, scores, labels)):
            roi_name = f"BBox {idx:2d} Label {label:2d} Score: {score.item():.3f}"
            color = [c / 255 for c in ImageColor.getrgb(graph.get_object_class_by_id(label.item()).color)]

            arr_center = box[:boxlist.n_dim] + .5 * (box[boxlist.n_dim:] - 1)
            arr_size = box[boxlist.n_dim:]

            img_center = torch.matmul(rot, arr_center) + offset
            img_size = torch.abs(torch.matmul(rot, arr_size))  # Abs call to cancel the direction

            # Note: There is no option to disable the fill opacity from the markups...
            markups.append({
                "name": roi_name,
                "type": "ROI",
                "coordinateSystem": "LPS",
                "coordinateUnits": "mm",
                "roiType": "Box",
                "center": img_center.tolist(),
                "orientation": orientation,
                "size": img_size.tolist(),
                "controlPoints": [
                    {
                        "id": "1",
                        "label": "1",
                        "position": img_center.tolist(),
                        "orientation": orientation,
                        "visibility": True
                    }
                ],
                "display": {
                    "visibility": True,
                    "selectedColor": color,
                    "propertiesLabelVisibility": True,
                    "pointLabelsVisibility": False,
                    "handlesInteractive": False
                }
            })

        with open(path, "w") as f:
            json.dump({
                "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/"
                           "Schema/markups-schema-v1.0.3.json#",
                "markups": markups
            }, f)

    @staticmethod
    def add_to_coco_annotation(
            boxlist: BoxListBase,
            dataset: SSGDataset,
            img_id: int,
            use_cats: bool = True
    ):
        """
        :param boxlist: BoxList to add to the dataset dictionary.
        :param dataset: SSGDataset where to add the labels.
        :param img_id: the id of the image corresponding to this annotation.
        :param use_cats:  if False, all labels are 0 or 1 and segmentations are ignored.
        Adds an ImageDict representing this target to a DatasetDict.
        Note: only implemented for 3D.
        Note: Only adds segmentation masks if the BoxList has the field AnnotationField.MASKS,
              and it's not binary classification.
        """
        from pycocotools3d import mask3d as mask_utils3d, mask as mask_utils

        # TODO add test
        assert boxlist.n_dim == 3, f"Only implemented for 3D, got {boxlist.n_dim}D."

        target = boxlist.convert(boxlist.Mode.zyxdhw)
        # No inspect because NotRequired does not seem supported on import
        # noinspection PyTypeChecker
        image_dict: Image = {
            "depth": target.size[0],
            "height": target.size[1],
            "width": target.size[2],
            "id": img_id
        }
        dataset["images"].append(image_dict)
        graph_objs = []
        mask_utils_lib = mask_utils if boxlist.n_dim == 2 else mask_utils3d
        for obj_idx in range(len(target)):
            z, y, x, d, h, w = target.boxes[obj_idx].tolist()  # tolist() is important to avoid having 1 element tensors
            cat_id = target.LABELS[obj_idx].item() if use_cats else 1
            annot = {
                "area": w * h * d,
                "iscrowd": 0,
                "image_id": img_id,
                "bbox": [z, y, x, d, h, w],
                "category_id": cat_id,
                # It is important for the valuation code (pycocotools 2d and 3d) that no id is 0
                "id": img_id * 2048 + obj_idx + 1
            }

            # Only add the mask if it's available
            if use_cats:
                try:
                    zyx_seg = FieldExtractor.masks(target)[obj_idx].get_mask_tensor().numpy().astype(np.uint8)
                    # noinspection PyTypeChecker
                    annot["segmentation"] = mask_utils_lib.encode(np.asfortranarray(zyx_seg))
                except ValueError:
                    # No segmentation masks to encode
                    pass
            graph_objs.append(annot)

        if use_cats:
            # Add relations
            relations = []
            relation_matrix = boxlist.get_field(boxlist.AnnotationField.RELATIONS)
            if relation_matrix is not None:
                for subj, obj in relation_matrix.nonzero():
                    relations.append([img_id, subj.item(), obj.item(), relation_matrix[subj, obj].item()])
            dataset["relations"] += relations

        dataset["annotations"] += graph_objs
