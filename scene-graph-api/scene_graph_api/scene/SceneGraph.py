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

from __future__ import annotations

import gzip
import json
from collections import defaultdict
from copy import deepcopy
from logging import Logger
from os import PathLike
from pathlib import Path
from typing import Iterator

import cc3d
import nibabel as nib
import numpy as np
from nibabel.wrapstruct import WrapStructError
from typing_extensions import Self

from .Object import BoundingBox, Object
from .ObjectAttribute import ObjectAttribute
from .Relation import Relation
from .SceneGraphComponent import SceneGraphComponent
from ..knowledge import KnowledgeGraph
from ..utils.array_utils import bbox_2d, bbox_3d
from ..utils.indexing import contiguous_mapping
from ..utils.nifti_io import NiftiImageWrapper
from ..utils.parsing import *


class SceneGraph(SceneGraphComponent):
    """
    Contains a list of objects, a list of relations, the nifti image and segmentation.
    Nifti volumes are ALWAYS gzipped when saved.
    Object ids NEED to be unique across bounding boxes AND segmentations.

    Regarding the UI, self._object_hitboxes is the labelmap containing also bounding boxes.
    It is used to check which object is at each position.
    Bounding boxes are first drawn from largest to smallest
    (in an attempt to avoid having completely occluded bbs from occlusion).
    self._object_overlay is the RGB array like self._object_hitboxes, containing the masks for display.

    Note: one can identify which knowledge graph was used to annotate a specific Scene Graph by comparing
          the saved hash for the Graph to the one of the knowledge graph that you are trying to use.
    """
    _bb_key = "bounding_boxes"
    _rel_key = "relations"
    _labelmap_key = "labelmap"
    _image_level_attributes_key = "image"
    _knowledge_graph_hash_key = "hash"

    def __init__(
            self,
            # Template required to build the relations by id dict
            # It's not required for objects as if there are no objects of a class, new ones won't appear
            # (Contrary to new relations)
            graph: KnowledgeGraph,
            image_affine: np.ndarray,
            image_header: nib.Nifti1Header | None,
            bounding_box_objects: list[BoundingBox],
            relations: list[Relation],
            object_labelmap: np.ndarray,
            image_level_attributes: list[ObjectAttribute] | None = None
    ):
        """Note: the labelmap/affine are expected to be depth-first."""
        self.knowledge_graph = graph
        self.image_affine = image_affine
        self.image_header = image_header
        self.image = Object(0, 0, "Background", attributes=image_level_attributes)

        # BBs and segmentations in the format {obj_clss_id: [objs]}
        # Also useful for grouping object by class id
        # We also need lists[obj] (instead of dict[obj.id: obj]) as input for validation on id unicity
        self._bb_by_class_id: dict[int, list[BoundingBox]] = {}
        for bounding_box in bounding_box_objects:
            if bounding_box.class_id not in self._bb_by_class_id:
                self._bb_by_class_id[bounding_box.class_id] = []
            self._bb_by_class_id[bounding_box.class_id].append(bounding_box)

        # Internal shorthand for fast object search
        # Duplicates data storage but eh
        self._bb_by_id: dict[int, BoundingBox] = {obj.id: obj for obj in bounding_box_objects}

        # List relations in the format {rule_id: [objs]}
        self.relations_by_rule_id: dict[int, list[Relation]] = {rule.id: [] for rule in graph.rules}
        for rel in relations:
            if rel.rule_id in self.relations_by_rule_id:  # Failsafe in case there is an unknown id
                self.relations_by_rule_id[rel.rule_id].append(rel)
        # Cast to uint8 since HAVE to display it at some point and labels are usually integers
        # Note: the call to round() is required as sometimes precision is lost e.g. 4.99999999 instead of 5.0
        self.object_labelmap = object_labelmap.round().astype(np.uint8)

    # noinspection PyMethodOverriding
    @classmethod
    def from_json(cls, knowledge_graph: KnowledgeGraph, json_dict: dict, logger: Logger | None = None) -> Self:
        # Note: we create a dummy empty segmentation if we cannot decode the image
        objects = [BoundingBox.from_json(obj_dict) for obj_dict in json_dict[cls._bb_key]]
        relations = [Relation.from_json(obj_dict) for obj_dict in json_dict[cls._rel_key]]
        labelmap_str = json_dict[cls._labelmap_key]

        try:
            # Convert string back to nifti image
            nifti_img = NiftiImageWrapper.from_str(labelmap_str)
        except gzip.BadGzipFile:
            if logger is not None:
                logger.error(
                    f"Failed to load the labelmap from its string representation. "
                    f"The string content is not even gzipped."
                )
            nifti_img = NiftiImageWrapper.empty()
        except WrapStructError:
            if logger is not None:
                logger.error(
                    f"Failed to load the labelmap from its string representation. "
                    f"Is the gzipped string content corrupted?"
                )
            nifti_img = NiftiImageWrapper.empty()

        image_level_attributes = [
            ObjectAttribute.from_json(obj_dict) for obj_dict in json_dict.get(cls._image_level_attributes_key, [])
        ]

        return cls(
            knowledge_graph,
            nifti_img.affine,
            nifti_img.header,
            objects,
            relations,
            np.asarray(nifti_img.get_fdata()).round().astype(np.uint8),  # In case there is some compression loss
            image_level_attributes
        )

    def to_json(self) -> dict:
        object_labelmap = NiftiImageWrapper(
            nib.Nifti1Image(self.object_labelmap, self.image_affine, self.image_header),
            True  # Always assume to be depth-first
        )
        return {
            self._bb_key: [bb.to_json() for obj_list in self._bb_by_class_id.values() for bb in obj_list],
            self._rel_key: [rel.to_json() for rel_list in self.relations_by_rule_id.values() for rel in rel_list],
            self._image_level_attributes_key: [attr.to_json() for attr in self.image.attributes],
            self._knowledge_graph_hash_key: self.knowledge_graph.hash,
            # The labelmap is converted to bytes, gzipped and finally encoded to str
            self._labelmap_key: object_labelmap.to_str(),
        }

    @classmethod
    def schema(cls) -> dict:
        return {
            "id": cls.schema_name(),
            "type": "object",
            "properties": {
                cls._bb_key: {"type": "array", "items": {"$ref": BoundingBox.schema_name()}},
                cls._rel_key: {"type": "array", "items": {"$ref": Relation.schema_name()}},
                cls._labelmap_key: {"type": "string"},
                cls._image_level_attributes_key: {"type": "array", "items": {"$ref": ObjectAttribute.schema_name()}},
                cls._knowledge_graph_hash_key: {"type": "integer"},
            },
            "required": [cls._bb_key, cls._rel_key, cls._labelmap_key],
            "additionalProperties": False
        }

    def validate(self, logger: Logger) -> bool:
        success = True

        # Validate bounding boxes
        n_dim = len(self.object_labelmap.shape)
        for bb_list in self._bb_by_class_id.values():
            for bb in bb_list:
                success &= bb.validate(self.knowledge_graph, logger)
                success &= bb.validate_bounding_box_length_with_mask(n_dim, logger)
        known_bb_ids = [bb.id for bb_list in self._bb_by_class_id.values() for bb in bb_list]
        success &= check_list_unicity(known_bb_ids, logger, "BoundingBox id")

        # Validate the segmentation
        if np.prod(self.object_labelmap.shape) == 0:
            logger.error(f"The labelmap is invalid because it could not be decoded properly.")
            success = False
        else:
            segmentation_object_ids = np.unique(self.object_labelmap)
            ids_seen = [0]

            for seg_list in self._bb_by_class_id.values():
                for seg in seg_list:
                    # Already validated as a bounding box (legacy code)
                    # success &= seg.validate(self.knowledge_graph, logger)

                    # Check that object id is in the segmentation mask
                    if seg.id not in segmentation_object_ids:
                        logger.error(f"Object with id {seg.id} is missing from the labelmap.")
                        success = False
                    ids_seen.append(seg.id)

            # Check that all ids found in the labelmap have a counterpart
            unseen_ids = set(segmentation_object_ids).difference(ids_seen)
            if unseen_ids:
                logger.error(f"Some ids (" + ", ".join(map(str, unseen_ids)) +
                             ") found in the labelmap do not match any object defined in the scene graph.")
                success = False

        # Validate relations
        for rel_list in self.relations_by_rule_id.values():
            for rel in rel_list:
                # Linter is too dumb to cast both lists to the list[Object] primitive
                # noinspection PyTypeChecker
                objects = [bb for bb_list in self._bb_by_class_id.values() for bb in bb_list]
                success &= rel.validate_references(self.knowledge_graph, objects, logger)

        # Validate image level attributes
        success &= self.image.validate_attributes(self.knowledge_graph.image, logger)

        return success

    def save(self, path: str | PathLike) -> bool:
        """
        Save as a json file at given destination path.
        Note: automatically remaps (in place) as contiguous indexing.
        :returns: success.
        """
        try:
            with open(path, "w") as f:
                self.remap_ids_as_contiguous()
                json.dump(self.to_json(), f)
            return True
        except FileNotFoundError:
            return False

    @classmethod
    def load(
            cls,
            path: str | PathLike,
            knowledge_graph: KnowledgeGraph,
            logger: Logger,
            force: bool = False
    ) -> Self | None:
        """
        Load the knowledge graph from the source path and validate the content.
        Only return the scene graph if it matches the knowledge graph.
        """
        obj_dict = load_json_from_path(path, Path(path).as_posix() + ":", logger)
        if obj_dict is not None:
            validator = get_validator(cls.schema(), registry=SceneGraphComponent.SCHEMA_REGISTRY)
            errors = list(validator.iter_errors(obj_dict))
            if errors:
                for error in errors:
                    logger.error(error.message)
                if not force:
                    return

            sg = cls.from_json(knowledge_graph, obj_dict, logger)
            if sg.validate(logger) or force:
                return sg

    def get_bounding_box_by_id(self, obj_id: int) -> BoundingBox | None:
        """Returns an object given the id if found"""
        return self._bb_by_id.get(obj_id)

    def iter_bounding_boxes(self) -> Iterator[BoundingBox]:
        """Iterator over all bounding boxes in the graph."""
        yield from self._bb_by_id.values()

    def iter_relations(self) -> Iterator[Relation]:
        """Iterator over all relations in the graph."""
        for rels in self.relations_by_rule_id.values():
            yield from rels

    @property
    def bounding_boxes_by_class_id(self) -> dict[int, list[BoundingBox]]:
        """Provide a deep-copy of the first-depth dictionary and value lists."""
        return {key: value.copy() for key, value in self._bb_by_class_id.items()}

    def remap_ids_as_contiguous(self) -> Self:
        """
        Remap all ids (in place) such they are contiguous.
        BoundingBox and Relation ids start at 1 (0 is background).
        WARNING: this method assumes that the knowledge graph is contiguously indexed.
        Note: updates the object_labelmap.
        Note: renames objects named with the default name, such that the new name matches the new id.
        """
        # BoundingBox ids
        all_bb_ids = [bb.id for bb in self._bb_by_id.values()]
        bb_mapping = contiguous_mapping(all_bb_ids, start=1)
        if bb_mapping:
            # Only do the processing if there is something to do
            # Remap class ids
            old_labelmap = self.object_labelmap.copy()
            for bb in self._bb_by_id.values():
                old_id = bb.id
                new_id = bb_mapping.get(bb.id, bb.id)
                bb.id = new_id
                # Also update the labelmap if there is a segmentation
                if old_id != new_id:
                    self.object_labelmap[old_labelmap == old_id] = new_id
                # Also rename if the name is the default one
                if bb.name == bb.default_name(self.knowledge_graph, bb.class_id, old_id):
                    bb.name = bb.default_name(self.knowledge_graph, bb.class_id, new_id)

            # Remap bb ids in relations
            for rel_list in self.relations_by_rule_id.values():
                for rel in rel_list:
                    rel.subject_id = bb_mapping.get(rel.subject_id, rel.subject_id)
                    rel.object_id = bb_mapping.get(rel.object_id, rel.object_id)

        # Note: attributes can only be remapped at a knowledge graph level (like object-class ids)

        # Update shorthand (also not need for mapping ids as they already have been updated)
        self._bb_by_id = {bb.id: bb for bb in self._bb_by_id.values()}
        return self

    def __repr__(self):
        bbs_repr = ",".join(map(repr, self._bb_by_id.values()))
        rels_repr = ",".join(map(repr, [rel for rule_id in self.relations_by_rule_id
                                        for rel in self.relations_by_rule_id[rule_id]]))
        image_attr_repr = ",".join(map(repr, self.image.attributes))
        return f"{type(self).__name__}(hash={self.knowledge_graph.hash}, affine={self.image_affine}, " \
               f"bounding_boxes=[{bbs_repr}],relations=[{rels_repr}]," \
               f"label_map_shape={self.object_labelmap.shape}, " \
               f"label_map_values={list(np.unique(self.object_labelmap))}," \
               f"image_level_attributes=({image_attr_repr})"

    def copy(self) -> Self:
        return deepcopy(self)

    # ------------------------------------------------------------------------------------------------------------------
    # Format conversion: FROM
    # ------------------------------------------------------------------------------------------------------------------

    @classmethod
    def create_from_segmentation(
            cls,
            knowledge_graph: KnowledgeGraph,
            segmentation: NiftiImageWrapper,
            logger: Logger
    ) -> Self | None:
        """
        Creates a scene graph from a 2D or 3D segmentation:
        - for each object class present in the segmentation and where is_unique=True:
          - save the mask for this object class and zero out the segmentation
            this is done such that the connected components computation will ignore these classes
        - compute the connected components (excluding the background)
        - for each object class present in the segmentation and where is_unique=True:
          - add the masks to the labelmap with the next available object index
        - for each object:
          - get the original label and check that it is defined in the knowledge graph
          - instantiate all attributes with default values
          - if a bounding box, then compute the bounding box from the mask and remove it from the labelmap
        """
        segmentation = segmentation.as_depth_first()  # Always enforce depth-first. Relevant for loading from JSON

        initial_segmentation = segmentation.get_mask_data()

        n_dim = len(initial_segmentation.shape)
        # Only 2D and 3D currently supported
        if n_dim not in [2, 3]:
            logger.error(f"Bounding boxes are currently only supported for 2D or 3D. Found {n_dim}D segmentation.")
            return None

        classes_present = np.unique(initial_segmentation)
        # First handle all ignored class that need to be removed
        # Note: this is now handled by create_fom_labelmap
        # for obj_class in knowledge_graph.classes:
        #     if obj_class.is_ignored and obj_class.id in classes_present:
        #         initial_segmentation[initial_segmentation == obj_class.id] = 0

        # Then handle all object classes that contain a unique item
        # And zero out these voxels to make the cc computation easier and bypass it
        # (We will add the unique objects afterward to the labelmap)
        unique_objects: dict[int, np.ndarray] = {}
        for obj_class in knowledge_graph.classes:
            if obj_class.is_unique and obj_class.id in classes_present:
                # noinspection PyTypeChecker
                mask: np.ndarray = initial_segmentation == obj_class.id
                unique_objects[obj_class.id] = mask
                initial_segmentation[mask] = 0

        # Computes object instances
        object_labelmap, n_objects = cc3d.connected_components(initial_segmentation, return_N=True)

        # We add the unique objects to the labelmap
        for offset, unique_obj_class_id in enumerate(unique_objects):
            mask = unique_objects[unique_obj_class_id]
            object_labelmap[mask] = n_objects + offset + 1
            initial_segmentation[mask] = unique_obj_class_id
            n_objects += 1

        # Compute id mapping to class ids
        ids_to_class: dict[int, int] = {}
        for obj_id in range(1, n_objects + 1):
            # Get class id and check knowledge graph
            first_occ = tuple(np.argwhere(object_labelmap == obj_id)[0])  # Cast to tuple from array is annoying
            ids_to_class[obj_id] = int(initial_segmentation[first_occ])

        return cls.create_fom_labelmap(
            knowledge_graph,
            NiftiImageWrapper(
                nib.Nifti1Image(object_labelmap, segmentation.affine, segmentation.header),
                segmentation.is_depth_first
            ),
            ids_to_class,
            logger
        )

    @classmethod
    def create_fom_labelmap(
            cls,
            knowledge_graph: KnowledgeGraph,
            labelmap: NiftiImageWrapper,
            ids_to_class: dict[int, int],
            logger: Logger
    ) -> Self | None:
        """
        Creates a scene graph from a 2D or 3D labelmap:
        - if a unique class is present multiple times in the labelmap, it will be merged to the first id occurrence
        - for each object:
          - get the original label and check that it is defined in the knowledge_graph
          - instantiate all attributes with default values
          - if a bounding box, then compute the bounding box from the mask and remove it from the labelmap
        """
        labelmap = labelmap.as_depth_first()  # Always enforce depth-first. Relevant for loading from JSON
        labelmap_array = labelmap.get_mask_data()

        n_dim = len(labelmap.shape)
        failed = False
        # Only 2D and 3D currently supported
        if n_dim not in [2, 3]:
            logger.error(f"Bounding boxes are currently only supported for 2D or 3D. Found {n_dim}D segmentation.")
            return

        # Figure out whether a unique object is present multiple times
        # We also check here that all object classes are declared in the knowledge graph
        # Compute reverse mapping
        class_to_ids = defaultdict(list)
        for obj_id, class_id in ids_to_class.items():
            class_to_ids[class_id].append(obj_id)
        # Iterate over unique classes
        for class_id in class_to_ids:
            obj_class = knowledge_graph.get_object_class_by_id(class_id)
            if obj_class is None:
                logger.error(f"Id mapping: found object class id {class_id}, "
                             f"but this id was not declared in the knowledge graph.")
                failed = True
                continue
            if not obj_class.is_unique or len(class_to_ids[class_id]) <= 1:
                continue
            tgt_id = class_to_ids[class_id][0]
            for to_remove_id in class_to_ids[class_id][1:]:
                labelmap_array[labelmap_array == to_remove_id] = tgt_id
                del ids_to_class[to_remove_id]

        bounding_box_instances = []
        # Exclude background
        for obj_id in ids_to_class:
            # Get class id and check the knowledge graph
            obj_class_id = ids_to_class[obj_id]
            obj_class = knowledge_graph.get_object_class_by_id(obj_class_id)

            # Remove ignored classes
            if knowledge_graph.get_object_class_by_id(obj_class_id).is_ignored:
                labelmap_array[labelmap_array == obj_id] = 0
                continue

            # Pre-instantiate object attributes with default value
            obj_default_attributes = [ObjectAttribute(attr.id, attr.default_value()) for attr in obj_class.attributes]
            # Add the object to the list
            # Compute the bounding box from the coarse segmentation
            obj_mask = labelmap_array == obj_id
            # noinspection PyTypeChecker
            bb = bbox_2d(obj_mask) if n_dim == 2 else bbox_3d(obj_mask)
            obj = BoundingBox(
                int(obj_class_id),  # Cast np.int32 to int to avoid JSON serialization issues
                int(obj_id),  # Cast + ~~make object ids start at 1~~ 0 is for background
                BoundingBox.default_name(knowledge_graph, obj_class_id, obj_id),
                obj_default_attributes,
                bb
            )

            bounding_box_instances.append(obj)
            # Note: We now want to keep as much information as possible, so we keep these masks.
            #       They are even useful when merging bounding boxes, as a rotated annotation will remain more precise.
            # if not obj_class.has_mask:
            #     # Finally, remove the coarse segmentation from the labelmap if it's only a bb computed from the mask
            #     labelmap[obj_mask] = 0

        if not failed:
            # Pre-instantiate image level attributes
            img_default_attributes = [
                ObjectAttribute(attr.id, attr.default_value())
                for attr in knowledge_graph.image.attributes
            ]

            # noinspection PyUnresolvedReferences
            graph = SceneGraph(
                knowledge_graph,
                labelmap.affine,
                labelmap.header,
                bounding_box_instances,
                [],
                labelmap_array,
                img_default_attributes
            )
            if graph.validate(logger):
                return graph

    # ------------------------------------------------------------------------------------------------------------------
    # Format conversion: TO
    # ------------------------------------------------------------------------------------------------------------------
    def to_labelmap(self) -> tuple[NiftiImageWrapper, dict[int, int]]:
        """Convert the SceneGraph to a labelmap. Be sure to call SceneGraph.remap_ids_as_contiguous first if needed."""
        labelmap = NiftiImageWrapper(nib.Nifti1Image(self.object_labelmap, self.image_affine, self.image_header), True)
        ids_to_class = {obj.id: obj.class_id for obj in self.iter_bounding_boxes()}
        return labelmap, ids_to_class

    def to_semantic_segmentation(self) -> NiftiImageWrapper:
        """Convert the SceneGraph to a semantic segmentation."""
        segmentation = np.zeros_like(self.object_labelmap)
        for obj in self.iter_bounding_boxes():
            segmentation[self.object_labelmap == obj.id] = obj.class_id

        return NiftiImageWrapper(nib.Nifti1Image(segmentation, self.image_affine, self.image_header), True)

    def add_coco_annotations(self, dataset_annotation: dict, image_id: int):
        """
        Add the annotations contained in this graph in a pycocotools3d.coco.abstractions.DatasetDict.
        Note: uses contiguous ids.
        Note: currently no support for attribute annotations.
        :param image_id: the image id corresponding to this annotation
        :param dataset_annotation: where to add the annotation
        """
        assert "annotations" in dataset_annotation
        assert "relations" in dataset_annotation

        n_dim = len(self.object_labelmap.shape)
        assert n_dim in [2, 3]

        from pycocotools3d.coco.abstractions.relation_detection import SSGAnnotation, SSGRelationAnnotation
        from pycocotools3d import mask as mask_utils, mask3d as mask_utils3d
        encode = mask_utils.encode if n_dim == 2 else mask_utils3d.encode

        # TODO ids as contiguous
        # TODO support attributes
        # TODO support image-level annotation
        contiguous_graph = self.copy().remap_ids_as_contiguous()

        # Format bounding boxes + segmentation
        annotations_: list[SSGAnnotation] = []
        for obj in contiguous_graph.iter_bounding_boxes():
            annotations_.append({
                "id": obj.id,
                "image_id": image_id,
                "category_id": obj.class_id,
                "bbox": list(obj.bounding_box[0] + obj.size()),
                "area": float(np.prod(obj.size())),
                "iscrowd": 0,
            })

            # Only encode the object mask if there is any
            if obj.id in contiguous_graph.object_labelmap:
                annotations_[-1]["segmentation"] = encode(contiguous_graph.object_labelmap == obj.id)

        # Format relations
        relations: list[SSGRelationAnnotation] = [(image_id, rel.subject_id, rel.object_id, rel.rule_id)
                                                  for rel in contiguous_graph.iter_relations()]

        dataset_annotation["annotations"] += annotations_
        dataset_annotation["relations"] += relations
