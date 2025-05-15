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

import cc3d
import numpy as np

from scene_graph_annotation.utils.array_utils import bbox_2d, bbox_3d
from scene_graph_api.scene import BoundingBox
from .Object import CompositeBoundingBox, CompositeBoundingBoxWithSegmentation
from .Relation import Relation
from .SceneGraph import SceneGraph


def split(
        comp_bb: CompositeBoundingBox,
        scene_graph: SceneGraph,
) -> tuple[BoundingBox, BoundingBox]:
    """
    Note: legacy method from the time, where not all objects had masks...

    Splits the composite bb back into 2 bbs and reverts the labelmap in place.
    Replaces the composite bb in the object list with the two original bbs.
    Relations referencing the composite bb will be duplicated so that there is a relation with each component.
    """
    # With bounding boxes, we have to compute everything again (in case there is some overlapping boxes)

    # Update object hitboxes: remove composite...
    # mask = scene_graph.object_hitboxes == comp_bb.id
    # scene_graph.object_hitboxes[mask] = 0
    # The background color is irrelevant since it's transparent...
    # scene_graph.object_overlay[mask] = 0
    # scene_graph.object_hitboxes[mask] = 0

    # scene_graph.object_hitboxes[comp_bb.comp1_mask] = comp_bb.comp1.id
    # scene_graph.object_overlay[comp_bb.comp1_mask] = color
    # scene_graph.object_hitboxes[comp_bb.comp2_mask] = comp_bb.comp2.id
    # scene_graph.object_overlay[comp_bb.comp2_mask] = color

    # Update bb objects
    scene_graph.remove_bounding_box(comp_bb)
    scene_graph.add_bounding_box(comp_bb.comp1)
    scene_graph.add_bounding_box(comp_bb.comp2)
    scene_graph.compute_objects_overlay()

    _after_split_relations_cleanup(comp_bb.id, [comp_bb.comp1.id, comp_bb.comp2.id], scene_graph)

    return comp_bb.comp1, comp_bb.comp2


def split_with_mask(
        comp_seg: CompositeBoundingBoxWithSegmentation,
        scene_graph: SceneGraph
) -> tuple[BoundingBox, BoundingBox]:
    """
    Splits the composite seg back into 2 seg and reverts the labelmap in place.
    Replaces the composite segmentation in the object list with the two original segmentations.
    Relations referencing the composite seg will be duplicated so that there is a relation with each component.
    """
    # Update labelmap
    current_labelmap = scene_graph.object_labelmap
    current_labelmap[comp_seg.comp1_mask] = comp_seg.comp1.id
    current_labelmap[comp_seg.comp2_mask] = comp_seg.comp2.id
    # Update object hitboxes
    # If there's overlap with a bounding box, the object with a mask is on top anyway
    scene_graph.object_hitboxes[comp_seg.comp1_mask] = comp_seg.comp1.id
    scene_graph.object_hitboxes[comp_seg.comp2_mask] = comp_seg.comp2.id

    # Add the components back
    scene_graph.remove_bounding_box(comp_seg)
    scene_graph.add_bounding_box(comp_seg.comp1)
    scene_graph.add_bounding_box(comp_seg.comp2)

    # Note: no need to update object hitboxes, because both components did not change shape or color

    _after_split_relations_cleanup(comp_seg.id, [comp_seg.comp1.id, comp_seg.comp2.id], scene_graph)

    return comp_seg.comp1, comp_seg.comp2


def split_into_connected_components(
        box: BoundingBox,
        scene_graph: SceneGraph,
        pre_computed_components: np.ndarray | None = None,
        n_components: int | None = None
) -> list[BoundingBox]:
    """
    Split a BoundingBox with mask into its connected components.
    Note: the object needs to have a mask.
    Note: the knowledge graph must specify that this object class is not unique.
    Note: if only 1 component is detected nothing happens and the input object is the only box returned.
    :returns: a list of all newly generated boxes.
    """
    knowledge_graph = scene_graph.knowledge_graph
    assert not knowledge_graph.get_object_class_by_id(box.class_id).is_unique

    if pre_computed_components is None or n_components is None:
        pre_computed_components, n_components = cc3d.connected_components(
            scene_graph.object_labelmap == box.id,
            return_N=True
        )

    if n_components == 1:
        return [box]

    n_dim = len(pre_computed_components.shape)

    new_boxes = []
    for idx in range(n_components):
        obj_mask = pre_computed_components == idx + 1
        # noinspection PyTypeChecker
        bounds = bbox_2d(obj_mask) if n_dim == 2 else bbox_3d(obj_mask)
        new_box_id = scene_graph.get_next_available_bounding_box_id()

        # Create new BoundingBox object
        new_box = BoundingBox(
            box.class_id,
            new_box_id,
            BoundingBox.default_name(knowledge_graph, box.class_id, new_box_id),
            bounds,
            common_attributes=box.common_attributes,
            common_keypoints=box.common_keypoints,
            attributes=box.attributes,
            keypoints=box.keypoints,
        )

        scene_graph.add_bounding_box(new_box)
        new_boxes.append(new_box)

        # Update labelmap and hitbox map; the overlay does not need any update
        scene_graph.object_labelmap[obj_mask] = new_box_id
        scene_graph.object_hitboxes[obj_mask] = new_box_id

    # Update relations
    _after_split_relations_cleanup(box.id, [new_box.id for new_box in new_boxes], scene_graph)

    # Finally remove input object, labelmap and hitbox map have already been taken care of
    scene_graph.remove_bounding_box(box)

    return new_boxes


def _after_split_relations_cleanup(split_obj_id: int, new_obj_ids: list[int], scene_graph: SceneGraph):
    """Clean up relations after split by replacing ids and removing duplicates."""
    # Update relations and duplicate
    for rel_list in scene_graph.relations_by_rule_id.values():
        for relation in rel_list.copy():
            # Reflexive relation
            if relation.object_id == split_obj_id and relation.subject_id == split_obj_id:
                scene_graph.remove_relation(relation)
                # Add all possible (non-reflexive) relations
                for id1 in new_obj_ids:
                    for id2 in new_obj_ids:
                        if id1 != id2:
                            rel_list.append(Relation(relation.rule_id, id1, id2))
                continue

            # At most one ref to the split object
            if relation.object_id == split_obj_id:
                scene_graph.remove_relation(relation)
                for id1 in new_obj_ids:
                    rel_list.append(Relation(relation.rule_id, relation.subject_id, id1))
            if relation.subject_id == split_obj_id:
                scene_graph.remove_relation(relation)
                for id1 in new_obj_ids:
                    rel_list.append(Relation(relation.rule_id, id1, relation.object_id))
