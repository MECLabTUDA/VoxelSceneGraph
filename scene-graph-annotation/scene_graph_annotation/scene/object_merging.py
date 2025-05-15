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

from scene_graph_api.scene import BoundingBox

from .Object import CompositeBoundingBox, CompositeBoundingBoxWithSegmentation
from .SceneGraph import SceneGraph


def merge(
        scene_graph: SceneGraph,
        comp1: BoundingBox,
        comp2: BoundingBox
) -> CompositeBoundingBox:
    """
    Note: legacy method from the time, where not all objects had masks...

    Creates the composite segmentation and returns it.
    Updates the labelmap in place.
    Replaces the two segmentations in the object list with the composite segmentation.
    Replaces occurrences of the segmentations in relations with the id of the composite segmentation.
    Duplicate relations will be removed.
    WARNING: the two segmentations cannot be the same object
    """
    assert comp1 != comp2
    assert comp1.class_id == comp2.class_id

    # Get next available id
    comp_bb_id = scene_graph.get_next_available_bounding_box_id()

    # With bounding boxes, we have to compute everything again (in case there is some overlapping boxes)
    # comp1_mask = scene_graph.object_hitboxes == comp1.id
    # comp2_mask = scene_graph.object_hitboxes == comp2.id
    new_name = CompositeBoundingBox.default_name(scene_graph.knowledge_graph, comp1.class_id, comp_bb_id)
    comp_bb = CompositeBoundingBox(comp_bb_id, new_name, comp1, comp2, None, None)

    # Draw new bb where only on the background: we first compute the mask
    # Note: bbs don't appear in the labelmap
    # Note: this code will make the new bounding box appear on top of any overlapping bounding box
    # labelmap = scene_graph.object_labelmap
    # slicer = [slice(coord_top_left, coord_bottom_right + 1)
    #           for coord_top_left, coord_bottom_right in zip(*comp_bb.bounding_box)]
    # mask = np.zeros_like(labelmap)
    # mask[tuple(slicer)] = 1
    # mask[labelmap != 0] = 0
    # mask = mask == 1
    # Do the actual hitbox update
    # scene_graph.object_hitboxes[mask] = comp_bb_id
    # In contrast to segmentations, we actually also have to update the overlay
    # hex_color = scene_graph.knowledge_graph.get_object_class_by_id(comp_bb.class_id).color
    # scene_graph.object_overlay[mask] = ImageColor.getcolor(hex_color, "RGB")

    # Update bb objects
    scene_graph.remove_bounding_box(comp1)
    scene_graph.remove_bounding_box(comp2)
    scene_graph.add_bounding_box(comp_bb)
    scene_graph.compute_objects_overlay()

    _after_merge_relation_cleanup(comp_bb, scene_graph)

    return comp_bb


def merge_with_mask(
        scene_graph: SceneGraph,
        comp1: BoundingBox,
        comp2: BoundingBox
) -> "CompositeBoundingBoxWithSegmentation":
    """
    Creates the composite segmentation and returns it.
    Updates the labelmap in place.
    Replaces the two segmentations in the object list with the composite segmentation.
    Replaces occurrences of the segmentations in relations with the id of the composite segmentation.
    Duplicate relations will be removed.
    WARNING: the two segmentations cannot be the same object
    """
    assert comp1 != comp2
    assert comp1.class_id == comp2.class_id

    # Get next available id
    comp_seg_id = scene_graph.get_next_available_bounding_box_id()

    comp1_mask = scene_graph.object_labelmap == comp1.id
    comp2_mask = scene_graph.object_labelmap == comp2.id
    new_name = CompositeBoundingBox.default_name(scene_graph.knowledge_graph, comp1.class_id, comp_seg_id)
    comp_seg = CompositeBoundingBoxWithSegmentation(comp_seg_id, new_name, comp1, comp2, comp1_mask, comp2_mask)

    # Update labelmap
    labelmap_before_merge = scene_graph.object_labelmap
    labelmap_before_merge[comp_seg.comp1_mask] = comp_seg_id
    labelmap_before_merge[comp_seg.comp2_mask] = comp_seg_id
    # Update object hitboxes
    # If there's overlap with a bounding box, the object with a mask is on top anyway
    scene_graph.object_hitboxes[comp_seg.comp1_mask] = comp_seg_id
    scene_graph.object_hitboxes[comp_seg.comp2_mask] = comp_seg_id

    # Update segmentation objects
    scene_graph.remove_bounding_box(comp1)
    scene_graph.remove_bounding_box(comp2)
    scene_graph.add_bounding_box(comp_seg)

    _after_merge_relation_cleanup(comp_seg, scene_graph)

    return comp_seg


def _after_merge_relation_cleanup(comp_bb: CompositeBoundingBox, scene_graph: SceneGraph):
    """Clean up relations after merge by replacing ids and removing duplicates."""
    # Update relations
    for rel_list in scene_graph.relations_by_rule_id.values():
        seen_pairs = set()
        for rel in rel_list.copy():
            if rel.object_id == comp_bb.comp1.id or rel.object_id == comp_bb.comp2.id:
                rel.object_id = comp_bb.id
            if rel.subject_id == comp_bb.comp1.id or rel.subject_id == comp_bb.comp2.id:
                rel.subject_id = comp_bb.id
            pair = rel.object_id, rel.subject_id
            # Duplicate relation removal
            if pair in seen_pairs:
                rel_list.remove(rel)
            else:
                seen_pairs.add(pair)
