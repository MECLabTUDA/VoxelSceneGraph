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

from ..knowledge import KnowledgeGraph
from ..scene import Attribute
from ..scene import SceneGraph

try:
    from networkx import MultiDiGraph
except ImportError:
    MultiDiGraph = None


def knowledge_graph_to_networkx(knowledge_graph: KnowledgeGraph) -> MultiDiGraph:
    """
    Convert a SceneGraphTemplate to a networkx directed multi-graph i.e. to it's corresponding knowledge graph.
    About node ids:
    - Image node has id 0
    - Image-level attributes have id attr.id
    - Object class node have id obj_class.id * infinity
    - Attributes for an object class have id obj_class.id * infinity + attr.id
    """

    infinity = 1000  # Number larger than number of attributes

    graph = MultiDiGraph()

    # Note: color and label data attributes are used by pyvis to customize the graph

    # Add image and image_level attributes if any
    if knowledge_graph.image.attributes:
        graph.add_node(0, label="Image")
        for attr in knowledge_graph.image.attributes:
            graph.add_node(
                attr.id,
                label=f"{attr.name}: {attr.get_attribute_type()}",
            )
            graph.add_edge(0, attr.id, label="attribute")

    # Add object classes and their attributes
    object_class_node_id_mapping = {}
    for obj_class in knowledge_graph.classes:
        obj_class_node_id = obj_class.id * infinity
        object_class_node_id_mapping[obj_class].id = obj_class_node_id
        graph.add_node(obj_class_node_id, label=obj_class.name, color=obj_class.color)
        # Add object attribute as a node with a relation named "attribute"
        for attr in obj_class.attributes:
            attr_node_id = obj_class_node_id + attr.id

            graph.add_node(
                attr_node_id,
                label=f"{attr.name}: {attr.get_attribute_type()}",
                color=obj_class.color
            )
            graph.add_edge(obj_class_node_id, attr_node_id, label="attribute")

    # Add relations to the graph (edges between object classes)
    all_object_classes = [obj_class.id for obj_class in knowledge_graph.classes]
    for rule in knowledge_graph.rules:
        # We need to consider all possible combinations
        for subj_id in rule.subject_filter.get_authorized_classes(all_object_classes):
            for obj_id in rule.object_filter.get_authorized_classes(all_object_classes):
                graph.add_edge(
                    object_class_node_id_mapping[subj_id],
                    object_class_node_id_mapping[obj_id],
                    label=rule.name
                )

    return graph


def scene_graph_to_networkx(scene_graph: SceneGraph) -> MultiDiGraph:
    """
    Convert a SceneGraph to a networkx directed multi-graph.
    About node ids:
        - Image node has id 0
        - Image-level attributes have id attr.id
        - Object node have id obj.id * infinity
        - Attributes for an object class have id obj.id * infinity + attr.id
    """
    infinity = 1000  # Number larger than number of attributes

    knowledge_graph = scene_graph.knowledge_graph

    graph = MultiDiGraph()

    # Note: color and label data attributes are used by pyvis to customize the graph

    # Add image and image_level attributes if any
    if knowledge_graph.image.attributes:
        graph.add_node(0, label="Image")
        for attr in scene_graph.image.attributes:
            # Expects the scene graph to be valid or else get_attribute_by_id will return None
            attr_name = knowledge_graph.image.get_attribute_by_id(attr.id).name

            graph.add_node(
                attr.id,
                label=f"{attr_name}: {attr.value}",
            )
            graph.add_edge(0, attr.id, label="attribute")

    # Add objects and their attributes
    object_node_id_mapping = {}
    for obj in scene_graph.iter_bounding_boxes():
        obj_node_id = obj.id * infinity
        object_node_id_mapping[obj.id] = obj_node_id
        obj_color = knowledge_graph.get_object_class_by_id(obj.class_id).color
        graph.add_node(obj_node_id, label=obj.name, color=obj_color)
        # Add object attribute as a node with a relation named "attribute"
        for attr in obj.attributes:  # type: int, Attribute
            attr_node_id = obj_node_id + attr.id

            # Expects the scene graph to be valid or else get_attribute_by_id will return None
            attr_name = knowledge_graph.get_object_class_by_id(obj.class_id).get_attribute_by_id(attr.id).name

            graph.add_node(attr_node_id, label=f"{attr_name}: {attr.value}", color=obj_color)
            graph.add_edge(obj_node_id, attr_node_id, label="attribute")

    # Add relations to the graph (edges between objects)
    for rel in scene_graph.iter_relations():
        graph.add_edge(
            object_node_id_mapping[rel.subject_id],
            object_node_id_mapping[rel.object_id],
            label=knowledge_graph.get_rule_by_id(rel.rule_id).name
        )

    return graph
