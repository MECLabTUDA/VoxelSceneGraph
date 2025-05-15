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

from PyQt6.QtWidgets import QVBoxLayout, QWidget

from scene_graph_annotation.scene import SceneGraph
from scene_graph_annotation.ui_utils import QCollapsibleBox, hbox_layout_with_vertical_line_left
from .RelationRuleListingWidget import RelationRuleListingWidget
from .UIManager import UIManager


class RelationListingWidget(QCollapsibleBox):
    """
    QCollapsibleBox with RelationRuleListingWidget in the collapsible content area for each rule.
    Also displays the total number of relations in the title.
    """

    def __init__(self, scene_graph: SceneGraph, ui_manager: UIManager):
        self._relations_by_rule = scene_graph.relations_by_rule_id
        self._scene_graph = scene_graph
        self._ui_manager = ui_manager

        # Layout for adding more editors
        self._editors_layout = QVBoxLayout()

        super().__init__(start_expanded=True)

        # Optionally connect add/remove objects to signals from manager to update title upon add delete
        ui_manager.relation_added.connect(self._update_title)
        ui_manager.relation_removed.connect(self._update_title)
        # We may need to update the title (number of relations) if an object has been removed
        ui_manager.sg_object_removed.connect(self._update_title)

        # Already done from super().__init__
        # self.init_ui()

    def init_ui(self):
        super().init_ui()

        # Set the editor widget as collapsible content area with a vertical line on the left
        content_area_layout = hbox_layout_with_vertical_line_left()
        editors_widget = QWidget()
        content_area_layout.addWidget(editors_widget)
        editors_widget.setLayout(self._editors_layout)
        self._editors_layout.setContentsMargins(15, 0, 0, 0)

        # Add an editor for each object
        for rel_rule_id in self._relations_by_rule:
            rel_rule = self._scene_graph.knowledge_graph.get_rule_by_id(rel_rule_id)
            listing = RelationRuleListingWidget(
                rel_rule,
                self._relations_by_rule[rel_rule_id],
                self._scene_graph,
                self._ui_manager
            )
            self._editors_layout.addWidget(listing)

        content = QWidget()
        content.setLayout(content_area_layout)
        self.set_content(content)
        self._update_title()

    def _update_title(self, _=None):
        """Updates the title to reflect the number of rules."""
        rel_cnt = sum(len(rel_list) for rel_list in self._relations_by_rule.values())
        self.set_title(f"Relations ({rel_cnt})")
