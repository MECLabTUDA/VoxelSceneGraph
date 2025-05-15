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

from PyQt6.QtGui import QIcon, QStandardItemModel, QStandardItem
from PyQt6.QtWidgets import QListView, QSizePolicy

from scene_graph_annotation.knowledge import RelationRule
from scene_graph_annotation.scene import Relation, SceneGraph, Object
from scene_graph_annotation.ui_utils import QCollapsibleBox, QAlignRightButton
from scene_graph_annotation.utils.asset_paths import delete_icon
from .UIManager import UIManager


class RelationRuleListingWidget(QCollapsibleBox):
    """
    QCollapsibleBox with a list view in the collapsible content area for a SPECIFIC relation rule.
    Relations are simply displayed in their string form with a delete button on the right.
    Also displays the number of relations in the title.
    Clicking a relation will select the relation subject and object.
    """

    def __init__(
            self,
            rule_class: RelationRule,
            relations: list[Relation],  # Direct reference to content of the scene graph
            scene_graph: SceneGraph,  # Required to convert obj id in Relations to obj name
            ui_manager: UIManager,
    ):
        self._rule_class = rule_class
        self._relations = relations
        self._scene_graph = scene_graph
        self._ui_manager = ui_manager

        self._delete_icon = QIcon(delete_icon.as_posix())
        # Widget for holding relations
        self._list_view = QListView()
        self._list_model = QStandardItemModel(self._list_view)
        self._list_view.setModel(self._list_model)

        super().__init__()

        # Connect signals
        ui_manager.relation_added.connect(self._relation_added)
        ui_manager.relation_removed.connect(self._relation_removed)
        ui_manager.name_changed.connect(self._object_renamed_added_removed)
        # Since there is some object merging/splitting, anything can happen
        # Additionally relations may be modified in place, so we have to redo the entire list
        ui_manager.sg_object_added.connect(self._object_renamed_added_removed)
        ui_manager.sg_object_removed.connect(self._object_renamed_added_removed)

        # Already done from super().__init__
        # self.init_ui()

    def init_ui(self):
        super().init_ui()

        self._list_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.MinimumExpanding)
        self._reset_list_content()
        self.set_content(self._list_view)

        # Select subject and object on relation click
        self._list_view.clicked.connect(
            lambda index: self._relation_selected(self._list_model.itemFromIndex(index).data())
        )

    def _reset_list_content(self):
        """Empties the list view and builds the items from the ground up. Also updates the title."""
        # Clear the list
        self._list_model.removeRows(0, self._list_model.rowCount())

        # Fill it back
        for relation in self._relations:
            self._append_row(relation)

        self._set_tile()

    def _set_tile(self):
        """Set the title."""
        self.set_title(f"{self._rule_class.name} ({len(self._relations)})")

    def _append_row(self, relation: Relation):
        """Append a row for given relation."""
        # Relation to str
        subj = self._scene_graph.get_bounding_box_by_id(relation.subject_id)
        obj = self._scene_graph.get_bounding_box_by_id(relation.object_id)
        rel_str = f"{subj.name} {self._rule_class.name} {obj.name}"

        item = QStandardItem(rel_str)
        self._list_model.appendRow(item)
        item.setData(relation)
        item.setToolTip("Select relation")

        button = QAlignRightButton(self._delete_icon)
        self._list_view.setIndexWidget(item.index(), button)
        button.clicked.connect(lambda _, rel=relation: self._delete_relation(rel))
        button.setToolTip("Delete relation")

    def _delete_relation(self, rel: Relation):
        """Callback for deleting a relation using the button."""
        self._scene_graph.remove_relation(rel)
        self._ui_manager.relation_removed.emit(rel)

    def _relation_added(self, relation: Relation):
        """If the relation that was added was for this rule, append a row."""
        if relation.rule_id == self._rule_class.id:
            # The relation was for this rule, so append row
            self._append_row(relation)
            self._set_tile()

    def _relation_removed(self, relation: Relation):
        """If the relation that was deleted was for this rule, remove the corresponding row."""
        if relation.rule_id != self._rule_class.id:
            return

        # The relation was for this rule, so remove appropriate row
        for row in range(self._list_model.rowCount()):
            index = self._list_model.index(row, 0)
            if self._list_model.itemFromIndex(index).data() == relation:
                # Remove row and update title
                self._list_model.removeRows(row, 1)
                self._set_tile()

                # If there is a row after this one, trigger the callback for the cursor entering and select next row
                # (as it's not done automatically)
                next_index = self._list_model.index(row, 0)
                if next_index.isValid():
                    # noinspection PyTypeChecker
                    self._list_view.indexWidget(next_index).enterEvent(None)
                    self._list_view.setCurrentIndex(next_index)

                break

    def _object_renamed_added_removed(self, obj: Object):
        """
        After an object was renamed/added/removed, check if any of our relations may have been updated.
        If so reset the list view content.
        """
        # Check if the class id of the object CAN appear in one of our relations
        # If not, just end processing here
        if not self._rule_class.subject_filter.is_class_authorized(obj.class_id) and \
                not self._rule_class.object_filter.is_class_authorized(obj.class_id):
            return

        # Check in any of our rules contains this object
        # If yes, reset the list
        for rel in self._relations:
            if rel.subject_id == obj.id or rel.object_id == obj.id:
                self._reset_list_content()
                break

    def _relation_selected(self, relation: Relation):
        """A relation was selected: also select the subject and object."""
        self._ui_manager.subject = self._scene_graph.get_bounding_box_by_id(relation.subject_id)
        self._ui_manager.object = self._scene_graph.get_bounding_box_by_id(relation.object_id)
