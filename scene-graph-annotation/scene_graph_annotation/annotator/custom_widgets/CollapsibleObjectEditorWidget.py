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

from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QWidget, QPushButton, QSizePolicy, QToolButton, QMessageBox

from scene_graph_annotation.knowledge import ObjectClass
from scene_graph_annotation.scene import BoundingBox, Object, SceneGraph
from scene_graph_annotation.ui_utils import QCollapsibleBox, hbox_layout_with_vertical_line_left
from scene_graph_annotation.utils.asset_paths import magnifier_icon, delete_icon
from .ObjectEditorWidget import ObjectEditorWidget
from .UIManager import UIManager


class CollapsibleObjectEditorWidget(QCollapsibleBox):
    """
    QCollapsibleBox with a ObjectEditorWidget in the collapsible content area.
    Top row shading becomes darker when the displayed object is selected as subject/object.
    Top row buttons are:
    - magnifier: selects a "correct" slice that we will show this object (only does something in 3D)
    - S: select this object as the subject (or unselect it if already selected)
    - O: select this object as the object (or unselect it if already selected)
    - bin: delete this object
    """

    def __init__(
            self,
            scene_graph: SceneGraph,
            object_class: ObjectClass,
            object_instance: BoundingBox,
            ui_manager: UIManager
    ):
        self._scene_graph = scene_graph
        self._object_class = object_class
        self._object = object_instance
        self._ui_manager = ui_manager

        self._object_editor = ObjectEditorWidget(object_class, object_instance, ui_manager)
        self.set_subject_button = QPushButton()
        self.set_object_button = QPushButton()

        # Used to change the shading to a darker style when the object is selected
        # Makes subject/object easier to find in the list
        self._object_is_selected = ui_manager.subject == self._object or ui_manager.object == self._object
        self._darker_shading_start = "#ecedf0"
        self._darker_shading_stop = "#c8c9ca"

        super().__init__(title=object_instance.name)

        # Just connect the name update method
        ui_manager.name_changed.connect(self._name_changed)
        ui_manager.subject_changed.connect(self._subject_changed)
        ui_manager.object_changed.connect(self._object_changed)

        # Already done from super().__init__
        # self.init_ui()

    def init_ui(self):
        super().init_ui()

        # Add a button to show the object
        show_object_button = QToolButton()
        self._top_row_layout.addWidget(show_object_button)
        show_object_button.setIcon(QIcon(magnifier_icon.as_posix()))
        show_object_button.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        show_object_button.setMaximumWidth(show_object_button.sizeHint().height())
        show_object_button.setToolTip("Show object")
        show_object_button.clicked.connect(lambda: self._ui_manager.show_object.emit(self._object.id))

        # Add buttons for selecting object as subject / object
        self._top_row_layout.addWidget(self.set_subject_button)
        self.set_subject_button.setText("S")
        self.set_subject_button.setStyleSheet("font-weight: bold;")
        self.set_subject_button.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.set_subject_button.setMaximumWidth(self.set_subject_button.sizeHint().height())
        self.set_subject_button.setToolTip("Set as subject")
        self.set_subject_button.setCheckable(True)
        self.set_subject_button.setChecked(False)
        self.set_subject_button.clicked.connect(self._select_as_subject)

        self.set_object_button = QPushButton()
        self._top_row_layout.addWidget(self.set_object_button)
        self.set_object_button.setText("O")
        self.set_object_button.setStyleSheet("font-weight: bold;")
        self.set_object_button.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.set_object_button.setMaximumWidth(self.set_object_button.sizeHint().height())
        self.set_object_button.setToolTip("Set as object")
        self.set_object_button.setCheckable(True)
        self.set_object_button.setChecked(False)
        self.set_object_button.clicked.connect(self._select_as_object)

        # Add button for deleting the object
        delete_object_button = QToolButton()
        self._top_row_layout.addWidget(delete_object_button)
        delete_object_button.setIcon(QIcon(delete_icon.as_posix()))
        delete_object_button.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        delete_object_button.setMaximumWidth(delete_object_button.sizeHint().height())
        delete_object_button.setToolTip("Delete object")
        delete_object_button.clicked.connect(self._delete_object)

        # Set the editor widget as collapsible content area with a vertical line on the left
        content_area_layout = hbox_layout_with_vertical_line_left()
        content_area_layout.addWidget(self._object_editor)

        content = QWidget()
        content.setLayout(content_area_layout)
        self.set_content(content)
        # Set shading in case the object is already selected
        # Note: putting this call at the start of the method (and after the super call) does not seem to work
        self._set_shading()

    def _name_changed(self, obj: Object):
        """Updates the title if the object's name has changed."""
        if obj == self._object:
            self.set_title(self._object.name)

    def _select_as_subject(self):
        """Sets this scene graph object as the subject (or unselects it if already selected)."""
        if self._ui_manager.subject != self._object:
            self._ui_manager.subject = self._object
        else:
            self._ui_manager.subject = None

    def _select_as_object(self):
        """Sets this scene graph object as the object (or unselects it if already selected)."""
        if self._ui_manager.object != self._object:
            self._ui_manager.object = self._object
        else:
            self._ui_manager.object = None

    def _delete_object(self):
        """
        Delete this object and remove it from the scene graph.
        Note: opens a confirmation QMessageBox.
        """
        reply = QMessageBox.question(
            self,
            "Are you sure?",
            f"Are you sure that you want to delete the object named \"{self._object.name}\"?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        # If this object was selected, unselect it
        if self._ui_manager.subject == self._object:
            self._ui_manager.subject = None
        if self._ui_manager.object == self._object:
            self._ui_manager.object = None

        # Remove relations associated to the object
        # Note: it's important to delete the relation before the object, as this can trigger some widgets
        for rel_list in self._scene_graph.relations_by_rule_id.values():
            for rel in rel_list.copy():
                if rel.subject_id == self._object.id or rel.object_id == self._object.id:
                    self._scene_graph.remove_relation(rel)
                    self._ui_manager.relation_removed.emit(rel)

        # Remove the object from the labelmap
        # Note: it's also important to update the overlay before removing the object,
        #       as removing it will already trigger a canvas update
        mask = self._scene_graph.object_labelmap == self._object.id
        self._scene_graph.object_hitboxes[mask] = 0
        self._scene_graph.object_overlay[mask] = 0
        self._scene_graph.object_labelmap[mask] = 0

        # Remove object from scene graph
        self._scene_graph.remove_bounding_box(self._object)
        self._ui_manager.sg_object_removed.emit(self._object)

    def _subject_changed(self):
        """Shorthand for _subject_or_object_changed when the subject changes."""
        self._subject_or_object_changed(self._ui_manager.subject)

    def _object_changed(self):
        """Shorthand for _subject_or_object_changed when the object changes."""
        self._subject_or_object_changed(self._ui_manager.object)

    def _subject_or_object_changed(self, obj: Object):
        """
        Callback for when the subject or object changed and:
        - we update the checked state of the set subject/object buttons
        - we check if we need to change the shading
        """
        self.set_subject_button.setChecked(self._ui_manager.subject == self._object)
        self.set_object_button.setChecked(self._ui_manager.object == self._object)

        if obj != self._object:
            # Another object was just selected as subject or object, check if our object is still selected
            self._object_is_selected = self._ui_manager.subject == self._object or \
                                       self._ui_manager.object == self._object
            if not self._object_is_selected:
                # Update only if our object is now not selected
                self._set_shading()
            return
        # Our object was just selected as subject or object
        self._object_is_selected = True
        self._set_shading()

    def _set_shading(self):
        """Changes the top row shading to a darker variant when our object is selected."""
        if self._object_is_selected:
            self._set_top_row_shading(self._darker_shading_start, self._darker_shading_stop)
        else:
            self._set_top_row_shading(self._shading_start, self._shading_stop)
