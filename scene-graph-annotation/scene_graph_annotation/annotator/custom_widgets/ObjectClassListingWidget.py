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

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QVBoxLayout, QWidget, QLabel, QSizePolicy

from scene_graph_annotation.knowledge import ObjectClass
from scene_graph_annotation.scene import BoundingBox, CompositeBoundingBox, SceneGraph
from scene_graph_annotation.ui_utils import QCollapsibleBox, hbox_layout_with_vertical_line_left
from .CollapsibleObjectEditorWidget import CollapsibleObjectEditorWidget
from .UIManager import UIManager

_SgObject = BoundingBox | CompositeBoundingBox


class ObjectClassListingWidget(QCollapsibleBox):
    """
    QCollapsibleBox with ObjectEditorWidget in the collapsible content area for a SPECIFIC object class.
    Also displays the number of objects in the title.
    Also displays the color of the object class in a small label.
    """

    def __init__(
            self,
            scene_graph: SceneGraph,
            object_class: ObjectClass,
            objects: list[_SgObject],
            ui_manager: UIManager
    ):
        self._scene_graph = scene_graph
        self._object_class = object_class
        self._objects = objects
        self._ui_manager = ui_manager

        # Useful we ever need to delete widgets (e.g. with segmentations)
        self._object_editor_by_id: dict[int, CollapsibleObjectEditorWidget] = {}

        # Layout for adding more editors
        self._editors_layout = QVBoxLayout()

        super().__init__(start_expanded=True)

        # Connect add/remove objects to signals from manager
        # to automatically add/remove widgets and update title upon merge/split
        ui_manager.sg_object_added.connect(self.add_object)
        ui_manager.sg_object_removed.connect(self.remove_object)

        # Already done from super().__init__
        # self.init_ui()

    def init_ui(self):
        super().init_ui()

        # Add a small label with the class color to the right
        color_label = QLabel()
        color_label.setStyleSheet(f"background-color: {self._object_class.color};")
        color_label.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        color_label.setMinimumWidth(30)
        self._top_row_layout.addWidget(color_label, alignment=Qt.AlignmentFlag.AlignRight)
        color_label.setToolTip("Annotation color")

        # Set the editor widget as collapsible content area with a vertical line on the left
        content_area_layout = hbox_layout_with_vertical_line_left()
        editors_widget = QWidget()
        content_area_layout.addWidget(editors_widget)
        editors_widget.setLayout(self._editors_layout)
        self._editors_layout.setContentsMargins(0, 0, 0, 0)
        editors_widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        # Add an editor for each object
        for obj in self._objects:
            self.add_object(obj)

        content = QWidget()
        content.setLayout(content_area_layout)
        self.set_content(content)
        self._update_title()

    def add_object(self, obj: _SgObject):
        """Adds a widget for the object if the object class id matches with ours."""
        if obj.class_id != self._object_class.id:
            return
        editor = CollapsibleObjectEditorWidget(self._scene_graph, self._object_class, obj, self._ui_manager)
        self._object_editor_by_id[obj.id] = editor
        self._editors_layout.addWidget(editor)
        # Update title at each addition to change the object count
        self._update_title()

    def remove_object(self, obj: _SgObject):
        """Deletes the widget associated with the associated object if the object class id matches with ours."""
        if obj.class_id != self._object_class.id:
            return
        editor = self._object_editor_by_id[obj.id]
        del self._object_editor_by_id[obj.id]
        editor.deleteLater()
        # Update title at each removal to change the object count
        self._update_title()

    def _update_title(self):
        """Updates the title to reflect the number of items displayed."""
        self.set_title(f"{self._object_class.name} ({len(self._object_editor_by_id)})")
