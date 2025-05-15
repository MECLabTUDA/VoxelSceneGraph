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

from scene_graph_annotation.scene import BoundingBox, CompositeBoundingBox, SceneGraph
from scene_graph_annotation.ui_utils import QCollapsibleBox, hbox_layout_with_vertical_line_left
from .ObjectClassListingWidget import ObjectClassListingWidget
from .UIManager import UIManager

_Objects = BoundingBox | CompositeBoundingBox


class ObjectListingWidget(QCollapsibleBox):
    """
    QCollapsibleBox with ObjectClassListingWidget in the collapsible content area for each object class.
    Also displays the total number of objects in the title.
    """

    def __init__(
            self,
            scene_graph: SceneGraph,
            objects_by_class: dict[int, list[_Objects]],
            ui_manager: UIManager,
            object_type_display_name: str,  # Either "Bounding boxes" or "Segmentations"
    ):
        self._scene_graph = scene_graph
        self._objects_by_class = objects_by_class
        self._ui_manager = ui_manager
        self._object_type_display_name = object_type_display_name

        # Layout for adding more editors
        self._editors_layout = QVBoxLayout()

        super().__init__(start_expanded=True)

        # Connect add/remove objects to signals from manager
        # to automatically update title upon merge/split
        ui_manager.sg_object_added.connect(self._update_title)
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
        self._editors_layout.setContentsMargins(0, 0, 0, 0)

        # Add an editor for each object
        for obj_class_id in self._objects_by_class:
            obj_class = self._scene_graph.knowledge_graph.get_object_class_by_id(obj_class_id)
            listing = ObjectClassListingWidget(
                self._scene_graph,
                obj_class,
                self._objects_by_class[obj_class_id],
                self._ui_manager
            )
            self._editors_layout.addWidget(listing)

        content = QWidget()
        content.setLayout(content_area_layout)
        self.set_content(content)
        self._update_title()

    def _update_title(self, _=None):
        """Callback for updating the title. Called for instance when the number of object changes."""
        obj_cnt = sum(len(obj_list) for obj_list in self._objects_by_class.values())
        self.set_title(f"{self._object_type_display_name} ({obj_cnt})")
