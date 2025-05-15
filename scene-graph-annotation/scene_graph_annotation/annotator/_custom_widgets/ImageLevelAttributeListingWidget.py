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

from scene_graph_annotation.knowledge import ObjectClass
from scene_graph_annotation.scene import BoundingBox, CompositeBoundingBox, Object
from scene_graph_annotation.ui_utils import QCollapsibleBox, hbox_layout_with_vertical_line_left
from .ImageLevelAttributesEditorWidget import ImageLevelAttributesEditorWidget

_Objects = BoundingBox | CompositeBoundingBox


class ImageLevelAttributeListingWidget(QCollapsibleBox):
    """QCollapsibleBox with a single ImageLevelAttributeListingWidget in the collapsible content area."""

    def __init__(self, image_object_class: ObjectClass, image_object: Object):
        # Layout for adding more editors
        self._editors_layout = QVBoxLayout()
        self._editor_widget = ImageLevelAttributesEditorWidget(image_object_class, image_object)

        super().__init__(start_expanded=True)

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

        # Add the editor
        self._editors_layout.addWidget(self._editor_widget)

        content = QWidget()
        content.setLayout(content_area_layout)
        self.set_content(content)

        self.set_title("Image-Level")
