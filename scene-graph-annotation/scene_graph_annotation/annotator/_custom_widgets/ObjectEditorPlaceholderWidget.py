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

import enum

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget, QLabel, QSizePolicy, QGroupBox, QStackedWidget, QVBoxLayout, QScrollArea

from scene_graph_annotation.knowledge import KnowledgeGraph
from .ObjectEditorWidget import ObjectEditorWidget
from .UIManager import UIManager


class ObjectEditorPlaceholderWidget(QGroupBox):
    """
    Widget used to hold the subject OR object for relation edition (if any is selected).
    Basically a QGroupbox containing a ScrollArea with a QStackedWidget.
    The QStackedWidget will either display a Label saying "No {} selected" or the ObjectEditorWidget for the object.
    """

    class ObjectRole(enum.Enum):
        SUBJECT = "Subject"
        OBJECT = "Object"

    def __init__(
            self,
            object_role: ObjectRole,
            knowledge_graph: KnowledgeGraph,  # Required to fetch the object class to build the editor widget
            ui_manager: UIManager
    ):
        super().__init__(title=str(object_role.value))

        self._object_role = object_role
        self._knowledge_graph = knowledge_graph
        self._ui_manager = ui_manager

        self._stacked_widget = QStackedWidget()

        # Connect update methods to signals
        if object_role == self.ObjectRole.SUBJECT:
            ui_manager.subject_changed.connect(self._subject_or_object_changed)
        else:
            ui_manager.object_changed.connect(self._subject_or_object_changed)

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

        # Add ScrollArea
        scroll_area = QScrollArea()
        layout.addWidget(scroll_area)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setWidgetResizable(True)

        # Add stacked widget
        placeholder = QLabel(f"No {str(self._object_role.value).lower()} selected")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Index 0 is placeholder, index 1 is editor
        self._stacked_widget.addWidget(placeholder)
        self._stacked_widget.addWidget(QWidget())
        self._stacked_widget.setCurrentIndex(0)
        self._stacked_widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        scroll_area.setWidget(self._stacked_widget)

        layout.addWidget(scroll_area)
        # Call this in case the subject/object is already not None when this widget is instanced
        self._subject_or_object_changed()

    def _subject_or_object_changed(self):
        """
        Depending on which is displayed (subject or object):
        - checks if it's None, in which case a text placeholder is displayed
        - we create a new ObjectEditorWidget, and we update the stacked widget
        """
        if self._object_role == self.ObjectRole.SUBJECT:
            sg_object = self._ui_manager.subject
        else:
            sg_object = self._ui_manager.object

        # Case: sg_object was deselected (because of merge or split)
        if sg_object is None:
            if self._stacked_widget.currentIndex() == 0:
                return
            # Replace editor with empty widget and switch to placeholder
            prev_widget = self._stacked_widget.currentWidget()
            self._stacked_widget.removeWidget(prev_widget)
            self._stacked_widget.addWidget(QWidget())
            self._stacked_widget.setCurrentIndex(0)
            prev_widget.deleteLater()
            return

        # Else we need to swap out the widget at index 1
        self._stacked_widget.setCurrentIndex(1)
        prev_widget = self._stacked_widget.currentWidget()
        self._stacked_widget.removeWidget(prev_widget)
        prev_widget.deleteLater()
        new_widget = ObjectEditorWidget(self._knowledge_graph.get_object_class_by_id(sg_object.class_id),
                                        sg_object,
                                        self._ui_manager)
        self._stacked_widget.addWidget(new_widget)
        self._stacked_widget.setCurrentIndex(1)
