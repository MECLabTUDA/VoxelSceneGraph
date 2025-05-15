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

from pathlib import Path

from PyQt6.QtCore import Qt, QTime, pyqtSignal
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import QWidget, QSizePolicy, QGroupBox, QVBoxLayout, QHBoxLayout, QScrollArea, QFrame, \
    QPushButton, QMessageBox, QComboBox, QFormLayout, QLabel

from scene_graph_annotation.knowledge import RadiologyImageKG, NaturalImageKG
from scene_graph_annotation.scene import SceneGraph
from scene_graph_annotation.utils import ArrayView
from .ImageLevelAttributeListingWidget import ImageLevelAttributeListingWidget
from .NaturalImageViewerWidget import NaturalImageViewerWidget
from .ObjectListingWidget import ObjectListingWidget
from .RadiologyImageViewerWidget import RadiologyImageViewerWidget
from .RelationListingWidget import RelationListingWidget
from .SubjectObjectNewRelationWidget import SubjectObjectNewRelationWidget
from .UIManager import UIManager
from ...utils.progress import Progress


class SceneGraphEditorWidget(QWidget):
    """
    Widget used to edit the entire scene graph.
    Contains:
    - A ScrollArea on the left for displaying bounding boxes, segmentations and relations.
      Under it, a ComboBox to select the progress on save (default to FINISHED if current prog is not PENDING_REVIEW).
      Note: ObjectListingWidget for bounding boxes/segmentations are only displayed if there is any object of that type.
    - On the right: a widget to display the image top, a SubjectObjectNewRelationWidget bottom and save button
      The tooltip for the save button shows when was the last save done (if any).
    """
    SAVE_SHORTCUT = "Ctrl+S"

    # Signal used to notify that the scene graph for this patient was saved.
    # The patient id/name is emitted to identify the scene graph.
    scene_graph_saved = pyqtSignal(str, Progress)

    def __init__(
            self,
            patient: str,  # Patient id from the PatientSelection widget
            image: ArrayView,
            scene_graph: SceneGraph,
            scene_graph_save_path: Path,
            progress: Progress  # Annotation progress for the patient to dynamically set the default state on save
    ):
        super().__init__()

        self._patient = patient
        self._image = image
        self._scene_graph = scene_graph
        self._scene_graph_save_path = scene_graph_save_path
        self._progress = progress  # Only needed for init_ui
        self.ui_manager = UIManager()

        # Max width to avoid the two columns to be split 50/50 in terms of width when expanding the window
        self._scroll_area_width = 350
        self._progress_combobox = QComboBox()
        self._save_sg_button = QPushButton()

        self._column_right = QWidget()

        match self._scene_graph.knowledge_graph:
            case RadiologyImageKG():
                self._image_widget = RadiologyImageViewerWidget(self._image, self._scene_graph, self.ui_manager)
            case NaturalImageKG():
                self._image_widget = NaturalImageViewerWidget(self._image, self._scene_graph, self.ui_manager)
            case _:
                raise RuntimeError(f"Unexpected knowledge graph type {type(self._scene_graph.knowledge_graph)}")

        # Set keyboard shortcut for saving
        self.shortcut = QShortcut(QKeySequence(self.SAVE_SHORTCUT), self)
        self.shortcut.activated.connect(self._save_sg)

        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout()
        self.setLayout(layout)
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)

        # ==============================================================================================================
        # Scroll area on the left
        left_column = QWidget()
        layout.addWidget(left_column)
        left_column_layout = QVBoxLayout()
        left_column.setLayout(left_column_layout)
        left_column_layout.setContentsMargins(0, 0, 0, 0)

        scroll_area = QScrollArea()
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setFixedWidth(self._scroll_area_width)

        content_widget = QGroupBox("Scene graph content")
        content_layout = QVBoxLayout()
        content_widget.setLayout(content_layout)
        content_widget.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Preferred)

        # Image level attributes (if any)
        if self._scene_graph.knowledge_graph.image.attributes:
            image_widget = ImageLevelAttributeListingWidget(
                self._scene_graph.knowledge_graph.image, self._scene_graph.image
            )
            content_layout.addWidget(image_widget)

        # Bounding boxes
        if len(self._scene_graph.bounding_boxes_by_class_id) > 0:
            bb_widget = ObjectListingWidget(
                self._scene_graph,
                self._scene_graph.bounding_boxes_by_class_id,
                self.ui_manager,
                "Objects"
            )
            content_layout.addWidget(bb_widget)

        # # Segmentations
        # if len(self._scene_graph.segmentations_by_class_id) > 0:
        #     seg_widget = ObjectListingWidget(
        #         self._scene_graph,
        #         self._scene_graph.segmentations_by_class_id,
        #         self.ui_manager,
        #         "Segmentations"
        #     )
        #     content_layout.addWidget(seg_widget)

        # Relations
        rel_widget = RelationListingWidget(self._scene_graph, self.ui_manager)
        content_layout.addWidget(rel_widget)

        # Add a stretch to keep all widgets aligned top
        content_layout.addStretch()

        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(content_widget)
        left_column_layout.addWidget(scroll_area)

        # Add ComboBox for the progress
        progress_widget = QWidget()
        left_column_layout.addWidget(progress_widget)
        progress_layout = QFormLayout()
        progress_widget.setLayout(progress_layout)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        progress_layout.addRow(QLabel("Progress:"), self._progress_combobox)

        # Add items
        for prog in Progress:
            self._progress_combobox.addItem(prog.name.replace("_", " ").title())

        # Set default (PENDING_REVIEW -> PENDING_REVIEW; default: FINISHED)
        if self._progress == Progress.PENDING_REVIEW:
            self._progress_combobox.setCurrentIndex(Progress.PENDING_REVIEW.value - 1)
        else:
            self._progress_combobox.setCurrentIndex(Progress.FINISHED.value - 1)

        # ==============================================================================================================
        # Part on the right
        self._column_right = QWidget()
        layout_right = QVBoxLayout()
        self._column_right.setLayout(layout_right)
        layout_right.setContentsMargins(0, 0, 0, 0)

        # Image display widget
        self._image_widget.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Fixed)
        layout_right.addWidget(self._image_widget, alignment=Qt.AlignmentFlag.AlignCenter)

        # Relation editor widget
        relation_editor_widget = SubjectObjectNewRelationWidget(self._scene_graph, self.ui_manager)
        layout_right.addWidget(relation_editor_widget)

        # Save button
        self._save_sg_button.setText("Save scene graph")
        self._save_sg_button.clicked.connect(self._save_sg)
        self._save_sg_button.setToolTip("Not saved yet this session")
        layout_right.addWidget(self._save_sg_button)

        layout.addWidget(self._column_right)

    def _save_sg(self):
        """Callback for saving a scene graph with message boxes to confirm the success."""
        if self._scene_graph.save(self._scene_graph_save_path.as_posix()):
            self._save_sg_button.setToolTip("Last saved " + QTime.currentTime().toString())
            self.scene_graph_saved.emit(self._patient, Progress(self._progress_combobox.currentIndex() + 1))
            QMessageBox.about(self, "Success", "The scene graph was saved successfully!")
        else:
            QMessageBox.warning(self, "Error", "The scene graph could not be saved. "
                                               "Please make sure that the destination folder still exists.")
