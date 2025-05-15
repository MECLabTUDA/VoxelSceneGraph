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

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QWidget, QLabel, QSizePolicy, QMainWindow, QStackedWidget, QHBoxLayout

from scene_graph_annotation.directed_graphs import dependencies_check
from scene_graph_annotation.knowledge import KnowledgeGraph
from scene_graph_annotation.utils.asset_paths import logo_path
from ._custom_widgets import PatientSelectionWidget, SceneGraphEditorWidget, UIManager


class AnnotationWindow(QMainWindow):
    """
    Window for scene graph annotation.
    Components:
    - Patient SelectionWidget on the left
    - QStackedWidget on the right, with placeholder or SceneGraphEditorWidget for currently selected patient.
    """

    def __init__(
            self,
            knowledge_graph: KnowledgeGraph,
            img_folder: Path,
            scene_graph_folder: Path,
            parent: QWidget | None = None
    ):
        super().__init__(parent=parent)
        self._knowledge_graph = knowledge_graph
        self._img_folder = img_folder
        self._scene_graph_folder = scene_graph_folder

        self._app_name = "Scene Graph Annotator"
        self._left_column_width = 300
        self._min_size = 1350, 800

        self._ui_manager = UIManager()

        # Stacked widget for
        self._stacked_widget = QStackedWidget()

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(self._app_name)
        window_icon = QIcon(logo_path.as_posix())
        self.setWindowIcon(window_icon)
        self.setMinimumSize(*self._min_size)

        central_widget = QWidget()
        central_layout = QHBoxLayout()
        central_widget.setLayout(central_layout)
        central_layout.setContentsMargins(0, 0, 0, 0)
        self.setCentralWidget(central_widget)

        # Patient selection widget
        patient_selection_widget = PatientSelectionWidget(
            self._knowledge_graph,
            self._img_folder,
            self._scene_graph_folder
        )

        # Wrap the patient selection widget in another widget with a layout to have some margin around the widget
        margin_widget = QWidget()
        margin_layout = QHBoxLayout()
        margin_widget.setLayout(margin_layout)
        margin_layout.addWidget(patient_selection_widget)
        # margin_widget.setFixedWidth(self._left_column_width)
        margin_widget.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        central_layout.addWidget(margin_widget)

        patient_selection_widget.currently_loading.connect(self._put_placeholder)
        patient_selection_widget.loading_failed.connect(self._put_default_message)
        patient_selection_widget.patient_selected.connect(self._put_editor_widget)

        # Stacked widget
        placeholder = QLabel()
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Index 0 is placeholder, index 1 is editor
        self._stacked_widget.addWidget(placeholder)
        self._stacked_widget.addWidget(QWidget())
        self._put_default_message()
        self._stacked_widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        central_layout.addWidget(self._stacked_widget)

        # If we can render graphs, then we need to load the dependencies for this here
        # The reason, is that on the first instantiation of QWebEngineView, the engine is sort of updated
        # This causes the window to close and open again
        # This behavior seem to appear on a per-window basis
        # https://forum.qt.io/topic/141398/qwebengineview-closes-reopens-window-when-added-dynamically/8
        # https://doc.qt.io/qt-6/qtwebengine-webenginewidgets-simplebrowser-example.html#creating-the-browser-main-window
        if dependencies_check.check_graph_render_dependencies()[0]:
            from PyQt6.QtWebEngineWidgets import QWebEngineView
            # To do the loading, the window needs to have a QWebEngineView child
            dummy_engine = QWebEngineView(self)
            dummy_engine.setFixedSize(0, 0)
            dummy_engine.page().setHtml("")

        self.showMaximized()

    def _put_default_message(self):
        """Sets the stacked widget on the left to the label and display the default message."""
        self._stacked_widget.setCurrentIndex(0)
        # Always a QLabel at index 0
        self._stacked_widget.currentWidget().setText(f"To start, please select a patient in the left panel.")

    def _put_placeholder(self, patient: str):
        """Sets the stacked widget on the left to the label and display the loading message."""
        self._stacked_widget.setCurrentIndex(0)
        # Always a QLabel at index 0
        widget = self._stacked_widget.currentWidget()
        # Due to concurrency issues, it can happen that the placeholder QLabel has already been replaced
        if isinstance(widget, QLabel):
            widget.setText(f"Patient {patient} is being loaded...")

    def _put_editor_widget(self, editor: SceneGraphEditorWidget):
        """Removes any widget at index 1 in the stacked widget and replaces it with the given editor widget."""
        self._stacked_widget.setCurrentIndex(1)
        prev_widget = self._stacked_widget.currentWidget()
        self._stacked_widget.removeWidget(prev_widget)
        editor.setParent(self)
        self._stacked_widget.addWidget(editor)
        self._stacked_widget.setCurrentIndex(1)
        # Don't delete later as widgets are cached
        # prev_widget.deleteLater()
