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

from PyQt6.QtWidgets import QTabWidget

from scene_graph_annotation.scene import SceneGraph
from .SceneGraphEditorWidget import SceneGraphEditorWidget
from ...ui_utils import QGraphView
from ...utils.ArrayView import ArrayView
from ...utils.progress import Progress


class PatientWidget(QTabWidget):
    """
    QTabWidget with 2 tabs:
    - a SceneGraphEditorWidget to edit the scene graph
    - a QGraphView to view the scene graph
    Note: an alias for the editor's signal is created for convenience.
    """

    def __init__(
            self,
            patient: str,  # Patient id from the PatientSelection widget
            image: ArrayView,
            scene_graph: SceneGraph,
            scene_graph_save_path: Path,
            progress: Progress  # Annotation progress for the patient to dynamically set the default state on save
    ):
        super().__init__()

        self._editor = SceneGraphEditorWidget(patient, image, scene_graph, scene_graph_save_path, progress)
        self._graph_view = QGraphView(scene_graph)

        # Create alias for signal
        self.scene_graph_saved = self._editor.scene_graph_saved

        # Bind signals such that the graph view gets updated when needed
        self._editor.ui_manager.name_changed.connect(self._graph_view.clear_graph)
        self._editor.ui_manager.sg_object_added.connect(self._graph_view.clear_graph)
        self._editor.ui_manager.sg_object_removed.connect(self._graph_view.clear_graph)
        self._editor.ui_manager.relation_added.connect(self._graph_view.clear_graph)
        self._editor.ui_manager.relation_removed.connect(self._graph_view.clear_graph)

        self.currentChanged.connect(self._on_tab_change)

        self.init_ui()

    def init_ui(self):
        self.addTab(self._editor, "Editor")
        self.addTab(self._graph_view, "Scene Graph")

        self.setContentsMargins(0, 0, 0, 0)

    # noinspection PyPep8Naming
    def _on_tab_change(self, idx: int):
        if idx == 1:
            # Tell the graph view to update its content
            self._graph_view.on_view()
