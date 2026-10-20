import logging
from pathlib import Path

import qt

from scene_graph_api.knowledge import KnowledgeGraph
from scene_graph_database.options import Options
from scene_graph_database.structures import StudyStructure
from ..load_state import LOAD_STATE_MANAGER
from ..utils import load_studies_from_db, confirm_discard_changes
from .patient_table import PatientProgressTable


class StudiesTab(qt.QWidget):
    """Tab widget that contains a study selection ComboBox and a PatientProgressTable."""

    def __init__(self):
        super().__init__()

        self._studies_combobox = qt.QComboBox()
        self._study_refresh_button = qt.QPushButton()
        self.patient_table = PatientProgressTable()

        # Connect to the study list update
        LOAD_STATE_MANAGER.study_list_updated.connect(self._set_studies)
        LOAD_STATE_MANAGER.select_patient_by_offset.connect(self.patient_table.select_patient_relative_to_current)

        self.init_ui()

    def init_ui(self):
        layout = qt.QFormLayout()
        self.setLayout(layout)

        # Intermediate widget with combobox and refresh button
        interm_widget = qt.QWidget()
        interm_layout = qt.QHBoxLayout()
        interm_layout.setContentsMargins(0, 0, 0, 0)
        interm_widget.setLayout(interm_layout)
        interm_layout.addWidget(self._studies_combobox, 1)
        interm_layout.addWidget(self._study_refresh_button, 0)

        layout.addRow(qt.QLabel("Study:"), interm_widget)

        # Intermediate widget with combobox and refresh button
        self._study_refresh_button.setIcon(qt.QApplication.style().standardIcon(qt.QStyle.SP_BrowserReload))
        self._study_refresh_button.clicked.connect(lambda: load_studies_from_db(Options.get_db_string()))

        self._set_studies()
        self._studies_combobox.currentIndexChanged.connect(self._study_selected)

        # Patient table
        layout.addRow(self.patient_table)

    def _set_studies(self):
        """Set the studies to display in the widget and update the UI."""
        studies = LOAD_STATE_MANAGER.study_list
        self._studies_combobox.clear()
        if studies is None:
            self._studies_combobox.setEnabled(False)
            self._study_refresh_button.setEnabled(False)
            return

        self._studies_combobox.addItem("")
        for study in studies:
            self._studies_combobox.addItem(study.name)
        self._studies_combobox.setEnabled(True)
        self._study_refresh_button.setEnabled(True)

    def _study_selected(self):
        """Updates the load state manager when a study is selected."""
        # Check that the patient is not modified
        if LOAD_STATE_MANAGER.selected_patient_scene_graph_modified and not confirm_discard_changes("Loading a new study"):
            return

        idx = self._studies_combobox.currentIndex
        if idx == 0:
            LOAD_STATE_MANAGER.clear_study_selection()
            return

        # Check that we can load the knowledge graph
        study = LOAD_STATE_MANAGER.study_list[idx - 1]
        # Note: study.knowledge_graph_path is relative to the database
        structure = StudyStructure(Path(Options.get_db_string()), study.knowledge_graph_path)

        logger = logging.getLogger(__name__)
        knowledge_graph = KnowledgeGraph.load(structure.knowledge_graph, logger)
        if knowledge_graph is None:
            print("study_selected: Could not load knowledge graph")
            return

        # Update the state
        LOAD_STATE_MANAGER.knowledge_graph = knowledge_graph
        LOAD_STATE_MANAGER.set_new_study_idx(idx - 1, knowledge_graph)
