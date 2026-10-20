import os
import platform
import subprocess
from pathlib import Path

import qt
from scene_graph_api.utils.pathing import remove_suffixes
from scene_graph_database.options import Options
from scene_graph_database.structures import AnnotationState, StudyStructure

from ..load_state import LOAD_STATE_MANAGER


class PatientTab(qt.QWidget):
    """Tab widget that contains comments and progress status about the current patient."""

    def __init__(self):
        super().__init__()

        self._patient_name_label = qt.QLabel()
        self._open_report_button = qt.QPushButton("Open")
        self._patient_progress = qt.QComboBox()
        self._comment_text = qt.QTextEdit()

        # Connect to the study list update
        LOAD_STATE_MANAGER.sg_loaded.connect(self._update_ui)

        self.init_ui()

    def init_ui(self):
        layout = qt.QFormLayout()
        self.setLayout(layout)

        layout.addRow(qt.QLabel("Patient:"), self._patient_name_label)

        layout.addRow(qt.QLabel("Report:"), self._open_report_button)
        self._open_report_button.clicked.connect(self._open_report)

        layout.addRow(qt.QLabel("Progress:"), self._patient_progress)
        for state in AnnotationState:
            self._patient_progress.addItem(state.to_str().replace("\n", " "))

        def f():
            LOAD_STATE_MANAGER.annotation_progress = AnnotationState(self._patient_progress.currentIndex)

        self._patient_progress.currentIndexChanged.connect(f)

        layout.addRow(qt.QLabel("Comment:"), self._comment_text)
        self._comment_text.textChanged.connect(self._handle_comment_changed)

        self._update_ui()

    def _update_ui(self):
        """Sets the state of form widgets."""
        name = LOAD_STATE_MANAGER.selected_patient_name
        self._patient_name_label.setText(name if name is not None else "(None)")
        scene_graph = LOAD_STATE_MANAGER.scene_graph

        # No patient
        if scene_graph is None:
            self._open_report_button.setEnabled(False)
            self._patient_progress.setEnabled(False)
            self._comment_text.setEnabled(False)
            return

        self._open_report_button.setEnabled(self._get_report_path().is_file())
        self._patient_progress.setEnabled(True)
        self._patient_progress.setCurrentIndex(LOAD_STATE_MANAGER.annotation_progress.value)
        self._comment_text.setEnabled(True)
        self._comment_text.setPlainText(scene_graph.comment)

    @staticmethod
    def _get_report_path() -> Path | None:
        """Get the path to the report of a patient in a study."""
        # Sanity check
        study = LOAD_STATE_MANAGER.selected_study
        selected_patient_name = LOAD_STATE_MANAGER.selected_patient_name
        if study is None or selected_patient_name is None:
            print("get_report_path: No study or patient selected")
            return

        # Construct path and check that a report is available
        structure = StudyStructure(Path(Options.get_db_string()), study.knowledge_graph_path)
        return structure.reports / (remove_suffixes(Path(selected_patient_name)) + ".pdf")

    @staticmethod
    def _open_report():
        """Opens the report of the selected patient."""
        path = PatientTab._get_report_path().resolve()
        if path is not None and path.is_file():
            try:
                system = platform.system()
                if system == "Windows":
                    subprocess.run(["cmd.exe", "/c", str(path)], shell=True)

                elif system == "Darwin":  # macOS
                    subprocess.Popen(["open", path.as_posix()],
                                     stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)

                else:  # Linux and others
                    # We use Popen with start_new_session=True to "detach" it from Python
                    # and redirect stderr to DEVNULL to hide those GTK/libgssapi warnings.
                    subprocess.Popen(["xdg-open", path.as_posix()],
                                     stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL,
                                     start_new_session=True)
            except Exception as e:
                print(f"Failed to open file: {e}")

    def _handle_progress_changed(self):
        """Updates the load state manager when a study is selected."""
        new_state = AnnotationState(self._patient_progress.currentIndex)
        if LOAD_STATE_MANAGER.annotation_progress != new_state:
            LOAD_STATE_MANAGER.annotation_progress = new_state
            # LOAD_STATE_MANAGER.selected_patient_data_modified = True

    def _handle_comment_changed(self):
        """Updates the load state manager when a study is selected."""
        scene_graph = LOAD_STATE_MANAGER.scene_graph
        if scene_graph is None:
            print("comment_changed: No scene graph loaded")
            return

        new_state = self._comment_text.toPlainText()
        if scene_graph.comment != new_state:
            scene_graph.comment = new_state
            # LOAD_STATE_MANAGER.selected_patient_data_modified = True
