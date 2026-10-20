from typing import Callable

import qt
import slicer
from PythonQt.QtCore import Qt
from PythonQt.QtGui import QHeaderView

from scene_graph_database.structures import Study, AnnotationProgress
from ..load_state import LOAD_STATE_MANAGER
from ..utils import confirm_discard_changes


class StudyTableModel(qt.QAbstractTableModel):
    """TableModel for displaying the progress in a study."""

    def __init__(self, study: Study | None, sort_callback: Callable | None = None):
        """
        :param study: Study object with all the data
        :param sort_callback: callback for updating the currentIndex after sorting
        """
        super().__init__()
        self.study_progress: list[tuple[str, AnnotationProgress]] = [
            (patient_name, study.progress[patient_name])
            for patient_name in sorted(study.progress.keys())
        ] if study is not None else []
        self.sort_callback = sort_callback

    def data(self, index, role):
        if role == Qt.DisplayRole:
            if index.column() == 0:
                return self.study_progress[index.row()][0]
            elif index.column() == 1:
                return self.study_progress[index.row()][1].progress.to_str().replace("\n", " ")
            elif index.column() == 2:
                return self.study_progress[index.row()][1].when.strftime("%Y-%m-%d %H:%M:%S")
            elif index.column() == 3:
                return self.study_progress[index.row()][1].reader
        elif role == Qt.TextAlignmentRole:
            return Qt.AlignCenter

    def rowCount(self, index=0):
        return len(self.study_progress)

    def columnCount(self, index=0):
        return 4

    def headerData(self, section, orientation, role):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return ["Patient Name", "Status", "Date", "Reader"][section]

    def sort(self, column, order=0):
        reverse = order == 1  # DescendingOrder
        if column == 0:
            key = lambda x: x[0]
        elif column == 1:
            key = lambda x: x[1].progress
        elif column == 2:
            key = lambda x: x[1].when
        elif column == 3:
            key = lambda x: x[1].reader
        else:
            raise NotImplementedError
        self.study_progress = sorted(self.study_progress, key=key, reverse=reverse)
        self.layoutChanged.emit()
        if self.sort_callback is not None:
            self.sort_callback()


class PatientProgressTable(qt.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._patients_table = slicer.qMRMLTableView()
        self._model = StudyTableModel(None)  # Just init for typehint

        self.set_study(None)
        self._setup_table_format()

        # Bind to relevant signals
        LOAD_STATE_MANAGER.study_selected.connect(lambda: self.set_study(LOAD_STATE_MANAGER.selected_study))

        # If a loading failed, the selected patient is cleared. So update table selection accordingly.
        LOAD_STATE_MANAGER.patient_selected.connect(
            lambda: self.clear_selection() if LOAD_STATE_MANAGER.selected_patient_name is None else None
        )
        # Enable table after loading (or failing to do so) of a new patient
        LOAD_STATE_MANAGER.patient_selected.connect(
            lambda: self.setEnabled(True)
        )

        self._init_ui()

    def _init_ui(self):
        # Just wrap the table
        layout = qt.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._patients_table)
        self._patients_table.selectionChanged.connect(self._handle_selection_changed)
        self.setLayout(layout)

    def _setup_table_format(self):
        """Reset the UI parameters for the patient table."""
        self._patients_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._patients_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self._patients_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self._patients_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self._patients_table.setSelectionBehavior(QHeaderView.SelectRows)
        self._patients_table.setSelectionMode(QHeaderView.SingleSelection)
        self._patients_table.setSortingEnabled(True)
        self._patients_table.sortByColumn(0, Qt.AscendingOrder)

    def _handle_selection_changed(self):
        """Handle the selectionChanged signal from the table and decide whether we should notify others."""
        # Empty table
        if len(self._patients_table.selectionModel().selectedRows()) == 0:
            return

        row_idx = self._patients_table.selectionModel().selectedRows()[0].row()
        patient_name, progress = self._model.study_progress[row_idx]  # type: str, AnnotationProgress

        # We are selecting the same patient again, so we don't need to notify the outside world
        selected_patient_name = LOAD_STATE_MANAGER.selected_patient_name
        if patient_name == selected_patient_name:
            return

        # Disable the table until we are done loading the patient to avoid having multiple load states at the same time
        self.setEnabled(False)
        # Check for any confirmation
        if LOAD_STATE_MANAGER.selected_patient_scene_graph_modified:
            confirmed = confirm_discard_changes("Loading a new patient")
            if not confirmed:
                # Reselect the previous patient
                self.select_row_with_current_patient()
                self.setEnabled(True)
                return

        LOAD_STATE_MANAGER.selected_patient_name = patient_name

    def set_study(self, study: Study | None):
        """Set the study to display in the patient table."""
        # Takes care of selecting the right row, after we sort
        self._model = StudyTableModel(study, self.select_row_with_current_patient)
        self._patients_table.setModel(self._model)
        # Somehow we have to do this again after changing the model
        self._setup_table_format()
        self.clear_selection()
        # We also have to reset the scroll bar location to the top
        self._patients_table.scrollTo(self._model.index(0, 0))

    def select_row_with_current_patient(self):
        """Update the table's currentIndex to the selected patient (where ever it is)"""
        selected_patient_name = LOAD_STATE_MANAGER.selected_patient_name

        if selected_patient_name is None:
            self._patients_table.clearSelection()
            return

        # Reselect previous patient
        for idx, (patient_name, _) in enumerate(self._model.study_progress):
            if patient_name == selected_patient_name:
                self._patients_table.setCurrentIndex(self._model.index(idx, 0))
                break

    def update_selected_row(self):
        """
        Update the table's row for the selected patient (where ever it is).
        E.g. because we saved the progress for the selected patient and need to update the table row.
        Note: we assume that we all share the same Study object.
        """
        for idx, (patient_name, progress) in enumerate(self._model.study_progress):
            if patient_name == LOAD_STATE_MANAGER.selected_patient_name:
                self._model.study_progress[idx] = (patient_name, LOAD_STATE_MANAGER.selected_study.progress[patient_name])
                self._model.dataChanged.emit(
                    self._model.index(idx, 0),
                    self._model.index(idx, self._model.columnCount(0))
                )
                break

    def select_patient_relative_to_current(self, offset: int):
        """Select a new patient relative to the currently selected one. Loop if we reach the end of the list."""
        if LOAD_STATE_MANAGER.selected_patient_name is None:
            # Abort on no patient selected
            print("select_patient_relative_to_current: No patient selected, cannot select a new patient")
            return
        # The safety features are handled by the method handling the patient loading anyway
        self._patients_table.setCurrentIndex(
            self._model.index(
                (self._patients_table.currentIndex().row() + offset + self._model.rowCount(0))
                % self._model.rowCount()
                , 0
            )
        )

    def clear_selection(self):
        """
        Clear the selection in the table.
        Note: we assume that the caller is the one who will handle unloading the volumes.
        """
        self._patients_table.clearSelection()
