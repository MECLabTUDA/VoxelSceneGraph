from __future__ import annotations

try:
    import sqlite_utils
except ImportError:
    import slicer

    slicer.util.pip_install("sqlite-utils")
    import sqlite_utils

import datetime
import json
import sqlite3
import sys
from __main__ import qt
from pathlib import Path
from typing import Optional, Callable

import vtkSegmentationCore
import slicer
import vtk
from PythonQt.QtCore import Qt
from PythonQt.QtGui import QHeaderView
from slicer import vtkMRMLScalarVolumeNode, vtkMRMLSegmentationNode
from slicer.ScriptedLoadableModule import *
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.parameterNodeWrapper import parameterNodeWrapper
from slicer.util import VTKObservationMixin

from annotation_database.functions import study_get_all, progress_update_to_study, progress_get_for_study
from annotation_database.options import Options
from annotation_database.structures import AnnotationState, Study, AnnotationProgress


def get_datadir() -> Path:
    """
    Returns a parent directory path
    where persistent application data can be stored.

    # linux: ~/.local/share
    # macOS: ~/Library/Application Support
    # windows: C:/Users/<USER>/AppData/Roaming
    """

    home = Path.home()
    if sys.platform == "win32":
        return home / "AppData/Roaming"
    elif sys.platform == "linux":
        return home / ".local/share"
    elif sys.platform == "darwin":
        return home / "Library/Application Support"
    else:
        return Path(".")


class AnnotationOverview(ScriptedLoadableModule):
    """
    Widget for easy patient list annotation and overview.
    Contains:
    - Global configuration zone with:
      - Path to the database
      - Name of the reader
      - Button to load studies
    - TabWidget with:
      - Combobox to select a study and a list of patients in a grid format
      - SegmentEditor tab
      - Data tab
      - Volume properties tab
    - Selected patient zone:
      - Comment text box
      - Combobox to change the current state of the annotation
      - Save button
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("Annotation Overview")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Segmentation")]
        self.parent.dependencies = []
        self.parent.contributors = ["Antoine P. Sanner (University Hospital Mayence, Germany)"]
        # TODO fix this text...
        self.parent.helpText = _("""Easily load and annotate patients for a study.
        Shortcuts:
        -s: save the progress for the selected patient (if any)
        -e: select the previous patient (if a patient is already selected)
        -r: select the previous patient (if a patient is already selected)
        -c: if there is a segment under the cursor, select it as the current segment in the segment editor
        """)
        self.parent.acknowledgementText = _("")


@parameterNodeWrapper
class AnnotationOverviewParameterNode:
    """
    The parameters needed by the module.
    Represents a selected patient (i.e. a volume and a segmentation).
    Patient name and study can be retrieved from widgets.
    """
    # Has to use "Optional" instead of "| None" because of slicer typing issues for serialization
    volume: Optional[vtkMRMLScalarVolumeNode]
    segmentation: Optional[vtkMRMLSegmentationNode]


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
                return self.study_progress[index.row()][1].progress.to_str()
            elif index.column() == 2:
                return self.study_progress[index.row()][1].when.strftime("%Y-%m-%d\n%H:%M:%S")
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


class AnnotationOverviewWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """
    Uses ScriptedLoadableModuleWidget base class, available at:
    Widget for easy patient list annotation and overview.
    Contains:
    - Global configuration zone with:
      - Path to the database
      - Name of the reader
      - Button to load studies
    - TabWidget with:
      - Combobox to select a study and a list of patients in a grid format
      - SegmentEditor tab
      - Data tab
      - Volume properties tab
    - Selected patient zone:
      - Comment text box
      - Combobox to change the current state of the annotation
      - Save button
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self._parameter_node: AnnotationOverviewParameterNode | None = None
        self._parameter_node_gui_tag = None

        self._config_save_path = get_datadir() / "slicer_annotation_overview.json"
        self._studies: list[Study] = []
        self._last_selected_study_index = 0  # We need to maintain an index to be able to cancel study selections
        self._selected_patient_name = ""
        self._selected_patient_modified = False
        self._current_windowing_index = 0
        self._windows = [(0, 100), (20, 60)]

        self._setting_patient_nodes = False  # Flag to avoid feedback loop
        self._cancel_patient_selection = False  # Flag to avoid feedback loop
        self._shortcuts = []  # List of active shortcuts

        self._db_string_text = qt.QLineEdit()
        self._reader_text = qt.QLineEdit()
        self._load_study_button = qt.QPushButton("Load studies")

        self._studies_combobox = qt.QComboBox()
        self._study_refresh_button = qt.QPushButton()

        self._tools_content = qt.QTabWidget()
        self._studies_tab = qt.QWidget()
        self._patients_table = slicer.qMRMLTableView()
        self._patients_table_model = StudyTableModel(None)
        self._segment_editor = slicer.modules.segmenteditor.widgetRepresentation()
        self._volumes_module = slicer.modules.volumes.widgetRepresentation()

        self._selected_patient_collapsible = slicer.qMRMLCollapsibleButton()
        self._selected_patient_comment_text = qt.QTextEdit()
        self._selected_patient_progress = qt.QComboBox()
        self._selected_patient_save_button = qt.QPushButton("Save")

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()
        self.setupUI()
        self._install_keyboard_shortcuts()

    def setupUI(self):
        # Make some room
        # slicer.util.setModuleHelpSectionVisible(False)

        # ==============================================================================================================
        # General config
        general_config_widget = slicer.qMRMLCollapsibleButton()
        self.layout.addWidget(general_config_widget)
        general_config_widget.setText("General Configuration")

        general_config_layout = qt.QFormLayout()
        general_config_widget.setLayout(general_config_layout)
        general_config_layout.addRow(qt.QLabel("Database path:"), self._db_string_text)
        general_config_layout.addRow(qt.QLabel("Reader:"), self._reader_text)

        # Try to load global config from disk
        if self._config_save_path.exists():
            try:
                with self._config_save_path.open("r") as f:
                    options = json.load(f)
                self._db_string_text.setText(options.get("db_string", ""))
                self._reader_text.setText(options.get("reader", ""))
            except json.JSONDecodeError:
                ...

        general_config_layout.addRow(self._load_study_button)
        self._load_study_button.clicked.connect(self.load_studies)

        # ==============================================================================================================
        # Studies lists and widgets
        tools_widget = slicer.qMRMLCollapsibleButton()
        self.layout.addWidget(tools_widget)
        tools_widget.setText("Tools")
        tools_layout = qt.QVBoxLayout()
        tools_widget.setLayout(tools_layout)
        tools_layout.addWidget(self._tools_content)

        self._studies_tab = qt.QWidget()
        studies_layout = qt.QFormLayout()
        self._studies_tab.setLayout(studies_layout)

        # Intermediate widget with combobox and refresh button
        self._study_refresh_button.setIcon(qt.QApplication.style().standardIcon(qt.QStyle.SP_BrowserReload))
        self._study_refresh_button.clicked.connect(self.update_study_progress_from_db)

        study_interm_widget = qt.QWidget()
        study_interm_layout = qt.QHBoxLayout()
        study_interm_layout.setContentsMargins(0, 0, 0, 0)
        study_interm_widget.setLayout(study_interm_layout)
        study_interm_layout.addWidget(self._studies_combobox, 1)
        study_interm_layout.addWidget(self._study_refresh_button, 0)

        studies_layout.addRow(qt.QLabel("Study:"), study_interm_widget)

        self._studies_combobox.currentIndexChanged.connect(self.study_selected)
        self._studies_combobox.setEnabled(False)
        self._study_refresh_button.setEnabled(False)

        studies_layout.addRow(self._patients_table)
        self._patients_table.setModel(self._patients_table_model)
        self._setup_table_format()
        self._patients_table.selectionChanged.connect(self.patient_selected)

        self.setup_tabs()

        self._tools_content.currentChanged.connect(self.selected_tab_changed)

        # ==============================================================================================================
        # Selected patient
        self.layout.addWidget(self._selected_patient_collapsible)
        self._selected_patient_collapsible.setText("Patient")
        selected_patient_layout = qt.QFormLayout()
        self._selected_patient_collapsible.setLayout(selected_patient_layout)

        selected_patient_layout.addRow(qt.QLabel("Comment:"), self._selected_patient_comment_text)
        self._selected_patient_comment_text.setFixedHeight(100)
        self._selected_patient_comment_text.setEnabled(False)
        self._selected_patient_comment_text.textChanged.connect(self._mark_patient_as_modified)

        selected_patient_layout.addRow(qt.QLabel("Progress:"), self._selected_patient_progress)
        for state in AnnotationState:
            self._selected_patient_progress.addItem(state.to_str().replace("\n", " "))
        self._selected_patient_progress.setEnabled(False)
        self._selected_patient_progress.currentIndexChanged.connect(self._mark_patient_as_modified)

        selected_patient_layout.addRow(self._selected_patient_save_button)
        self._selected_patient_save_button.clicked.connect(self.save_progress)
        self._selected_patient_save_button.setEnabled(False)
        self._selected_patient_save_button.setShortcut(qt.QKeySequence("s"))

        # ==============================================================================================================
        # Add some small stretch to allow the data probe widget to expand without creating a scrollbar
        # when hovering over a segment with the cursor
        # Note: the tools QTabWidget has a minimum height set from one of the tools widget that we add
        self.layout.addStretch(1)

    def setup_tabs(self):
        self._tools_content.clear()
        self._tools_content.addTab(self._studies_tab, "Studies")
        # Cannot create a new widget, but we can set nodes globally
        self._tools_content.addTab(slicer.modules.data.widgetRepresentation(), "Data")
        # Could create a new widget, but we can set nodes globally
        self._tools_content.addTab(self._segment_editor, "Segment Editor")
        # Cannot create a new widget, but we can set nodes globally
        self._tools_content.addTab(self._volumes_module, "Volume Properties")

    def selected_tab_changed(self):
        """Check whether we have anything specific to do."""
        if self._tools_content.currentIndex == 2:
            # Segment editor
            self._segment_editor.self().editor.installKeyboardShortcuts()
        else:
            self._segment_editor.self().editor.uninstallKeyboardShortcuts()

    def load_studies(self) -> None:
        """Access the db, store config if it worked, load studies and populate widgets."""
        Options.set_db_string(self._db_string_text.text)
        try:
            Options.get_db()
            print(f"Connected to: {self._db_string_text.text}")
            # Save options
            with open(self._config_save_path, "w") as f:
                json.dump({
                    "db_string": self._db_string_text.text,
                    "reader": self._reader_text.text
                }, f)
        except sqlite3.OperationalError as e:
            slicer.util.confirmOkCancelDisplay(
                f"Could not access database: {e}",
                windowTitle="Database unreachable", parent=None
            )
            return

        # Load studies and populate the combobox
        self._studies = list(study_get_all().values())
        self._studies_combobox.clear()
        self._studies_combobox.addItem("")
        for study in self._studies:
            self._studies_combobox.addItem(study.name)
        self._studies_combobox.setEnabled(True)

        self.reset_patient_selection()

    def reset_patient_selection(self):
        """Reset stuff related to having a selected patient."""

        self._selected_patient_name = ""
        # Whether to ignore the next patient selection trigger (because we did some internal update)
        self._cancel_patient_selection = False
        self._current_windowing_index = 0

        # We actually don't need to update the model
        # because we can keep the table as is, even if some loaded nodes are deleted
        # self._patients_table_model = StudyTableModel(None)
        # self._patients_table.setModel(self._patients_table_model)

        if self._parameter_node is not None:
            self._setting_patient_nodes = True
            self._parameter_node.setValue("volume", None)
            self._parameter_node.setValue("segmentation", None)
            self._setting_patient_nodes = False

        self._selected_patient_collapsible.setText("Patient")
        self._selected_patient_comment_text.setText("")
        self._selected_patient_comment_text.setEnabled(False)
        self._selected_patient_progress.setEnabled(False)
        self._selected_patient_save_button.setEnabled(False)
        self._patients_table.clearSelection()

        # We need to do that after resetting the comment text box
        self._selected_patient_modified = False

    def remove_nodes(self):
        """Remove any existing nodes in the parameter node from the scene."""
        # Unset parameter node (if it exists)
        if self._parameter_node is not None:
            self._setting_patient_nodes = True
            if self._parameter_node.volume is not None:
                slicer.mrmlScene.RemoveNode(self._parameter_node.volume)
            if self._parameter_node.segmentation is not None:
                slicer.mrmlScene.RemoveNode(self._parameter_node.segmentation)
            self._setting_patient_nodes = False

    def study_selected(self):
        """Load progress from annotation_database and update table model."""
        # Detect no-op feedback loops
        if self._studies_combobox.currentIndex == self._last_selected_study_index:
            return

        # We have to deselect the current patient when changing study,
        # because the save path is dependent on the current study
        if self._selected_patient_modified:
            # Ask for confirmation
            cancel = not self.confirm_discard_changes("study")
            if cancel:
                self._studies_combobox.setCurrentIndex(self._last_selected_study_index)
                return

        # Unset parameter node (if it exists)
        self.remove_nodes()
        self.reset_patient_selection()

        if self._studies_combobox.currentIndex <= 0:
            self._patients_table.setModel(StudyTableModel(None))
            self._study_refresh_button.setEnabled(False)
        else:
            selected_study = self._studies[self._studies_combobox.currentIndex - 1]
            selected_study.progress = progress_get_for_study(selected_study.id)
            self._patients_table_model = StudyTableModel(selected_study, self.select_row_with_current_patient)
            self._patients_table.setModel(self._patients_table_model)
            self._study_refresh_button.setEnabled(True)
        # Somehow we have to do this again after changing the model
        self._setup_table_format()
        self._last_selected_study_index = self._studies_combobox.currentIndex
        # We also have to reset the scroll bar location to the top
        self._patients_table.scrollTo(self._patients_table_model.index(0, 0))

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

    @staticmethod
    def confirm_discard_changes(thing_selected: str = "patient") -> bool:
        """Opens a dialog asking whether we want to discard changes to a modified patient."""
        return slicer.util.confirmOkCancelDisplay(
            "Some modifications to the current progress are not saved. "
            f"Selecting a new {thing_selected} will erase it. Do you want to continue?",
            windowTitle="Unsaved progress", parent=None
        )

    def patient_selected(self):
        """Load data from hard drive and do UI updates."""
        if self._cancel_patient_selection:
            self._cancel_patient_selection = False
            return

        if len(self._patients_table.selectionModel().selectedRows()) == 0:
            return
        row_idx = self._patients_table.selectionModel().selectedRows()[0].row()

        # Add safety mechanism using flag before changing patient
        if self._selected_patient_modified:
            # Ask for confirmation
            cancel = not self.confirm_discard_changes("patient")
            if cancel:
                self.select_row_with_current_patient()
                return

        patient_name, progress = self._patients_table_model.study_progress[row_idx]  # type: str, AnnotationProgress
        selected_study = self._studies[self._studies_combobox.currentIndex - 1]
        vol_path = Path(Options.get_db_string()).parent / selected_study.img_folder / patient_name
        seg_path = Path(Options.get_db_string()).parent / selected_study.label_folder / patient_name

        # Check that everything is here and can be loaded
        fail = not (vol_path.is_file() and seg_path.is_file())
        try:
            vol_node = slicer.util.loadVolume(vol_path.as_posix())
            seg_node = slicer.util.loadSegmentation(seg_path.as_posix())
        except RuntimeError:
            fail = True

        if fail:
            slicer.util.confirmOkCancelDisplay(
                f"Patient {patient_name} is missing its image or volume or both or they cannot be loaded.",
                windowTitle="Patient not found", parent=None
            )
            self.select_row_with_current_patient()
            return

        # Unset parameter node (if it exists)
        self.remove_nodes()

        # Set parameter_node
        self._setting_patient_nodes = True
        self._parameter_node.setValue("volume", vol_node)
        self._parameter_node.setValue("segmentation", seg_node)
        self._setting_patient_nodes = False

        # ==============================================================================================================
        import time
        t = time.time()
        # Normalize segment names based on study before we set observers
        segmentation = seg_node.GetSegmentation()
        # Removing and then adding segments is extremely expensive
        # So to avoid triggers, we replace the observed segmentation in the scene node with a dummy segmentation
        # FIXME: ideally, we want to support semantic segmentation and label maps fully
        tmp_segmentation = vtkSegmentationCore.vtkSegmentation()
        seg_node.SetAndObserveSegmentation(tmp_segmentation)

        # Get all segments and their ids
        existing_segments = {}
        for idx in range(segmentation.GetNumberOfSegments()):
            segment = segmentation.GetNthSegment(idx)
            existing_segments[segment.GetLabelValue()] = segment

        # Then remove all segments from the segmentation
        for segment in existing_segments.values():
            segmentation.RemoveSegment(segment)
        # Then iterate over expected segments in the study and add segments back
        for idx, segment_name in enumerate(selected_study.segments, start=1):
            if idx in existing_segments:
                segment = existing_segments[idx]
                segmentation.AddSegment(segment)
            else:
                segmentation.AddEmptySegment()
                segment = segmentation.GetNthSegment(segmentation.GetNumberOfSegments() - 1)
            segment.SetName(segment_name)
        # Then handle segments that are repeated last segment
        if selected_study.last_segment_can_repeat:
            keys_to_handle = sorted([key for key in existing_segments if key > len(selected_study.segments)])
            counter = 2
            for key in keys_to_handle:
                segment = existing_segments[key]
                segmentation.AddSegment(segment)
                segment.SetName(f"{selected_study.segments[-1]} {counter}")
                counter += 1
        # Finally reset the color
        colormap = slicer.mrmlScene.GetNodeByID(slicer.modules.colors.logic().GetDefaultLabelMapColorNodeID())
        buffer = [0, 0, 0, 0]
        for idx in range(segmentation.GetNumberOfSegments()):
            colormap.GetColor(idx + 1, buffer)
            segmentation.GetNthSegment(idx).SetColor(*buffer[:-1])
        seg_node.SetAndObserveSegmentation(segmentation)
        del tmp_segmentation
        # ==============================================================================================================

        # Do the processing of the volume/segmentation pair
        # We need to do that after the
        self._process_node_pair()

        # Set window width / length if specified
        if selected_study.window_width is not None and selected_study.window_length is not None:
            # min_v = selected_study.window_length - selected_study.window_width / 2
            # max_v = selected_study.window_length + selected_study.window_width / 2
            vol_node.GetDisplayNode().SetAutoWindowLevel(False)
            # vol_node.GetDisplayNode().SetWindowLevelMinMax(min_v, max_v)
            vol_node.GetDisplayNode().SetWindowLevelMinMax(self._windows[0][0], self._windows[0][1])

        # Change selected node for other modules
        self._segment_editor.self().editor.setSegmentationNode(seg_node)
        self._segment_editor.self().editor.setSourceVolumeNode(vol_node)
        if segmentation.GetNumberOfSegments() > 0:
            self._segment_editor.self().editor.setCurrentSegmentID(segmentation.GetNthSegmentID(0))
        self._volumes_module.setEditedNode(vol_node)

        # Enable widgets on selecting patient + progress and comment update
        self._selected_patient_collapsible.setText(f"Patient {patient_name}")
        self._selected_patient_comment_text.setText(progress.comment)
        self._selected_patient_comment_text.setEnabled(True)
        self._selected_patient_progress.setCurrentIndex(progress.progress.value)
        self._selected_patient_progress.setEnabled(True)
        self._selected_patient_save_button.setEnabled(True)

        # Update selected patient name
        self._selected_patient_name = patient_name
        self._selected_patient_modified = False

    def select_row_with_current_patient(self):
        """Update the table's currentIndex to the selected patient (where ever it is)"""
        if not self._selected_patient_name:
            self._patients_table.clearSelection()
            return

        # Reselect previous patient
        for idx, (patient_name, _) in enumerate(self._patients_table_model.study_progress):
            if patient_name == self._selected_patient_name:
                # Note: this might get triggered during any sort, i.e. also when we're browsing another study.
                #       In this case, we must only cancel the patient selection if we actually find the patient...
                self._cancel_patient_selection = True
                self._patients_table.setCurrentIndex(self._patients_table_model.index(idx, 0))
                break

    def update_study_progress_from_db(self):
        """Update the progress for a selected study by reading from the database. Any unsaved changes are discarded."""
        # Add safety mechanism using flag before changing patient
        if self._selected_patient_modified:
            # Ask for confirmation
            cancel = not self.confirm_discard_changes("study")
            if cancel:
                self.select_row_with_current_patient()
                return

        selected_study = self._studies[self._studies_combobox.currentIndex - 1]
        selected_study.progress = progress_get_for_study(selected_study.id)
        self.reset_patient_selection()
        self._patients_table_model = StudyTableModel(selected_study, self.select_row_with_current_patient)
        self._patients_table.setModel(self._patients_table_model)

    def save_progress(self):
        """Save progress for the current patient. Checks are made for a safe usage."""
        if (self._studies_combobox.currentIndex == 0 or
                self._parameter_node is None or
                self._parameter_node.volume is None or
                self._parameter_node.segmentation is None):
            return

        selected_study = self._studies[self._studies_combobox.currentIndex - 1]

        # Save to hard drive
        if self._parameter_node.segmentation is not None:
            seg_path = Path(Options.get_db_string()).parent / selected_study.label_folder / self._selected_patient_name
            tmp_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
            segmentation = self._parameter_node.segmentation.GetSegmentation()
            slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(
                self._parameter_node.segmentation,
                [segmentation.GetNthSegmentID(idx) for idx in range(segmentation.GetNumberOfSegments())],
                tmp_node,
                self._parameter_node.volume
            )
            fail = not slicer.util.saveNode(tmp_node, seg_path.as_posix())
            # Abort if there was an issue
            if fail:
                slicer.util.confirmOkCancelDisplay(
                    f"Could not save the segmentation for patient {self._selected_patient_name}."
                    f"Please try to save it manually somewhere else.",
                    windowTitle="Error", parent=None
                )
                return

            slicer.mrmlScene.RemoveNode(tmp_node)

        # Save to database
        new_progress = AnnotationProgress(
            self._reader_text.text,
            AnnotationState(self._selected_patient_progress.currentIndex),
            datetime.datetime.now(),
            self._selected_patient_comment_text.toPlainText()
        )
        progress_update_to_study(selected_study.id, {self._selected_patient_name: new_progress})

        # Update corresponding table row (it might have changed because of a sort operation)
        for idx, (patient_name, progress) in enumerate(self._patients_table_model.study_progress):
            if patient_name == self._selected_patient_name:
                self._patients_table_model.study_progress[idx] = (patient_name, new_progress)
                self._patients_table_model.dataChanged.emit(
                    self._patients_table_model.index(idx, 0),
                    self._patients_table_model.index(idx, self._patients_table_model.columnCount(0))
                )
                break

        self._selected_patient_modified = False

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # The user might have visited the data, segment editor, volumes module, which might have reset the layout
        self.setup_tabs()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.
        param_node = slicer.mrmlScene.GetSingletonNode(self.moduleName, "vtkMRMLScriptedModuleNode")
        if param_node is None:
            param_node = slicer.mrmlScene.CreateNodeByClass("vtkMRMLScriptedModuleNode")
            param_node.UnRegister(None)  # object is owned by the Python variable now
            param_node.SetSingletonTag(self.moduleName)
            # Add module name in an attribute to allow filtering in node selector widgets
            # Note that SetModuleName is not used anymore as it would be redundant with the ModuleName attribute.
            param_node.SetAttribute("ModuleName", self.moduleName)
            param_node.SetName(slicer.mrmlScene.GenerateUniqueName(self.moduleName))
            slicer.mrmlScene.AddNode(param_node)

        self.setParameterNode(AnnotationOverviewParameterNode(param_node))
        self.reset_patient_selection()

    def setParameterNode(self, inputParameterNode: AnnotationOverviewParameterNode | None) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self._parameter_node:
            self._parameter_node.disconnectGui(self._parameter_node_gui_tag)
            self.removeObserver(self._parameter_node, vtk.vtkCommand.ModifiedEvent, self._check_both_nodes_exist)

        self._parameter_node = inputParameterNode

        if self._parameter_node is not None:
            # Add observation that both nodes should always exist
            self.addObserver(self._parameter_node, vtk.vtkCommand.ModifiedEvent, self._check_both_nodes_exist)
            self._check_both_nodes_exist()

        self._process_node_pair()

    def _process_node_pair(self):
        """
        Add observers to the volume+seg in the parameter node.
        Also add the volume as the master node
        """
        if self._parameter_node is None:
            return

        # Add observation on segmentation
        if self._parameter_node.segmentation is not None:
            self._parameter_node.segmentation.AddObserver(
                self._parameter_node.segmentation.GetSegmentation().SegmentAdded,
                self._rename_new_segment
            )
            self._parameter_node.segmentation.AddObserver(
                self._parameter_node.segmentation.GetSegmentation().SegmentModified,
                self._mark_patient_as_modified
            )
            self._parameter_node.segmentation.AddObserver(
                self._parameter_node.segmentation.GetSegmentation().SegmentRemoved,
                self._mark_patient_as_modified
            )

            # Set volume as master of the segmentation
            if self._parameter_node.volume is not None:
                self._parameter_node.segmentation.SetReferenceImageGeometryParameterFromVolumeNode(
                    self._parameter_node.volume
                )

    def _rename_new_segment(self, caller=None, event=None):
        """
        If the last segment can repeat, it's likely that new segments will be added during the annotation.
        So we need to update its name and color automatically.
        Note: we currently don't rename the segments when one gets deleted.
        """
        if self._studies_combobox.currentIndex <= 0:
            # Segmentation is still there but we changed study and don't want to do anything
            return

        selected_study = self._studies[self._studies_combobox.currentIndex - 1]
        if not selected_study.last_segment_can_repeat:
            return

        segmentation = self._parameter_node.segmentation.GetSegmentation()
        n_segments = segmentation.GetNumberOfSegments()
        segment = segmentation.GetNthSegment(n_segments - 1)
        segment.SetName(f"{selected_study.segments[-1]} {n_segments - len(selected_study.segments) + 1}")

        # Finally reset the color
        colormap = slicer.mrmlScene.GetNodeByID(slicer.modules.colors.logic().GetDefaultLabelMapColorNodeID())
        buffer = [0, 0, 0, 0]
        colormap.GetColor(n_segments, buffer)
        segment.SetColor(*buffer[:-1])

    def _check_both_nodes_exist(self, caller=None, event=None) -> None:
        """If not, we consider that the patient has been unselected."""
        if self._setting_patient_nodes:
            return

        # We need to disable the save button when we are missing either node
        if self._parameter_node and (self._parameter_node.volume is None or self._parameter_node.segmentation is None):
            self.reset_patient_selection()

    def _mark_patient_as_modified(self, caller=None, event=None):
        self._selected_patient_modified = True

    def _select_patient_relative_to_current(self, offset: int):
        """Select a new patient relative to the currently selected one. Loop if we reach the end of the list."""
        if not self._selected_patient_name:
            # Abort on no patient selected
            return
        # The safety features are handled by the method handling the patient loading anyway
        self._patients_table.setCurrentIndex(
            self._patients_table_model.index(
                (self._patients_table.currentIndex().row() + offset + self._patients_table_model.rowCount(0))
                % self._patients_table_model.rowCount()
                , 0
            )
        )

    def _select_segment_under_cursor(self):
        """
        Try to find out if there is a segment under the cursor.
        If it belongs to the current segmentation, select this segment as the current segment in the segment editor.
        """
        if self._parameter_node is None or self._parameter_node.segmentation is None:
            # No current segmentation
            return

        # Find out the position of the cursor
        crosshair_node = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLCrosshairNode")
        if crosshair_node is None:
            return

        ras_pos = [0, 0, 0]
        inside_view = crosshair_node.GetCursorPositionRAS(ras_pos)
        xyz_pos = [0, 0, 0]
        slice_node = crosshair_node.GetCursorPositionXYZ(xyz_pos)
        if not inside_view or not slice_node:
            return

        slice_widget = slicer.app.layoutManager().sliceWidget(slice_node.GetName())
        seg_disp_manager = slice_widget.sliceView().displayableManagerByClassName(
            "vtkMRMLSegmentationsDisplayableManager2D"
        )

        # Get segment ids at RAS position
        segmentation_node = self._parameter_node.segmentation
        segment_ids = vtk.vtkStringArray()
        seg_disp_manager.GetVisibleSegmentsForPosition(
            ras_pos, segmentation_node.GetDisplayNode(), segment_ids
        )

        # Set the first segment found as the selected segment
        if segment_ids.GetNumberOfValues() > 0:
            self._segment_editor.self().editor.setCurrentSegmentID(segment_ids.GetValue(0))

    def _change_windowing(self):
        if self._parameter_node is None or self._parameter_node.volume is None:
            return
        self._current_windowing_index = (self._current_windowing_index + 1) % len(self._windows)
        self._parameter_node.volume.GetDisplayNode().SetWindowLevelMinMax(
            self._windows[self._current_windowing_index][0], self._windows[self._current_windowing_index][1]
        )

    def _install_keyboard_shortcuts(self):
        """Install some additional keybind shortcuts. These are always active when the module is open."""
        if not self._shortcuts:
            # Select previous patient
            prev_patient_shortcut = qt.QShortcut(self.parent)
            prev_patient_shortcut.setKey(qt.QKeySequence("e"))
            prev_patient_shortcut.connect("activated()", lambda: self._select_patient_relative_to_current(-1))
            self._shortcuts.append(prev_patient_shortcut)

            # Select next patient
            next_patient_shortcut = qt.QShortcut(self.parent)
            next_patient_shortcut.setKey(qt.QKeySequence("r"))
            next_patient_shortcut.connect("activated()", lambda: self._select_patient_relative_to_current(1))
            self._shortcuts.append(next_patient_shortcut)

            # Switch the value windowing for the volume
            switch_windowing_shortcut = qt.QShortcut(self.parent)
            switch_windowing_shortcut.setKey(qt.QKeySequence("v"))
            switch_windowing_shortcut.connect("activated()", self._change_windowing)
            self._shortcuts.append(switch_windowing_shortcut)

            # Select segment under the cursor
            select_segment_cursor_shortcut = qt.QShortcut(self.parent)
            select_segment_cursor_shortcut.setKey(qt.QKeySequence("c"))
            select_segment_cursor_shortcut.connect("activated()", self._select_segment_under_cursor)
            self._shortcuts.append(select_segment_cursor_shortcut)
