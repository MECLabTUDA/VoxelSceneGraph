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

import logging
import re
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal, QRunnable, QThreadPool, QMutex
from PyQt6.QtGui import QIcon, QStandardItemModel, QStandardItem, QKeySequence, QShortcut
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSizePolicy, QGroupBox, QLineEdit, QListView, \
    QMessageBox, QPushButton
from lru import LRU

from scene_graph_annotation.knowledge import KnowledgeGraph
from scene_graph_annotation.logging_handlers import RecordDisplayHandler
from scene_graph_annotation.scene import SceneGraph
from scene_graph_annotation.ui_utils import QAlignRightButton
from scene_graph_annotation.utils import string_to_regex_pattern
from scene_graph_annotation.utils.asset_paths import padding_icon, reload_icon, communication_icon, checkmark_icon
from scene_graph_annotation.utils.image_utils import find_patients_to_annotate
from .PatientWidget import PatientWidget
from ...utils.ArrayView import ArrayView
from ...utils.image_utils import load_array_view
from ...utils.progress import CohortAnnotationProgress, Progress


class PatientSelectionWidget(QGroupBox):
    """
    Widget used to select the patient image that one wish to annotate.
    Clicking a row will make the scene graph to be read from file (and make computationally expensive preparations),
    unless the scene graph is already loaded in memory.
    Additionally, we track the annotation progress in a separate file in the scene graph folder.
    Components:
    - A name filter (Label "Search:" Line edit),
      filtering is done automatically at each text change (no filtering if filter is empty)
    - A list of patient names, with a reload button on each line to load the scene graph back from file if present
      Pressing "Enter" will select the patient.
      Pressing "Tab" will select the next patient (cycling when at the end).
      Pressing "Shift+Tab" will select the previous patient (cycling when at the start).
      Note: these shortcuts work globally, so be careful if reusing them in another widget.
    - A reload button to refresh the patient list

    When clicking on a list element, the computation required to prepare the widget to display may be long.
    So it's done on another thread. There are 2 signals (currently_loading and patient_selected)
    used to know what to display.
    Anti-spam for loading is included i.e. only one thread will ever attempt to load the same patient.
    The main reload button is disabled while any loading is being done.
    """

    # Signal used to say that the currently selected patient is being loaded and
    # that some placeholder need to be displayed with the patient's name
    currently_loading = pyqtSignal(str)

    loading_failed = pyqtSignal()
    # Signal used to emit the widget that needs to be displayed
    patient_selected = pyqtSignal(PatientWidget)

    # The loading may fail because the loaded scene graph is not valid
    # However, opening a QMessage box from a thread does not work
    # So another signal is required...
    _error_message_relay = pyqtSignal(RecordDisplayHandler)

    # Signal to bridge the gap between the SceneGraph that was created on another thread
    # and the display widget that needs to be created on the main thread
    # The tuple contains:
    # - the patient str
    # - the scene graph
    # - the nifti image of the image to be displayed
    # - the save path
    create_widget_for_scene_graph = pyqtSignal(tuple)

    # QRunnable is not a QObject and need to move the "finished" signal here
    dcr_thread_cnt = pyqtSignal()

    KB_SELECT_PATIENT_SHORTCUT = "Return"
    KB_SELECT_NEXT_PATIENT_SHORTCUT = "Tab"
    KB_SELECT_PREV_PATIENT_SHORTCUT = "Shift+Tab"

    def __init__(
            self,
            knowledge_graph: KnowledgeGraph,
            img_folder: Path,
            scene_graph_folder: Path,
    ):
        super().__init__("Patients")
        self.knowledge_graph = knowledge_graph
        self._img_folder = img_folder
        self._scene_graph_folder = scene_graph_folder
        # Pat str: (img path, ann path)
        pat_paths = find_patients_to_annotate(self.knowledge_graph, img_folder, scene_graph_folder)
        # Note: sort here, so we do it once (dicts are ordered now)
        self._patient_paths = {k: pat_paths[k] for k in sorted(pat_paths.keys())}
        self._progress = CohortAnnotationProgress.load(scene_graph_folder)

        # Dict used to store already loaded scene graphs
        # TODO add app config either with max number of widgets or memory budget
        #  (the latter requires writing code for memory usage computation)
        self._widget_by_patient_cache: LRU[str, PatientWidget] = LRU(10)
        self._patient_to_row_item: dict[str, QStandardItem] = {}  # Used to retrieve row for icon update
        self._currently_selected_patient: str = ""
        self._patients_being_loaded: set[str] = set()  # Avoid loading the same patient multiple time if user is dumb
        self._threads_running = 0
        self._threads_running_mtx = QMutex()

        # Widgets
        self._reload_icon = QIcon(reload_icon.as_posix())
        self._filter_lineedit = QLineEdit()
        self._list_view = QListView()
        self._list_model = QStandardItemModel(self._list_view)
        self._list_view.setModel(self._list_model)
        self._main_reload_button = QPushButton()

        # Add keyboard shortcuts
        self.select_shortcut = QShortcut(QKeySequence(self.KB_SELECT_PATIENT_SHORTCUT), self)
        self.select_shortcut.activated.connect(lambda: self._kb_select_patient(0))
        self.select_next_shortcut = QShortcut(QKeySequence(self.KB_SELECT_NEXT_PATIENT_SHORTCUT), self)
        self.select_next_shortcut.activated.connect(lambda: self._kb_select_patient(1))
        self.select_prev_shortcut = QShortcut(QKeySequence(self.KB_SELECT_PREV_PATIENT_SHORTCUT), self)
        self.select_prev_shortcut.activated.connect(lambda: self._kb_select_patient(-1))

        # Icon constants
        # self._file_icon = QIcon(file_icon.as_posix())
        self._padding_icon = QIcon(padding_icon.as_posix())
        self._pending_review_icon = QIcon(communication_icon.as_posix())
        self._finished_icon = QIcon(checkmark_icon.as_posix())
        # self._save_icon = QIcon(save_icon.as_posix())

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Filter / search
        filter_widget = QWidget()
        layout.addWidget(filter_widget)
        filter_layout = QHBoxLayout()
        filter_widget.setLayout(filter_layout)
        filter_layout.setContentsMargins(0, 0, 0, 0)

        filter_label = QLabel("Search:")
        filter_layout.addWidget(filter_label, alignment=Qt.AlignmentFlag.AlignLeft)
        filter_label.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)

        filter_layout.addWidget(self._filter_lineedit)
        self._filter_lineedit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self._filter_lineedit.textChanged.connect(self._reset_list_content)

        # List view
        layout.addWidget(self._list_view)
        self._list_view.clicked.connect(
            lambda index: self._get_or_compute_patient_widget(self._list_model.itemFromIndex(index).data())
        )

        # Reload button
        layout.addWidget(self._main_reload_button)
        self._main_reload_button.setText("Reload patient list")
        self._main_reload_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._main_reload_button.clicked.connect(self._reload_patient_list)

        # Dirty signal instead of "finished" in the QRunnable
        self.dcr_thread_cnt.connect(self._dcr_thread_cnt)
        self.create_widget_for_scene_graph.connect(lambda tup: self._create_widget_for_scene_graph(*tup))

        # Error relay setup
        # See comment above signal definition
        self._error_message_relay.connect(lambda handler: handler.display_records(self))

        # Populate the list
        self._reset_list_content()

    def remove_widget_from_cache(self, patient: str):
        """Public method for removing a widget from the cache if present. No signals are emitted."""
        if patient in self._widget_by_patient_cache:
            widget = self._widget_by_patient_cache[patient]
            widget.deleteLater()
            del widget

    def _reload_patient_list(self):
        """If no thread is running, clears the widget cache and resets the list content."""
        self._threads_running_mtx.lock()
        if self._threads_running == 0:
            # Clear the list to avoid anyone pressing any patient reload button or anything like that
            self._list_model.removeRows(0, self._list_model.rowCount())
            self._patient_paths = find_patients_to_annotate(
                self.knowledge_graph,
                self._img_folder,
                self._scene_graph_folder
            )
            # Clear any cached patient that is not there anymore
            for pat in list(self._widget_by_patient_cache.keys()):
                if pat not in self._patient_paths.keys():
                    del self._widget_by_patient_cache[pat]
            self._reset_list_content()
        self._threads_running_mtx.unlock()

    def _reset_list_content(self):
        """Clears the listview and builds it again."""
        # Clear the list
        self._list_model.removeRows(0, self._list_model.rowCount())

        # Filter patients
        pattern = string_to_regex_pattern(self._filter_lineedit.text())
        # Note: filtering preserves sorting
        pat_name_whitelist = sorted([pat for pat in self._patient_paths.keys() if re.match(pattern, pat)])

        # Fill the list
        self._patients_being_loaded = set()
        self._patient_to_row_item = {}
        for pat in pat_name_whitelist:
            # Legacy code: now we always have a file,
            #  but we should be better at showing whether the annotation is complete
            # if self._patient_paths[pat][-1].exists():
            #     # Annotation file exists, so add file icon
            #     icon = self._file_icon
            # else:
            #     # No icon
            #     icon = self._padding_icon

            # Select icon based on progress
            item = QStandardItem(self._progress_to_icon(self._progress[pat]), pat)
            self._list_model.appendRow(item)
            item.setData(pat)
            item.setToolTip(f"Select patient {pat}")
            button = QAlignRightButton(self._reload_icon)
            self._list_view.setIndexWidget(item.index(), button)
            button.clicked.connect(lambda _, pat_=pat: self._reload_patient(pat_, confirm=True))
            button.setToolTip(f"Reload from disk")

            # Update mapping from patient to row item
            self._patient_to_row_item[pat] = item

    def _incr_thread_cnt(self):
        """
        Acquires the mutex lock and increases the thread counter.
        Thread counter is used to disable the main reload button.
        """
        # Update counter and disable main reload button
        self._threads_running_mtx.lock()
        self._threads_running += 1
        self._main_reload_button.setEnabled(False)
        self._threads_running_mtx.unlock()

    def _dcr_thread_cnt(self):
        """
        Acquires the mutex lock and decreases the thread counter.
        Thread counter is used to disable the main reload button.
        """
        # Update counter and disable main reload button
        self._threads_running_mtx.lock()
        self._threads_running -= 1
        if self._threads_running == 0:
            self._main_reload_button.setEnabled(True)
        self._threads_running_mtx.unlock()

    def _create_widget_for_scene_graph(
            self,
            patient: str,
            scene_graph: SceneGraph,
            img: ArrayView,
            save_path: Path
    ):
        """Given the computational results from the thread, create a widget on the main thread."""

        widget = PatientWidget(patient, img, scene_graph, save_path, self._progress[patient])

        self._widget_by_patient_cache[patient] = widget
        widget.scene_graph_saved.connect(self._patient_saved)

        # Emit the new widget
        if patient == self._currently_selected_patient:
            self.patient_selected.emit(widget)

    def _kb_select_patient(self, offset: int):
        """
        :param offset: 0 if selecting current, -1 for previous and +1 for next.
        """
        if self._list_model.rowCount() == 0:
            # Empty list
            return

        # Select current, only if valid
        if offset == 0:
            if self._list_view.currentIndex().isValid():
                self._get_or_compute_patient_widget(self._list_view.currentIndex().data())
                return

        # Otherwise check if any item is selected
        if not self._list_view.currentIndex().isValid():
            new_index = 0
        else:
            # Needs to be always positive
            new_index = (self._list_model.rowCount() + self._list_view.currentIndex().row() +
                         offset) % self._list_model.rowCount()

        self._list_view.setCurrentIndex(self._list_model.index(new_index, 0))
        self._get_or_compute_patient_widget(self._list_view.currentIndex().data())

    def _reload_patient(self, patient: str, confirm: bool = False):
        """Reloads a scene graph from disk (JSOn or build from segmentation) for the specific patient."""

        class SceneGraphLoader(QRunnable):
            """Runnable used for loading the scene graph."""

            def __init__(self, patient_selection_widget: PatientSelectionWidget):
                super().__init__()
                self.selection_widget = patient_selection_widget

            def run(self):
                # Build widget
                img_path, sg_path = self.selection_widget._patient_paths[patient]

                # This code section might be run by multiple threads at the same time
                logger = logging.Logger("AnnotationWindow._editor_widget_factory")
                handler = RecordDisplayHandler()
                logger.addHandler(handler)

                handler.purge()

                # TODO it would be useful to still load the Scene Graph when things are missing
                # Load from JSON if file exists
                scene_graph = SceneGraph.load(sg_path.as_posix(), self.selection_widget.knowledge_graph, logger,
                                              force=True)
                img = load_array_view(self.selection_widget.knowledge_graph, img_path, logger)

                # Check for errors
                if handler.has_errors() or scene_graph is None:
                    self.selection_widget._error_message_relay.emit(handler)
                    # Change placeholder if widget is still selected
                    if patient == self.selection_widget._currently_selected_patient:
                        self.selection_widget.loading_failed.emit()
                else:
                    # Create widget
                    self.selection_widget.create_widget_for_scene_graph.emit((patient, scene_graph, img, sg_path))

                # Do clean-up
                self.selection_widget._patients_being_loaded.remove(patient)
                self.selection_widget.dcr_thread_cnt.emit()

        # Add confirmation window before reload
        if confirm:
            reply = QMessageBox.question(
                self,
                "Are you sure?",
                f"Are you sure that you want to reload the patient {patient}? "
                "Any annotation that was not saved will be lost.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        # Check whether the patient is the currently selected one in which case we need to emit
        if patient == self._currently_selected_patient:
            self.currently_loading.emit(patient)

        # Abort if other thread already running
        if patient in self._patients_being_loaded:
            return
        self._patients_being_loaded.add(patient)

        # Remove old widget from cache
        self.remove_widget_from_cache(patient)

        # Load widget from files on a new thread
        loader = SceneGraphLoader(self)
        self._incr_thread_cnt()
        QThreadPool.globalInstance().start(loader)

    def _get_or_compute_patient_widget(self, patient: str):
        """Handler when a patient name is clicked. Loads from a cache or reloads from disk."""
        self._currently_selected_patient = patient

        # Widget is in cache, just emit it
        if patient in self._widget_by_patient_cache:
            self.patient_selected.emit(self._widget_by_patient_cache[patient])
            return

        # Otherwise reload it from disk
        self._reload_patient(patient)

    def _patient_saved(self, patient: str, progress: Progress):
        """Set the icon for the patient to the save icon, update the progress and save."""
        self._patient_to_row_item[patient].setIcon(self._progress_to_icon(progress))
        self._progress[patient] = progress
        self._progress.save()

    def _progress_to_icon(self, progress: Progress) -> QIcon:
        match progress:
            case Progress.PENDING_REVIEW:
                return self._pending_review_icon
            case Progress.FINISHED:
                return self._finished_icon
            case _:
                return self._padding_icon
