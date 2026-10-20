import datetime

import qt
import slicer

from scene_graph_database.functions import progress_update_to_study
from scene_graph_database.structures import AnnotationProgress
from .collapsible_button_widget import CollapsibleButtonWidget
from .patient_tab import PatientTab
from .studies_tab import StudiesTab
from .toggle import ResponsiveToggle
from ..load_state import LOAD_STATE_MANAGER
from ..nodes import ParameterNodeHandler
from .objects_tab import ObjectsTab
from .relations_tab import RelationsTab


class ToolsWidget(CollapsibleButtonWidget):
    """Widget with parameters for the annotation overview. I.e. the path to the database and the reader name."""

    def __init__(
            self,
            node_handler: ParameterNodeHandler,
            segment_editor_widget: qt.QWidget,
    ):
        super().__init__()

        self._node_handler = node_handler

        # Slicer-common widgets that we need to track
        self._data_module = slicer.modules.data.widgetRepresentation()
        self._segment_editor_widget = segment_editor_widget
        self._volumes_module = slicer.modules.volumes.widgetRepresentation()

        self._tools_content = qt.QTabWidget()
        self._studies_tab = StudiesTab()
        self._patient_tab = PatientTab()
        self._objects_tab = ObjectsTab()
        self._relations_tab = RelationsTab()

        self._show_preview_toggle = ResponsiveToggle()
        self._show_ai_pred_toggle = ResponsiveToggle()
        self._select_prev_patient_button = qt.QPushButton("Previous")
        self._select_next_patient_button = qt.QPushButton("Next")
        self._save_button = qt.QPushButton("Save")

        # Connect to relevant signals
        def f():
            self._show_preview_toggle.setChecked(LOAD_STATE_MANAGER.show_preview)
            self._show_ai_pred_toggle.setChecked(LOAD_STATE_MANAGER.show_ai_prediction)

        LOAD_STATE_MANAGER.study_selected.connect(self.setup_tabs)
        LOAD_STATE_MANAGER.sg_loaded.connect(self._update_button_states)
        LOAD_STATE_MANAGER.preview_toggled.connect(f)
        LOAD_STATE_MANAGER.ai_prediction_toggled.connect(f)

        self.init_ui()

    def init_ui(self):
        self.setText("Tools")

        layout = qt.QVBoxLayout()
        self.setButtonLayout(layout)

        # Toggle views
        button_row = qt.QWidget()
        button_layout = qt.QHBoxLayout()
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_row.setLayout(button_layout)
        layout.addWidget(button_row)

        interm_widget = qt.QWidget()
        interm_layout = qt.QFormLayout()
        button_layout.addWidget(interm_widget)
        interm_layout.setContentsMargins(0, 0, 0, 0)
        interm_widget.setLayout(interm_layout)
        interm_layout.addRow(qt.QLabel("Show preview"), self._show_preview_toggle)

        def f(checked):
            LOAD_STATE_MANAGER.show_preview = checked

        self._show_preview_toggle.toggled.connect(f)

        interm_widget = qt.QWidget()
        interm_layout = qt.QFormLayout()
        button_layout.addWidget(interm_widget)
        interm_layout.setContentsMargins(0, 0, 0, 0)
        interm_widget.setLayout(interm_layout)
        interm_layout.addRow(qt.QLabel("Show AI prediction"), self._show_ai_pred_toggle)

        def f(checked):
            LOAD_STATE_MANAGER.show_ai_prediction = checked

        self._show_ai_pred_toggle.toggled.connect(f)

        # Buttons select prev/next patient
        interm_widget = qt.QWidget()
        interm_layout = qt.QHBoxLayout()
        layout.addWidget(interm_widget)
        interm_layout.setContentsMargins(0, 0, 0, 0)
        interm_widget.setLayout(interm_layout)
        interm_layout.addWidget(self._select_prev_patient_button)
        interm_layout.addWidget(self._select_next_patient_button)

        self._select_prev_patient_button.setToolTip("Select the previous patient")
        self._select_next_patient_button.setToolTip("Select the next patient")

        self._select_prev_patient_button.clicked.connect(
            lambda: self._studies_tab.patient_table.select_patient_relative_to_current(-1)
        )
        self._select_next_patient_button.clicked.connect(
            lambda: self._studies_tab.patient_table.select_patient_relative_to_current(1)
        )

        # Save button
        layout.addWidget(self._save_button)
        self._save_button.clicked.connect(self.save_patient)
        self._save_button.setShortcut(qt.QKeySequence("s"))
        self._save_button.setStyleSheet("font-weight: bold;")
        self._save_button.setToolTip("Save the current patient")
        self._update_button_states()

        # Tabs
        layout.addWidget(self._tools_content)
        self.setup_tabs()
        self._tools_content.currentChanged.connect(self.selected_tab_changed)

    def setup_tabs(self):
        """Resets the tabs in the QTabWidget."""
        self._tools_content.clear()
        self._tools_content.addTab(self._studies_tab, "Studies")
        self._tools_content.addTab(self._patient_tab, "Patient")
        # Cannot create a new widget, but we can set nodes globally
        self._tools_content.addTab(self._data_module, "Data")
        # Could create a new widget, but we can set nodes globally
        self._tools_content.addTab(self._segment_editor_widget, "Segment Editor")
        # Cannot create a new widget, but we can set nodes globally
        self._tools_content.addTab(self._volumes_module, "Volume")

        # Check if we need to disable some tabs
        kg = LOAD_STATE_MANAGER.knowledge_graph
        if kg is not None:
            # Check whether there is anything relevant for objects
            objects_enabled = len(kg.image.attributes) > 0 or len(kg.image.keypoints) > 0 or \
                              any(len(obj_cls.attributes) > 0 or len(obj_cls.keypoints) > 0 for obj_cls in kg.classes)
            if objects_enabled:
                self._tools_content.addTab(self._objects_tab, "Objects")

            self._objects_tab.setEnabled(objects_enabled)
            # Check whether there is anything relevant for relations
            relations_enabled = len(kg.rules) > 0
            if relations_enabled:
                self._tools_content.addTab(self._relations_tab, "Relations")

        # Also update tabs states
        self._update_button_states()

    def selected_tab_changed(self):
        """Sets/unsets keyboard shortcuts depending on which tab is selected."""
        if self._tools_content.currentIndex == 3:
            # Segment editor
            self._segment_editor_widget.self().editor.installKeyboardShortcuts()
        else:
            try:
                self._segment_editor_widget.self().editor.uninstallKeyboardShortcuts()
            except ValueError:
                # On closing the app, the editor is already destroyed
                ...

    def _update_button_states(self):
        """Enable or disable the save button depending on whether a patient is selected."""
        sg_loaded = LOAD_STATE_MANAGER.scene_graph is not None
        self._select_prev_patient_button.setEnabled(sg_loaded)
        self._select_next_patient_button.setEnabled(sg_loaded)
        self._save_button.setEnabled(sg_loaded)

        # Also enable the objects/relations tab (if added)
        for tab_idx in range(5, self._tools_content.count):
            self._tools_content.setTabEnabled(tab_idx, sg_loaded)

    def save_patient(self):
        """Save the current patient."""
        try:
            # Two things: save the progress and the Scene Graph itself
            if LOAD_STATE_MANAGER.selected_patient_scene_graph_modified:
                self._node_handler.save_all_annotations()

            new_progress = AnnotationProgress(
                reader=LOAD_STATE_MANAGER.reader,
                progress=LOAD_STATE_MANAGER.annotation_progress,
                when=datetime.datetime.now()
            )
            progress_update_to_study(
                LOAD_STATE_MANAGER.selected_study.id,
                {LOAD_STATE_MANAGER.selected_patient_name: new_progress}
            )
            # Also update our local progress
            LOAD_STATE_MANAGER.selected_study.progress[LOAD_STATE_MANAGER.selected_patient_name] = new_progress
            self._studies_tab.patient_table.update_selected_row()

            slicer.util.confirmOkCancelDisplay(f"Scene Graph saved!", windowTitle="Success", parent=None)
        except Exception as e:
            slicer.util.errorDisplay(f"Could not save Scene Graph: {e}", windowTitle="Error", parent=None)
