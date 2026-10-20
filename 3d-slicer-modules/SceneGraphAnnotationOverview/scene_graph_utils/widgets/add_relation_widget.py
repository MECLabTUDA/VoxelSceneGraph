import qt

from scene_graph_api.scene import Relation
from ..load_state import LOAD_STATE_MANAGER
from .collapsible_button_widget import CollapsibleButtonWidget
from PythonQt.QtGui import QColor, QPixmap, QIcon
from .object_selector import ObjectSelector


class AddRelationWidget(CollapsibleButtonWidget):
    """
    Widget to add a relation between two objects.
    It's composed of 3 QComboBoxes and a button.
    """

    def __init__(self):
        super().__init__()

        self._subject_combobox = ObjectSelector()
        self._relation_combobox = qt.QComboBox()
        self._object_combobox = ObjectSelector()
        self._add_button = qt.QPushButton("Add")

        # Connect to the study list update
        LOAD_STATE_MANAGER.sg_loaded.connect(self._update_ui)

        LOAD_STATE_MANAGER.subject_changed.connect(self._subject_select_current)
        LOAD_STATE_MANAGER.object_changed.connect(self._object_select_current)

        self.destroyed.connect(self._disconnect_signals)

        self.init_ui()

    def init_ui(self):
        self.setText("Add relation")

        layout = qt.QFormLayout()
        self.setButtonLayout(layout)

        layout.addRow(qt.QLabel("Subject:"), self._subject_combobox)
        layout.addRow(qt.QLabel("Object:"), self._object_combobox)
        layout.addRow(qt.QLabel("Relation:"), self._relation_combobox)
        layout.addRow(self._add_button)

        self._subject_combobox.currentIndexChanged.connect(self._update_relation_combobox)
        self._object_combobox.currentIndexChanged.connect(self._update_relation_combobox)
        self._relation_combobox.currentIndexChanged.connect(self._enable_button)
        self._add_button.clicked.connect(self._add_relation)

        self._update_ui()

    def _update_ui(self):
        """Sets the state of form widgets."""
        # FIXME remove this when triggering is fixed
        self._subject_combobox.update_object_list()
        self._object_combobox.update_object_list()

        self._update_relation_combobox()
        self._enable_button()

    def _update_relation_combobox(self):
        """Updates selected subject and object. Also updates the predicate selector."""
        if LOAD_STATE_MANAGER.scene_graph is None:
            return

        # Update LOAD_STATE_MANAGER.subject and LOAD_STATE_MANAGER.object
        if self._subject_combobox.currentIndex > 0:
            LOAD_STATE_MANAGER.subject = self._subject_combobox.currentData
        else:
            LOAD_STATE_MANAGER.subject = None
        if self._object_combobox.currentIndex > 0:
            LOAD_STATE_MANAGER.object = self._object_combobox.currentData
        else:
            LOAD_STATE_MANAGER.object = None

        if LOAD_STATE_MANAGER.subject is None or LOAD_STATE_MANAGER.object is None:
            self._relation_combobox.clear()
            self._relation_combobox.addItem("No possible relation")
            self._relation_combobox.setEnabled(False)
            return

        self._relation_combobox.clear()
        self._relation_combobox.setEnabled(True)
        # Compute possible relations
        possible_rules = LOAD_STATE_MANAGER.scene_graph.knowledge_graph.get_valid_rule_ids(
            LOAD_STATE_MANAGER.subject.class_id, LOAD_STATE_MANAGER.object.class_id
        )
        for rule in possible_rules:
            self._relation_combobox.addItem(rule.name, rule.id)  # Rule id as item data
        # Select first item if not empty so that a rule is ALWAYS selected and the add button has no issue
        if possible_rules:
            self._relation_combobox.setCurrentIndex(0)

        self._enable_button()

    def _enable_button(self):
        """Enables the add button if all selections are valid."""
        self._add_button.setEnabled(
            self._subject_combobox.currentIndex > 0 and
            self._relation_combobox.currentIndex != -1 and
            self._object_combobox.currentIndex > 0
        )

    def _add_relation(self):
        """Add a relation between the selected objects."""
        rule_id = self._relation_combobox.currentData
        new_rel = Relation(rule_id, LOAD_STATE_MANAGER.subject.id, LOAD_STATE_MANAGER.object.id)
        scene_graph = LOAD_STATE_MANAGER.scene_graph
        new_rel.init_attributes_keypoints(
            scene_graph.knowledge_graph.get_rule_by_id(rule_id),
            len(scene_graph.object_labelmap.shape),
            scene_graph.knowledge_graph.rel_common
        )
        scene_graph.add_relation(new_rel)
        LOAD_STATE_MANAGER.selected_patient_scene_graph_modified = True
        LOAD_STATE_MANAGER.relation_added.emit(new_rel)

    def _subject_select_current(self):
        """Selects the current subject in the subject combobox."""
        self._subject_combobox.select_object(LOAD_STATE_MANAGER.subject)

    def _object_select_current(self):
        """Selects the current object in the subject combobox."""
        self._object_combobox.select_object(LOAD_STATE_MANAGER.object)

    def _disconnect_signals(self):
        """Disconnects all signals."""
        self._subject_combobox.currentIndexChanged.disconnect(self._update_relation_combobox)
        self._object_combobox.currentIndexChanged.disconnect(self._update_relation_combobox)
        self._relation_combobox.currentIndexChanged.disconnect(self._enable_button)
        self._add_button.clicked.disconnect(self._add_relation)

        LOAD_STATE_MANAGER.sg_loaded.disconnect(self._update_ui)

        LOAD_STATE_MANAGER.subject_changed.disconnect(self._subject_select_current)
        LOAD_STATE_MANAGER.object_changed.disconnect(self._object_select_current)
