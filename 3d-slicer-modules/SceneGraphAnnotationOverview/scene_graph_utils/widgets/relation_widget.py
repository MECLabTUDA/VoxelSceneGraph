import qt

from scene_graph_api.scene import Relation, BoundingBox
from ..load_state import LOAD_STATE_MANAGER


class RelationWidget(qt.QGroupBox):
    """
    Widget with labels representing the relations. Also a delete button.
    Note: this widget is self-aware and will delete itself if the relation is removed.
    """

    def __init__(self, relation: Relation):
        super().__init__()

        self._relation = relation

        self._subject_icon = qt.QLabel()
        self._subject_label = qt.QLabel()
        self._relation_label = qt.QLabel()
        self._object_icon = qt.QLabel()
        self._object_label = qt.QLabel()
        self._delete_button = qt.QPushButton("Delete")

        self._old_subj_color = ""
        self._old_obj_color = ""

        LOAD_STATE_MANAGER.name_changed.connect(self._update_names)
        LOAD_STATE_MANAGER.update_color.connect(self._update_colors)
        LOAD_STATE_MANAGER.sg_object_removed.connect(self._maybe_suicide)

        self.init_ui()

    def init_ui(self):
        layout = qt.QHBoxLayout()
        self.setLayout(layout)
        self._subject_icon.setSizePolicy(qt.QSizePolicy.Policy.Minimum, qt.QSizePolicy.Policy.Preferred)
        layout.addWidget(self._subject_icon)
        layout.addWidget(self._subject_label)
        layout.addWidget(self._relation_label)
        self._relation_label.setText(
            LOAD_STATE_MANAGER.knowledge_graph.get_rule_by_id(self._relation.rule_id).name
        )
        self._object_icon.setSizePolicy(qt.QSizePolicy.Policy.Minimum, qt.QSizePolicy.Policy.Preferred)
        layout.addWidget(self._object_icon)
        layout.addWidget(self._object_label)
        layout.addStretch()
        layout.addWidget(self._delete_button)
        self._delete_button.clicked.connect(self._delete_relation)

        self._update_names(LOAD_STATE_MANAGER.scene_graph.get_bounding_box_by_id(self._relation.subject_id))
        self._update_names(LOAD_STATE_MANAGER.scene_graph.get_bounding_box_by_id(self._relation.object_id))
        self._update_colors()

    def _update_names(self, obj: BoundingBox):
        """Updates the name of the subject and object."""
        if obj.id == self._relation.subject_id:
            self._subject_label.setText(obj.name)
        elif obj.id == self._relation.object_id:
            self._object_label.setText(obj.name)

    def _update_colors(self):
        """Updates the icon colors."""
        new_subj_color = LOAD_STATE_MANAGER.colors.get(self._relation.subject_id)
        if new_subj_color != self._old_subj_color:
            self._old_subj_color = new_subj_color
            self._subject_icon.setStyleSheet(f"background-color: {new_subj_color};")

        new_obj_color = LOAD_STATE_MANAGER.colors.get(self._relation.object_id)
        if new_obj_color != self._old_obj_color:
            self._old_obj_color = new_obj_color
            self._object_icon.setStyleSheet(f"background-color: {new_obj_color};")

    def _maybe_suicide(self, obj: BoundingBox):
        """Deletes the widget if the object is removed."""
        if obj.id == self._relation.subject_id or obj.id == self._relation.object_id:
            try:
                self.deleteLater()
            except ValueError:
                # Object already destroyed
                return

    def _delete_relation(self):
        """Deletes the relation."""
        LOAD_STATE_MANAGER.selected_patient_scene_graph_modified = True
        LOAD_STATE_MANAGER.scene_graph.remove_relation(self._relation)
        LOAD_STATE_MANAGER.relation_removed.emit(self._relation)
        self.deleteLater()
