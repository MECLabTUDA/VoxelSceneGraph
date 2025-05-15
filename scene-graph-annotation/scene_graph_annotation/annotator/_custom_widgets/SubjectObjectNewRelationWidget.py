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

import cc3d
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QWidget, QLabel, QSizePolicy, QGroupBox, QStackedWidget, QVBoxLayout, QHBoxLayout, \
    QToolButton, QPushButton, QComboBox, QMessageBox

from scene_graph_annotation.scene import SceneGraph, BoundingBox, CompositeBoundingBox, \
    Object, Relation
from scene_graph_annotation.scene.object_merging import merge_with_mask
from scene_graph_annotation.scene.object_splitting import split_with_mask, split_into_connected_components
from scene_graph_annotation.utils.asset_paths import swap_icon, merge_icon, split_icon, split_cc_icon
from .ObjectEditorPlaceholderWidget import ObjectEditorPlaceholderWidget
from .UIManager import UIManager


class SubjectObjectNewRelationWidget(QGroupBox):
    """
    Widget displayed under the image and used for:
    - seeing/editing currently selected subject and object
    - swapping subject and object
    - merging (if possible) subject and object
    - split subject (exclusive) or object if it is a composite segmentation and the other is empty
    - split subject (exclusive) or object into connected components (opens a confirmation prompt)
    - add a new relation from a combobox of possible/valid relation rules based on subject/object classes.
    """

    def __init__(self, scene_graph: SceneGraph, ui_manager: UIManager):
        super().__init__(title="Relation Editor")

        self._scene_graph = scene_graph
        self._ui_manager = ui_manager

        self._stacked_widget = QStackedWidget()

        self._fixed_height = 250

        # Widgets that need to be manually updated when subject/object changes
        self._swap_button = QToolButton()
        self._merge_button = QToolButton()
        self._split_button = QToolButton()
        self._split_cc_button = QToolButton()
        self._subject_name_label = QLabel()
        self._relation_combobox = QComboBox()
        self._object_name_label = QLabel()
        self._add_relation_button = QPushButton()

        # Connect self to signals
        ui_manager.subject_changed.connect(self._subject_or_object_changed)
        ui_manager.object_changed.connect(self._subject_or_object_changed)
        ui_manager.name_changed.connect(self._update_subject_label_text)
        ui_manager.name_changed.connect(self._update_object_label_text)

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)
        self.setFixedHeight(self._fixed_height)

        # ==============================================================================================================
        # Add first row i.e. stretch, subject groupbox, spacing, buttons column, spacing, object groupbox, stretch
        editors_buttons_row = QWidget()
        editors_buttons_layout = QHBoxLayout()
        editors_buttons_row.setLayout(editors_buttons_layout)
        editors_buttons_layout.setContentsMargins(0, 0, 0, 0)
        editors_buttons_row.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Subject
        subject_editor_groupbox = ObjectEditorPlaceholderWidget(
            ObjectEditorPlaceholderWidget.ObjectRole.SUBJECT,
            self._scene_graph.knowledge_graph,
            self._ui_manager
        )
        editors_buttons_layout.addWidget(subject_editor_groupbox)
        editors_buttons_layout.addSpacing(5)
        subject_editor_groupbox.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # Buttons column
        buttons_column = QWidget()
        buttons_column_layout = QVBoxLayout()
        buttons_column.setLayout(buttons_column_layout)
        buttons_column_layout.addStretch()

        self._swap_button.setIcon(QIcon(swap_icon.as_posix()))
        buttons_column_layout.addWidget(self._swap_button)
        self._swap_button.setToolTip("Swap subject and object")
        self._swap_button.clicked.connect(self._swap)
        buttons_column_layout.addSpacing(5)

        self._merge_button.setIcon(QIcon(merge_icon.as_posix()))
        buttons_column_layout.addWidget(self._merge_button)
        self._merge_button.setToolTip("Merge subject and object")
        self._merge_button.clicked.connect(self._merge)
        buttons_column_layout.addSpacing(5)

        self._split_button.setIcon(QIcon(split_icon.as_posix()))
        buttons_column_layout.addWidget(self._split_button)
        self._split_button.setToolTip(
            "Undo a merge operation\n"
            "(only a subject or an object needs to be selected)"
        )
        self._split_button.clicked.connect(self._split)

        self._split_cc_button.setIcon(QIcon(split_cc_icon.as_posix()))
        buttons_column_layout.addWidget(self._split_cc_button)
        self._split_cc_button.setToolTip(
            "Split either a subject or an object into its connected components\n"
            "(only a subject or an object needs to be selected, needs to have a mask and"
            "must not be unique)"
        )
        self._split_cc_button.clicked.connect(self._split_cc)

        buttons_column_layout.addStretch()
        editors_buttons_layout.addWidget(buttons_column)

        editors_buttons_layout.addSpacing(5)
        # Object
        object_editor_groupbox = ObjectEditorPlaceholderWidget(
            ObjectEditorPlaceholderWidget.ObjectRole.OBJECT,
            self._scene_graph.knowledge_graph,
            self._ui_manager
        )
        editors_buttons_layout.addWidget(object_editor_groupbox)
        object_editor_groupbox.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        layout.addWidget(editors_buttons_row)
        # ==============================================================================================================
        # Add second row i.e. groupbox with label subject, spacing, combobox, spacing, label, object, spacing,
        # add button
        add_relation_widget = QGroupBox("Add new relation")
        add_relation_layout = QHBoxLayout()
        add_relation_widget.setLayout(add_relation_layout)
        add_relation_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self._subject_name_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        add_relation_layout.addWidget(self._subject_name_label)
        add_relation_layout.addSpacing(5)

        self._relation_combobox.setEditable(False)
        add_relation_layout.addWidget(self._relation_combobox)
        add_relation_layout.addSpacing(5)

        self._object_name_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        add_relation_layout.addWidget(self._object_name_label)
        add_relation_layout.addSpacing(5)

        self._add_relation_button.setText("Add relation")
        self._add_relation_button.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        add_relation_layout.addWidget(self._add_relation_button)
        self._add_relation_button.clicked.connect(self._add_relation)

        layout.addWidget(add_relation_widget)

        # Update state of widgets
        self._subject_or_object_changed()

    def _subject_or_object_changed(self):
        """
        Callback for when teh subject or object changes:
        - Updates the state (isEnabled) or the merge and split buttons.
          These should only be enabled when the subject/object can be merged/split.
        - Updates the text in the label for relation creation.
        - Updates the content of the combobox with the possible relation rules.
        - Enables the "add relation" button if there is at least one valid rule for the combination.
        """
        subject_inst = self._ui_manager.subject
        object_inst = self._ui_manager.object

        # Update buttons column state
        # Swap enabled if at least one not None
        self._swap_button.setEnabled(subject_inst is not None or object_inst is not None)
        # Merge enabled only if subject and object are both segmentations with the same class id
        merge_enabled = (subject_inst is not None and
                         object_inst is not None and
                         # subject_inst.has_mask == object_inst.has_mask and
                         subject_inst.class_id == object_inst.class_id and
                         subject_inst.id != object_inst.id)
        self._merge_button.setEnabled(merge_enabled)
        # Split enabled if subject or object is a composite segmentation and the other is None
        # Or also if subject == object and is a composite segmentation
        split_enabled = (
                (isinstance(subject_inst, CompositeBoundingBox) and object_inst is None) or
                (subject_inst is None and isinstance(object_inst, CompositeBoundingBox)) or
                (isinstance(subject_inst, CompositeBoundingBox) and subject_inst == object_inst)
        )
        self._split_button.setEnabled(split_enabled)
        # Split cc enabled if subject or object is not None + has a mask + not unique bbox class and the other is None
        # Or also if subject == object and has a mask
        knowledge_graph = self._scene_graph.knowledge_graph
        split_cc_enabled = (
                (
                        subject_inst is not None and
                        # subject_inst.has_mask and
                        not knowledge_graph.get_object_class_by_id(subject_inst.class_id).is_unique and
                        object_inst is None
                ) or (
                        subject_inst is None and
                        object_inst is not None and
                        # object_inst.has_mask and
                        not knowledge_graph.get_object_class_by_id(object_inst.class_id).is_unique
                ) or (
                        subject_inst is not None and
                        # subject_inst.has_mask and
                        not knowledge_graph.get_object_class_by_id(subject_inst.class_id).is_unique and
                        subject_inst == object_inst
                )
        )
        self._split_cc_button.setEnabled(split_cc_enabled)

        # Set labels
        self._update_subject_label_text(subject_inst)
        self._update_object_label_text(object_inst)
        # Set combobox clear values and set disabled or values
        self._relation_combobox.clear()
        if subject_inst is None or object_inst is None:
            self._relation_combobox.setEnabled(False)
        else:
            self._relation_combobox.setEnabled(True)
            possible_rules = self._scene_graph.knowledge_graph.get_valid_rule_ids(
                subject_inst.class_id, object_inst.class_id
            )
            for rule in possible_rules:
                self._relation_combobox.addItem(rule.name, rule.id)  # Rule id as item data
            # Select first item if not empty so that a rule is ALWAYS selected and the add button has no issue
            if possible_rules:
                self._relation_combobox.setCurrentIndex(0)
        # Add button enabled only if the combobox is enabled and has values
        add_rel_enabled = self._relation_combobox.isEnabled() and self._relation_combobox.count()
        self._add_relation_button.setEnabled(add_rel_enabled)

    def _update_subject_label_text(self, obj: Object):
        """Updates the text in the label if the object that changed is the subject."""
        if obj == self._ui_manager.subject:
            self._subject_name_label.setText(obj.name if obj is not None else "(No subject selected)")

    def _update_object_label_text(self, obj: Object):
        """Updates the text in the label if the object that changed is the object."""
        if obj == self._ui_manager.object:
            self._object_name_label.setText(obj.name if obj is not None else "(No object selected)")

    def _swap(self):
        """Callback for swapping subject and object."""
        ui_manager = self._ui_manager
        ui_manager.subject, ui_manager.object = ui_manager.object, ui_manager.subject

    def _merge(self):
        """
        Callback for merging subject and object:
        - create a composite segmentation
        - update the subject and object
        - add/remove the corresponding segmentations
        """
        subject_inst = self._ui_manager.subject
        object_inst = self._ui_manager.object
        # If the button is enabled, we already know that the subject and object are Segmentations
        # The method already removes the subject/object from the scene and add the composite segmentation
        if isinstance(subject_inst, BoundingBox):
            comp = merge_with_mask(self._scene_graph, subject_inst, object_inst)
        else:
            # If we're here, then the checks for enabling the merge button have failed
            raise RuntimeError("Trying to merge when only one or none objects are selected.")
        # Update subject and object
        self._ui_manager.subject = comp
        self._ui_manager.object = None
        # Emit deletions/creation
        self._ui_manager.sg_object_removed.emit(subject_inst)
        self._ui_manager.sg_object_removed.emit(object_inst)
        self._ui_manager.sg_object_added.emit(comp)

    def _split(self):
        """
        Callback for splitting subject or object:
        - get the original pair of objects from the composite segmentation
        - add/remove the corresponding segmentations and update associated relations
        """
        # If the button is enabled, then we know that either the subject or the object is a CompositeSegmentation
        # and that the other is None
        subj_or_obj = self._ui_manager.subject or self._ui_manager.object
        if not isinstance(subj_or_obj, CompositeBoundingBox):
            # If we're here, then the checks for enabling the split button have failed
            raise RuntimeError("Trying to split a non-composite bounding-box.")

        # The method already removes the composite bb from the scene and adds the two components
        # if subj_or_obj.has_mask:
        comp1, comp2 = split_with_mask(subj_or_obj, self._scene_graph)
        # else:
        #     comp1, comp2 = split(subj_or_obj, self._scene_graph)

        # Update subject and object
        self._ui_manager.subject = comp1
        self._ui_manager.object = comp2
        # Emit deletions/creation
        self._ui_manager.sg_object_removed.emit(subj_or_obj)
        self._ui_manager.sg_object_added.emit(comp1)
        self._ui_manager.sg_object_added.emit(comp2)

    def _split_cc(self):
        """
        Callback for splitting subject or object into connected components:
        - check that the object has a mask
        - compute connected components
        - open confirmation prompt with number of components found
        - add/remove the corresponding segmentations and update associated relations
        Note: if only one connected component is found, nothing happens too.
        """
        # If the button is enabled, then we know that either the subject or the object has a mask and the other is None
        subj_or_obj = self._ui_manager.subject or self._ui_manager.object
        # if not subj_or_obj.has_mask:
        #     # If we're here, then the checks for enabling the split cc button have failed
        #     raise RuntimeError("Trying to split a bounding box in to connected components.")

        components, n_components = cc3d.connected_components(
            self._scene_graph.object_labelmap == subj_or_obj.id,
            return_N=True
        )

        # Confirmation dialog
        ans = QMessageBox.question(
            self,
            "Are you sure?",
            f"{n_components} connected components have been found. Do you want to proceed with the split?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel
        )

        if n_components == 1 or ans == QMessageBox.StandardButton.Cancel:
            # Even if the user accepts, nothing has to be done
            return

        new_boxes = split_into_connected_components(subj_or_obj, self._scene_graph, components, n_components)

        # Update subject and object
        self._ui_manager.subject = None
        self._ui_manager.object = None
        # Emit deletions/creation
        self._ui_manager.sg_object_removed.emit(subj_or_obj)
        for new_box in new_boxes:
            self._ui_manager.sg_object_added.emit(new_box)

    def _add_relation(self):
        """Callback for adding a new relation."""
        rule_id = self._relation_combobox.currentData()
        subject_inst = self._ui_manager.subject
        object_inst = self._ui_manager.object
        new_rel = Relation(rule_id, subject_inst.id, object_inst.id)
        self._scene_graph.add_relation(new_rel)
        self._ui_manager.relation_added.emit(new_rel)
