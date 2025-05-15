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

from typing import Any

from PyQt6.QtCore import QLocale, Qt
from PyQt6.QtGui import QIntValidator, QDoubleValidator
from PyQt6.QtWidgets import QWidget, QFormLayout, QLabel, QSizePolicy, QLineEdit, QComboBox, QCheckBox

from scene_graph_annotation.knowledge import ObjectClass, StrAttribute, IntAttribute, FloatAttribute, BoolAttribute, \
    EnumAttribute
from scene_graph_annotation.scene import BoundingBox, Attribute
from .UIManager import UIManager


class ObjectEditorWidget(QWidget):
    """
    Widget used to edit an object's name and attributes.
    Components: form layout with
    - A line edit for the name
    - A label with the bounding box in zyx, dhz mode
    - A line edit for str, int and float with the correct validator
    - A combo box for each bool and enum attributes
    Provides signal for name and attribute signals.
    Provides public methods for refreshing widget contents if there is an external change (per attribute basis).
    """

    def __init__(self, object_class: ObjectClass, object_instance: BoundingBox, ui_manager: UIManager):
        super().__init__()

        self._object_class = object_class
        self._object = object_instance
        self._ui_manager = ui_manager

        # Connect update methods to signals
        ui_manager.name_changed.connect(self.update_name_widget)
        ui_manager.attribute_changed.connect(self.update_attr_widget)

        # Bool attributes to avoid feedback-loops i.e. we receive an update from the signal and re-emit
        self._external_name_change = False
        self._external_attribute_change = False
        # Bool attributes to avoid feedback-loops i.e. we send a signal and we process it
        self._internal_name_change = False
        self._internal_attribute_change = False

        self._name_lineedit = QLineEdit()
        self._attr_widgets_by_id: dict[int, QLineEdit | QComboBox | QCheckBox] = {}

        self.init_ui()

    def init_ui(self):
        layout = QFormLayout()
        self.setLayout(layout)
        self.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Expanding)

        # Add name row
        layout.addRow(QLabel("Name:"), self._name_lineedit)
        self._name_lineedit.setText(self._object.name)
        self._name_lineedit.textChanged.connect(self._name_changed)

        # Add bounding box row
        bbox = self._object.bounding_box
        n_dim = len(bbox[0])
        coord_format = "(" + ", ".join(["d", "h", "w"][:n_dim]) + ")"
        coord_formatted = "(" + ", ".join(str(x2 - x1 + 1) for x1, x2 in zip(bbox[0], bbox[1])) + ")"
        layout.addRow(QLabel(f"Size {coord_format}:"), QLabel(coord_formatted))

        # Attribute rows
        for attr_class in sorted(self._object_class.attributes, key=lambda a: a.name):
            attr = self._object.get_attribute_by_id(attr_class.id)

            # Init the value widget for each possible attribute type
            if isinstance(attr_class, StrAttribute):
                # Str
                value_widget = QLineEdit()
                value_widget.setText(str(attr.value))
                value_widget.textChanged.connect(
                    lambda new_text, attr_=attr: self._attribute_changed(attr_, new_text))
            elif isinstance(attr_class, IntAttribute):
                def _str_to_int(s: str) -> int:
                    if not s:
                        return 0
                    return int(s)

                # Int
                value_widget = QLineEdit()
                value_widget.setText(str(attr.value))
                value_widget.setValidator(QIntValidator())
                value_widget.textChanged.connect(
                    lambda new_text, attr_=attr: self._attribute_changed(attr_, _str_to_int(new_text)))
            elif isinstance(attr_class, FloatAttribute):
                def _str_to_float(s: str) -> float:
                    # We currently only accept floats with a dot
                    if s in ["", "."]:
                        return 0.
                    return float(s)

                # Float
                value_widget = QLineEdit()
                value_widget.setText(str(attr.value))
                val = QDoubleValidator()
                value_widget.setValidator(val)
                # We currently only accept floats with a dot
                loc = QLocale(QLocale.Language.English)
                loc.setNumberOptions(QLocale.NumberOption.RejectGroupSeparator)
                val.setLocale(loc)
                value_widget.textChanged.connect(
                    lambda new_text, attr_=attr: self._attribute_changed(attr_, _str_to_float(new_text)))
            elif isinstance(attr_class, BoolAttribute):
                # Bool
                value_widget = QCheckBox("")
                value_widget.setTristate(False)
                value_widget.setChecked(attr.value)
                value_widget.checkStateChanged.connect(
                    lambda state, attr_=attr: self._attribute_changed(attr_, state == Qt.CheckState.Checked))
            elif isinstance(attr_class, EnumAttribute):
                # Enum
                value_widget = QComboBox()
                # If we wish to sort enum values, then we also need to change how we retrieve the value for the callback
                value_widget.addItems(map(str, attr_class.values))
                value_widget.setCurrentText(str(attr.value))  # Handy bool to int
                value_widget.setEditable(False)
                # Use the value list rather than the text to preserve value type
                value_widget.currentIndexChanged.connect(
                    lambda idx, attr_=attr, tmp=attr_class: self._attribute_changed(attr_, tmp.values[idx]))
            else:
                assert False

            self._attr_widgets_by_id[attr.id] = value_widget
            layout.addRow(QLabel(attr_class.name + ":"), value_widget)

    def update_name_widget(self, obj: BoundingBox):
        """Method to call to update the name widget when there was an external change to its value."""
        if obj == self._object and not self._internal_name_change:
            # Prevent feedback-loop
            self._external_name_change = True
            self._name_lineedit.setText(self._object.name)
            self._external_name_change = False

    def update_attr_widget(self, obj: BoundingBox, attr: Attribute):
        """Method to call to update a specific attribute widget when there was an external change to its value."""
        if obj != self._object or self._internal_attribute_change:
            return

        # Prevent feedback-loop
        self._external_attribute_change = True
        attr_class = self._object_class.get_attribute_by_id(attr.id)
        if isinstance(attr_class, StrAttribute) or \
                isinstance(attr_class, IntAttribute) or \
                isinstance(attr_class, FloatAttribute):
            # Widget is a lineedit
            attr_widget: QLineEdit = self._attr_widgets_by_id[attr.id]
            attr_widget.setText(str(attr.value))
        elif isinstance(attr_class, BoolAttribute):
            # Widget is a combobox
            attr_widget: QCheckBox = self._attr_widgets_by_id[attr.id]
            attr_widget.setChecked(attr.value)
        elif isinstance(attr_class, EnumAttribute):
            # Widget is a combobox
            attr_widget: QComboBox = self._attr_widgets_by_id[attr.id]
            # We need to find the index of the value in the value list
            idx = attr_class.values.index(attr.value)
            attr_widget.setCurrentIndex(idx)
        else:
            assert False
        self._external_attribute_change = False

    def _name_changed(self):
        """Callback for updating the name of the object when the text in the lineedit changes."""
        # Prevent feedback-loop
        if not self._external_name_change:
            self._internal_name_change = True
            self._object.name = self._name_lineedit.text()
            self._ui_manager.name_changed.emit(self._object)
            self._internal_name_change = False

    def _attribute_changed(self, attr: Attribute, value: Any):
        """Callback for updating the attribute of the object when the value in the widget changes."""
        # Prevent feedback-loop
        if not self._external_attribute_change:
            self._internal_attribute_change = True
            attr.value = value
            self._ui_manager.attribute_changed.emit(self._object, attr)
            self._internal_attribute_change = False
