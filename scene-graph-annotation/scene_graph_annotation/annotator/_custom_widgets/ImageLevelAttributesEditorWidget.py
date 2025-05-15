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

from PyQt6.QtCore import QLocale
from PyQt6.QtGui import QIntValidator, QDoubleValidator
from PyQt6.QtWidgets import QWidget, QFormLayout, QLabel, QSizePolicy, QLineEdit, QComboBox

from scene_graph_annotation.knowledge import StrAttribute, IntAttribute, \
    FloatAttribute, BoolAttribute, EnumAttribute, ObjectClass
from scene_graph_annotation.scene import Attribute, Object


class ImageLevelAttributesEditorWidget(QWidget):
    """
    Widget used to edit attributes at image level (i.e. dumbed-down version of ObjectEditorWidget).
    Components: form layout with
    - A combo box for each bool and enum attributes
    Provides signal for name and attribute signals.
    Provides public methods for refreshing widget contents if there is an external change (per attribute basis).
    """

    def __init__(self, image_object_class: ObjectClass, image_object: Object):
        super().__init__()

        self._image_object_class = image_object_class
        self._image_object = image_object

        self.init_ui()

    def init_ui(self):
        layout = QFormLayout()
        self.setLayout(layout)
        self.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Expanding)

        # Attribute rows
        for attr_class in sorted(self._image_object_class.attributes, key=lambda a: a.name):
            attr = self._image_object.get_attribute_by_id(attr_class.id)

            # Init the value widget for each possible attribute type
            if isinstance(attr_class, StrAttribute):
                # Str
                value_widget = QLineEdit()
                value_widget.setText(str(attr.value))
                value_widget.textChanged.connect(lambda new_text, attr_=attr: self._attribute_changed(attr_, new_text))

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
                value_widget = QComboBox()
                value_widget.addItems(["False", "True"])
                value_widget.setCurrentIndex(int(attr.value))  # Handy bool to int
                value_widget.setEditable(False)
                value_widget.currentIndexChanged.connect(
                    lambda idx, attr_=attr: self._attribute_changed(attr_, bool(idx)))  # Handy int to bool

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

            layout.addRow(QLabel(attr_class.name + ":"), value_widget)

    @staticmethod
    def _attribute_changed(attr: Attribute, value: Any):
        """Callback for updating the attribute of the object when the value in the widget changes."""
        attr.value = value
