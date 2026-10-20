from typing import Any

import qt
from PythonQt.QtCore import QLocale, Qt
from PythonQt.QtGui import QIntValidator, QDoubleValidator

from scene_graph_api.knowledge import ObjectClass, StrAttribute, IntAttribute, FloatAttribute, BoolAttribute, \
    EnumAttribute, Attribute as AttributeClass
from scene_graph_api.scene import BoundingBox, Attribute
from .collapsible_button_widget import CollapsibleButtonWidget
from ..load_state import LOAD_STATE_MANAGER
from .toggle import ResponsiveToggle


class ObjectEditorWidget(CollapsibleButtonWidget):
    """
    Widget used to edit an object's name and attributes.
    Components: form layout with
    - A line edit for the name (if not image-level)
    - A label for the object class
    - A line edit for str, int and float with the correct validator
    - A combo box for each bool and enum attributes
    A row of buttons to select as subject/object and show the object in the views.
    The icon color should match the segment color.
    """

    def __init__(
            self,
            object_class: ObjectClass,
            object_instance: BoundingBox,
            common_object_class: ObjectClass | None = None,
            is_image_level: bool = False,  # Whether this is for the annotation of an image-level attributes
    ):
        super().__init__()

        self._object_class = object_class
        self._common_object_class = common_object_class
        self._object = object_instance
        self._is_image_level = is_image_level

        self._old_color = ""

        self._colored_icon = qt.QLabel()
        self.set_subject_button = qt.QPushButton()
        self.set_object_button = qt.QPushButton()

        # Connect update methods to signals
        LOAD_STATE_MANAGER.name_changed.connect(self.update_name_widget)
        LOAD_STATE_MANAGER.update_color.connect(self._set_color)
        LOAD_STATE_MANAGER.subject_changed.connect(self._subject_changed)
        LOAD_STATE_MANAGER.object_changed.connect(self._object_changed)

        self.destroyed.connect(self._disconnect_signals)

        self._name_lineedit = qt.QLineEdit()
        self.init_ui()

    def init_ui(self):
        if self._is_image_level:
            self.setText("Image-level Attributes")
        else:
            self.setText(self._object.name)

        layout = qt.QFormLayout()
        self.setButtonLayout(layout)
        self.setSizePolicy(qt.QSizePolicy.Policy.MinimumExpanding, qt.QSizePolicy.Policy.Expanding)

        if not self._is_image_level:
            # Add name row
            layout.addRow(qt.QLabel("Name:"), self._name_lineedit)
            self._name_lineedit.setText(self._object.name)
            self._name_lineedit.textChanged.connect(self._name_changed)

            line = qt.QLineEdit(self._object_class.name)
            line.setReadOnly(True)
            layout.addRow(qt.QLabel("Class:"), line)

        # Attribute rows
        if self._common_object_class is not None:
            for attr_class in self._common_object_class.attributes:
                attr = self._object.get_common_attribute_by_id(attr_class.id)
                value_widget = self._prepare_attribute_widget(attr_class, attr)
                layout.addRow(qt.QLabel(attr_class.name + ":"), value_widget)
        for attr_class in self._object_class.attributes:
            attr = self._object.get_attribute_by_id(attr_class.id)
            value_widget = self._prepare_attribute_widget(attr_class, attr)
            layout.addRow(qt.QLabel(attr_class.name + ":"), value_widget)

        if self._is_image_level:
            return

        # Buttons row
        buttons_widget = qt.QWidget()
        buttons_layout = qt.QHBoxLayout()
        buttons_widget.setLayout(buttons_layout)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        self._colored_icon.setSizePolicy(qt.QSizePolicy.Policy.Expanding, qt.QSizePolicy.Policy.Preferred)
        layout.addRow(self._colored_icon, buttons_widget)
        buttons_layout.addStretch()

        # Add a button to delete the object
        show_object_button = qt.QPushButton()
        buttons_layout.addWidget(show_object_button)
        show_object_button.setText("D")
        show_object_button.setStyleSheet("font-weight: bold;")
        show_object_button.setSizePolicy(qt.QSizePolicy.Policy.Minimum, qt.QSizePolicy.Policy.Minimum)
        show_object_button.setMaximumWidth(show_object_button.sizeHint.height())
        show_object_button.setToolTip("Delete object")
        show_object_button.clicked.connect(lambda: LOAD_STATE_MANAGER.remove_segment_for_object.emit(self._object.id))

        buttons_layout.addStretch()

        # Add a button to show the object
        show_object_button = qt.QPushButton()
        buttons_layout.addWidget(show_object_button)
        show_object_button.setText("L")
        show_object_button.setStyleSheet("font-weight: bold;")
        show_object_button.setSizePolicy(qt.QSizePolicy.Policy.Minimum, qt.QSizePolicy.Policy.Minimum)
        show_object_button.setMaximumWidth(show_object_button.sizeHint.height())
        show_object_button.setToolTip("Show object")
        show_object_button.clicked.connect(lambda: LOAD_STATE_MANAGER.show_object.emit(self._object))

        buttons_layout.addStretch()

        # Add buttons for selecting Scene Graph object as subject / object
        buttons_layout.addWidget(self.set_subject_button)
        self.set_subject_button.setText("S")
        self.set_subject_button.setStyleSheet("font-weight: bold;")
        self.set_subject_button.setSizePolicy(qt.QSizePolicy.Policy.Minimum, qt.QSizePolicy.Policy.Minimum)
        self.set_subject_button.setMaximumWidth(self.set_subject_button.sizeHint.height())
        self.set_subject_button.setToolTip("Set as subject")
        self.set_subject_button.setCheckable(True)
        self.set_subject_button.setChecked(False)
        self.set_subject_button.clicked.connect(self._select_as_subject)

        buttons_layout.addStretch()

        buttons_layout.addWidget(self.set_object_button)
        self.set_object_button.setText("O")
        self.set_object_button.setStyleSheet("font-weight: bold;")
        self.set_object_button.setSizePolicy(qt.QSizePolicy.Policy.Minimum, qt.QSizePolicy.Policy.Minimum)
        self.set_object_button.setMaximumWidth(self.set_object_button.sizeHint.height())
        self.set_object_button.setToolTip("Set as object")
        self.set_object_button.setCheckable(True)
        self.set_object_button.setChecked(False)
        self.set_object_button.clicked.connect(self._select_as_object)

        buttons_layout.addStretch()

        # If we added a new segment, then the color should already be available in the mapping
        # Also the signal would not get triggered after this widget is created, so we need to manually call it
        self._set_color()

    def update_name_widget(self, obj: BoundingBox):
        """Method to call to update the name widget when there was an external change to its value."""
        if obj == self._object and obj.name != self._name_lineedit.text:
            self._name_lineedit.setText(self._object.name)

    def _name_changed(self):
        """Callback for updating the name of the object when the text in the lineedit changes."""
        old_name = self._object.name
        self._object.name = self._name_lineedit.text
        if old_name != self._object.name:
            self.setText(self._object.name)
            LOAD_STATE_MANAGER.selected_patient_scene_graph_modified = True
            LOAD_STATE_MANAGER.name_changed.emit(self._object)

    def _set_color(self):
        """Sets the color of the icon."""
        new_color = LOAD_STATE_MANAGER.colors.get(self._object.id)
        if new_color != self._old_color:
            self._old_color = new_color
            self._colored_icon.setStyleSheet(f"background-color: {new_color};")

    @staticmethod
    def _prepare_attribute_widget(attr_class: AttributeClass, attr: Attribute) -> qt.QWidget:
        def _attribute_changed(attr__: Attribute, value__: Any):
            LOAD_STATE_MANAGER.selected_patient_scene_graph_modified = True
            attr__.value = value__

        # Init the value widget for each possible attribute type
        if isinstance(attr_class, StrAttribute):
            # Str
            value_widget = qt.QLineEdit()
            value_widget.setText(str(attr.value))
            value_widget.textChanged.connect(
                lambda new_text, attr_=attr: _attribute_changed(attr_, new_text))

        elif isinstance(attr_class, IntAttribute):
            # Int
            def _str_to_int(s: str) -> int:
                if not s:
                    return 0
                return int(s)

            value_widget = qt.QLineEdit()
            value_widget.setText(str(attr.value))
            value_widget.setValidator(QIntValidator())
            value_widget.textChanged.connect(
                lambda new_text, attr_=attr: _attribute_changed(attr_, _str_to_int(new_text)))

        elif isinstance(attr_class, FloatAttribute):
            # Float
            def _str_to_float(s: str) -> float:
                # We currently only accept floats with a dot
                if s in ["", "."]:
                    return 0.
                return float(s)

            value_widget = qt.QLineEdit()
            value_widget.setText(str(attr.value))
            val = QDoubleValidator()
            value_widget.setValidator(val)
            # We currently only accept floats with a dot
            loc = QLocale(QLocale.Language.English)
            loc.setNumberOptions(QLocale.NumberOption.RejectGroupSeparator)
            val.setLocale(loc)
            value_widget.textChanged.connect(
                lambda new_text, attr_=attr: _attribute_changed(attr_, _str_to_float(new_text)))

        elif isinstance(attr_class, BoolAttribute):
            # Bool
            value_widget = ResponsiveToggle()
            value_widget.setTristate(False)
            value_widget.setChecked(attr.value)
            value_widget.stateChanged.connect(
                lambda state, attr_=attr: _attribute_changed(attr_, state == Qt.CheckState.Checked))

        elif isinstance(attr_class, EnumAttribute):
            # Enum
            value_widget = qt.QComboBox()
            # If we wish to sort enum values, then we also need to change how we retrieve the value for the callback
            value_widget.addItems(map(str, attr_class.values))
            value_widget.setCurrentText(str(attr.value))  # Handy bool to int
            value_widget.setEditable(False)
            # Use the value list rather than the text to preserve value type
            value_widget.currentIndexChanged.connect(
                lambda idx, attr_=attr, tmp=attr_class: _attribute_changed(attr_, tmp.values[idx]))

        else:
            assert False

        return value_widget

    def _select_as_subject(self):
        """Sets this scene graph object as the subject (or unselects it if already selected)."""
        if LOAD_STATE_MANAGER.subject == self._object:
            LOAD_STATE_MANAGER.subject = None
        else:
            LOAD_STATE_MANAGER.subject = self._object

    def _select_as_object(self):
        """Sets this scene graph object as the object (or unselects it if already selected)."""
        if LOAD_STATE_MANAGER.object == self._object:
            LOAD_STATE_MANAGER.object = None
        else:
            LOAD_STATE_MANAGER.object = self._object

    def _subject_changed(self):
        """Shorthand for _subject_or_object_changed when the subject changes."""
        self.set_subject_button.setChecked(LOAD_STATE_MANAGER.subject == self._object)

    def _object_changed(self):
        """Shorthand for _subject_or_object_changed when the object changes."""
        self.set_object_button.setChecked(LOAD_STATE_MANAGER.object == self._object)

    def _disconnect_signals(self):
        """Disconnects all signals from the widget."""
        LOAD_STATE_MANAGER.name_changed.disconnect(self.update_name_widget)
        LOAD_STATE_MANAGER.update_color.disconnect(self._set_color)
        LOAD_STATE_MANAGER.subject_changed.disconnect(self._subject_changed)
        LOAD_STATE_MANAGER.object_changed.disconnect(self._object_changed)
