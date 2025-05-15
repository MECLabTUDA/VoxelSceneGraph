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

import sys
from unittest import TestCase
from unittest.mock import Mock

from PyQt6.QtWidgets import QApplication

# noinspection PyProtectedMember
from scene_graph_annotation.annotator._custom_widgets import ObjectEditorWidget, UIManager
from scene_graph_annotation.knowledge import ObjectClass, StrAttribute, IntAttribute, FloatAttribute, BoolAttribute, \
    EnumAttribute
from scene_graph_annotation.scene import Attribute, BoundingBox


class TestObjectEditorWidget(TestCase):
    app = QApplication(sys.argv)

    str_attr = StrAttribute(1, "Str")
    int_attr = IntAttribute(2, "Int")
    float_attr = FloatAttribute(3, "Float")
    bool_attr = BoolAttribute(4, "Bool")
    enum_attr = EnumAttribute(5, "Enum", [1, "2", None, 4., False])
    obj_class = ObjectClass(1, "sdf", [str_attr, int_attr, float_attr, bool_attr, enum_attr])
    obj = BoundingBox(
        1, 1, "Name", [[], []],
        # Attribute order here is important for tests (retrieved via index)
        attributes=[
            Attribute(1, "a"), Attribute(2, 2),
            Attribute(3, 3.), Attribute(4, True),
            Attribute(5, None)
        ]
    )

    @classmethod
    def tearDownClass(cls):
        cls.app.exit()

    def test_name_changed(self):
        signals = UIManager()
        w = ObjectEditorWidget(self.obj_class, self.obj, signals)
        mock_callback = Mock()
        signals.name_changed.connect(mock_callback)
        w._name_lineedit.setText("test_name_changed")
        self.assertEqual("test_name_changed", self.obj.name)
        mock_callback.assert_called_once()

    def test_name_changed_other_obj(self):
        signals = UIManager()
        w = ObjectEditorWidget(self.obj_class, self.obj, signals)
        self.obj.name = "test_name_changed_other_obj"
        signals.name_changed.emit(BoundingBox(1, 1, "", [[], []]))
        self.assertNotEqual("test_name_changed_other_obj", w._name_lineedit.text())

    def test_str_attribute_changed(self):
        signals = UIManager()
        w = ObjectEditorWidget(self.obj_class, self.obj, signals)
        mock_callback = Mock()
        signals.attribute_changed.connect(mock_callback)
        w._attr_widgets_by_id[self.str_attr.id].setText("abc")
        self.assertEqual("abc", self.obj.attributes[0].value)
        mock_callback.assert_called_once_with(self.obj, self.obj.attributes[0])

    def test_int_attribute_changed(self):
        signals = UIManager()
        w = ObjectEditorWidget(self.obj_class, self.obj, signals)
        mock_callback = Mock()
        signals.attribute_changed.connect(mock_callback)
        w._attr_widgets_by_id[self.int_attr.id].setText("1")
        self.assertEqual(1, self.obj.attributes[1].value)
        mock_callback.assert_called_once_with(self.obj, self.obj.attributes[1])

    def test_int_attribute_changed_empty(self):
        signals = UIManager()
        w = ObjectEditorWidget(self.obj_class, self.obj, signals)
        mock_callback = Mock()
        signals.attribute_changed.connect(mock_callback)
        w._attr_widgets_by_id[self.int_attr.id].setText("")
        self.assertEqual(0, self.obj.attributes[1].value)
        mock_callback.assert_called_once_with(self.obj, self.obj.attributes[1])

    def test_float_attribute_changed(self):
        signals = UIManager()
        w = ObjectEditorWidget(self.obj_class, self.obj, signals)
        mock_callback = Mock()
        signals.attribute_changed.connect(mock_callback)
        w._attr_widgets_by_id[self.float_attr.id].setText("1")
        self.assertEqual(1., self.obj.attributes[2].value)
        mock_callback.assert_called_once_with(self.obj, self.obj.attributes[2])

    def test_float_attr_empty(self):
        signals = UIManager()
        w = ObjectEditorWidget(self.obj_class, self.obj, signals)
        mock_callback = Mock()
        signals.attribute_changed.connect(mock_callback)
        w._attr_widgets_by_id[self.float_attr.id].setText("")
        self.assertEqual(0., self.obj.attributes[2].value)
        mock_callback.assert_called_once_with(self.obj, self.obj.attributes[2])

    def test_float_attr_just_point(self):
        signals = UIManager()
        w = ObjectEditorWidget(self.obj_class, self.obj, signals)
        mock_callback = Mock()
        signals.attribute_changed.connect(mock_callback)
        w._attr_widgets_by_id[self.float_attr.id].setText(".")
        self.assertEqual(0., self.obj.attributes[2].value)
        mock_callback.assert_called_once_with(self.obj, self.obj.attributes[2])

    def test_bool_attribute_changed(self):
        signals = UIManager()
        w = ObjectEditorWidget(self.obj_class, self.obj, signals)
        mock_callback = Mock()
        signals.attribute_changed.connect(mock_callback)
        w._attr_widgets_by_id[self.bool_attr.id].setChecked(False)
        self.assertEqual(False, self.obj.attributes[3].value)
        mock_callback.assert_called_once_with(self.obj, self.obj.attributes[3])

    def test_enum_attribute_changed(self):
        signals = UIManager()
        w = ObjectEditorWidget(self.obj_class, self.obj, signals)
        for idx, val in enumerate(self.enum_attr.values):
            mock_callback = Mock()
            signals.attribute_changed.connect(mock_callback)
            w._attr_widgets_by_id[self.enum_attr.id].setCurrentIndex(idx)
            self.assertEqual(val, self.obj.attributes[4].value)
            self.assertEqual(type(val), type(self.obj.attributes[4].value))
            mock_callback.assert_called_once_with(self.obj, self.obj.attributes[4])

    def test_attribute_changed_other_obj(self):
        signals = UIManager()
        w = ObjectEditorWidget(self.obj_class, self.obj, signals)
        self.obj.attributes[0].value = "test_attribute_changed_other_obj"
        signals.attribute_changed.emit(BoundingBox(1, 1, "", [[], []]), self.obj.attributes[0])
        self.assertNotEqual("test_attribute_changed_other_obj", w._attr_widgets_by_id[self.str_attr.id].text())

    def test_update_name_widget_no_feedback_loop(self):
        signals = UIManager()
        w = ObjectEditorWidget(self.obj_class, self.obj, signals)
        mock_callback = Mock()
        signals.name_changed.connect(mock_callback)
        self.obj.name = "test_update_name_widget_no_feedback_loop"
        w.update_name_widget(self.obj)
        self.assertEqual("test_update_name_widget_no_feedback_loop", w._name_lineedit.text())
        mock_callback.assert_not_called()

    def test_update_attr_widget_str_attribute_changed_and_no_feedback_loop(self):
        signals = UIManager()
        w = ObjectEditorWidget(self.obj_class, self.obj, signals)
        mock_callback = Mock()
        signals.attribute_changed.connect(mock_callback)
        self.obj.attributes[0].value = "123"
        w.update_attr_widget(self.obj, self.obj.attributes[0])
        self.assertEqual("123", w._attr_widgets_by_id[self.str_attr.id].text())
        mock_callback.assert_not_called()

    def test_update_attr_widget_int_attribute_changed_and_no_feedback_loop(self):
        signals = UIManager()
        w = ObjectEditorWidget(self.obj_class, self.obj, signals)
        mock_callback = Mock()
        signals.attribute_changed.connect(mock_callback)
        self.obj.attributes[1].value = 123
        w.update_attr_widget(self.obj, self.obj.attributes[1])
        self.assertEqual("123", w._attr_widgets_by_id[self.int_attr.id].text())
        mock_callback.assert_not_called()

    def test_update_attr_widget_float_attribute_changed_and_no_feedback_loop(self):
        signals = UIManager()
        w = ObjectEditorWidget(self.obj_class, self.obj, signals)
        mock_callback = Mock()
        signals.attribute_changed.connect(mock_callback)
        self.obj.attributes[2].value = 123.1
        w.update_attr_widget(self.obj, self.obj.attributes[2])
        self.assertEqual("123.1", w._attr_widgets_by_id[self.float_attr.id].text())
        mock_callback.assert_not_called()

    def test_update_attr_widget_bool_attribute_changed_and_no_feedback_loop(self):
        signals = UIManager()
        w = ObjectEditorWidget(self.obj_class, self.obj, signals)
        w._attr_widgets_by_id[self.bool_attr.id].setChecked(False)  # Need to do this to ensure that there is a change
        mock_callback = Mock()
        signals.attribute_changed.connect(mock_callback)
        self.obj.attributes[3].value = True
        w.update_attr_widget(self.obj, self.obj.attributes[3])
        self.assertEqual(True, w._attr_widgets_by_id[self.bool_attr.id].isChecked())
        mock_callback.assert_not_called()

    def test_update_attr_widget_enum_attribute_changed_and_no_feedback_loop(self):
        signals = UIManager()
        w = ObjectEditorWidget(self.obj_class, self.obj, signals)
        w._attr_widgets_by_id[self.bool_attr.id].setChecked(False)  # Need to do this to ensure that there is a change
        mock_callback = Mock()
        signals.attribute_changed.connect(mock_callback)
        self.obj.attributes[4].value = None
        w.update_attr_widget(self.obj, self.obj.attributes[4])
        self.assertEqual(2, w._attr_widgets_by_id[self.enum_attr.id].currentIndex())
        mock_callback.assert_not_called()
