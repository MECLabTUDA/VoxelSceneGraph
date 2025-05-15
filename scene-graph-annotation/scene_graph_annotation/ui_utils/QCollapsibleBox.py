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

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QSizePolicy, QToolButton, QStackedWidget


class QCollapsibleBox(QWidget):
    """
    Widget with a collapsible content area underneath.
    Content of the collapsible area can be set using the set_content method.
    The top row has some shading to make it look like a button and can
    also be customized e.g. by adding buttons using the _top_row_layout attribute.
    """

    def __init__(self, title="", start_expanded: bool = False, parent=None):
        super().__init__(parent)
        self._title = title
        self._start_expanded = start_expanded

        # This fixed height was added because without buttons in the top row, the height is 23px and with it's 28
        # So we want to have a harmonized look
        self._top_row_height = 30

        # Shading colors
        self._shading_start = "#f6f7fa"
        self._shading_stop = "#dadbde"

        self._top_row = QWidget()
        self._top_row_layout = QHBoxLayout()
        self._toggle_button = QToolButton()
        self._content_area = QStackedWidget()

        self.init_ui()

        # Note: we cannot do that or call self._toggle_button.click() because the parents are not yet set,
        #       and it causes weird windows to appear
        # if start_expanded:
        #     self._toggle_button.setChecked(False)
        #     self.on_pressed()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        # Set object name so that its stylesheet does not impact any children
        self._top_row.setObjectName("QCollapsibleBoxTopRow")
        layout.addWidget(self._top_row)
        self._top_row.setLayout(self._top_row_layout)
        self._top_row.setFixedHeight(self._top_row_height)
        self._top_row_layout.setContentsMargins(2, 2, 2, 2)
        self._set_top_row_shading(self._shading_start, self._shading_stop)

        self._top_row.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Fixed)
        # Toggle button
        self._top_row_layout.addWidget(self._toggle_button, Qt.AlignmentFlag.AlignLeft)
        self._toggle_button.setText(self._title)
        self._toggle_button.setCheckable(True)
        self._toggle_button.setChecked(self._start_expanded)
        self._toggle_button.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        # Make button transparent to make the shading of the parent visible
        self._toggle_button.setStyleSheet("QToolButton { border: none; background: transparent; }")
        self._toggle_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._toggle_button.setArrowType(Qt.ArrowType.DownArrow if self._start_expanded else Qt.ArrowType.RightArrow)
        self._toggle_button.pressed.connect(self.on_pressed)
        self._toggle_button.setToolTip("Expand" if not self._start_expanded else "Collapse")

        # Content area
        layout.addWidget(self._content_area)
        self._content_area.setVisible(self._start_expanded)
        self._content_area.setContentsMargins(0, 0, 0, 0)

    def set_content(self, content: QWidget):
        """Method used to set the content of the collapsible area."""
        # Set content widget
        if self._content_area.count():
            self._content_area.removeWidget(self._content_area.currentWidget())
        self._content_area.addWidget(content)
        self._content_area.setCurrentWidget(content)

    def on_pressed(self):
        """
        Update the arrow icon and tooltip.
        Note: Not setting this widget (self) to invisible,
              before changing the visibility of the content area can cause flickering.
              Of course, we need to set self visible afterward.
        """
        checked = self._toggle_button.isChecked()
        self.setVisible(False)
        self._toggle_button.setArrowType(Qt.ArrowType.DownArrow if not checked else Qt.ArrowType.RightArrow)
        self._content_area.setVisible(not checked)
        self._toggle_button.setToolTip("Expand" if checked else "Collapse")
        self.setVisible(True)

    def set_title(self, title: str):
        self._title = title
        self._toggle_button.setText(title)

    def _set_top_row_shading(self, shading_start: str, shading_stop: str):
        """Method used to set the shading of the top row. Can be used to make it darker for instance."""
        self._top_row.setStyleSheet(
            f"QWidget#{self._top_row.objectName()}"
            "{ background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, "
            f"stop: 0 {shading_start}, stop: 1 {shading_stop});"
            "}"
        )
