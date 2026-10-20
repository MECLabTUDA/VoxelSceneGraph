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
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QToolButton


class QAlignRightButton(QWidget):
    """
    Button class used for displaying a button on the right on an item in a list view.
    The Wrapper class is required for the AlignRight alignment.
    Otherwise, the button takes the whole row.
    Also, only makes the button appear when the mouse is hovering over the button
    """

    # noinspection PyMethodParameters
    def __init__(self, icon: QIcon, parent=None):
        super().__init__(parent)
        self.button = QToolButton()
        lay = QHBoxLayout(self)
        lay.addWidget(self.button, alignment=Qt.AlignmentFlag.AlignRight)
        lay.setContentsMargins(0, 0, 0, 0)
        self.button.setIcon(icon)
        self.button.setVisible(False)
        self.clicked = self.button.clicked

    # noinspection PyMethodParameters
    def enterEvent(self, a0):
        self.button.setVisible(True)

    # noinspection PyMethodParameters
    def leaveEvent(self, a0):
        self.button.setVisible(False)

    def setToolTip(self, a0: str):
        # Since self is taking the whole row, setting its tooltip masks the tooltip of the list item
        # So we only set the tooltip for the actual button
        self.button.setToolTip(a0)
