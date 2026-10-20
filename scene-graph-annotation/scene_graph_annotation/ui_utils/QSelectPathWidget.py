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

import os
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QGroupBox, QLineEdit, QHBoxLayout

from .QSelectPathButton import QSelectPathButton


class _QSelectPathWidget(QGroupBox):
    """A widget for selecting/displaying an image path and a button to signal that the image path is ready."""

    def __init__(self, path_button: QSelectPathButton, title: str):
        super().__init__(title)

        self._folder_button = path_button
        self._line_edit = QLineEdit(os.getcwd())
        path_button.current_path = self._line_edit.text()
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout()
        self.setLayout(layout)

        layout.addWidget(self._folder_button)
        layout.addWidget(self._line_edit)

        self._line_edit.setMinimumWidth(300)
        self._line_edit.textChanged.connect(self._update_button_current_path)

        # Change line edit path
        self._folder_button.path_selected.connect(self._line_edit.setText)

        # Prevent the folder button from being in the Tab Order list for convenience
        self._folder_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)

    def set_text(self, text: str):
        """Sets the text in the line edit and update the current path of the QSelectPathButton."""
        self._line_edit.setText(text)
        self._folder_button.current_path = text

    def get_path(self) -> Path:
        """Returns the path from the line edit as a Path object."""
        return Path(self._line_edit.text())

    def _update_button_current_path(self, text: str):
        """Update the current path of the QSelectPathButton on text change in the QLineEdit."""
        self._folder_button.current_path = text


class QSelectFolderWidget(_QSelectPathWidget):
    """Shorthand for selecting folders."""

    def __init__(self, title: str, is_save: bool = False):
        path_button = QSelectPathButton(is_directory=True, is_save=is_save)
        super().__init__(path_button, title)


class QSelectJsonFileWidget(_QSelectPathWidget):
    """Shorthand for selecting JSON files."""

    def __init__(self, title: str, is_save: bool = False):
        path_button = QSelectPathButton(is_directory=False, is_save=is_save, filters=QSelectPathButton.Filters_Json)
        super().__init__(path_button, title)
