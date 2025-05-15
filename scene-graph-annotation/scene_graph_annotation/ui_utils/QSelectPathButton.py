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

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QToolButton, QFileDialog, QDialog

from scene_graph_annotation.knowledge import RadiologyImageKG, NaturalImageKG
from scene_graph_annotation.utils.asset_paths import folder_icon
from scene_graph_api.utils.image_utils import supported_extensions_by_type


def _format_supported_extensions(name: str, extensions: list[str]) -> str:
    """Formats a list of extensions to a filter string."""
    return f"{name} (" + " ".join(f"*{ext}" for ext in extensions) + ")"


class QSelectPathButton(QToolButton):
    """
    A QButton that takes that opens a file selection Dialog Box.
    If no text is supplied the folder icon will be used.
    See constructor for options.
    """

    # Signal used to emit the path that was selected
    path_selected = pyqtSignal(str)
    # Signal used to emit the pathS that were selected, please select the correct one depending on your needs
    paths_selected = pyqtSignal(list)

    Filters_CSV = ["CSV (*.csv)"]
    Filters_Dicom = ["DICOM (*.dcm)"]
    Filters_Images = [_format_supported_extensions(
        "Image",
        supported_extensions_by_type[NaturalImageKG.get_graph_type()]
    )]
    Filters_Json = ["JSON (*.json)"]
    Filters_Nifti = [_format_supported_extensions(
        "Nifti",
        supported_extensions_by_type[RadiologyImageKG.get_graph_type()]
    )]

    def __init__(
            self,
            current_path: str = "",
            text: str | None = None,
            filters: list[str] | None = None,
            is_save: bool = True,
            multiple_files: bool = False,
            is_directory: bool = False,
    ):
        """
        :param current_path: if this path exists, open the file dialog to this path. Otherwise, default to the cwd.
        :param text: the text displayed on the button. If none is supplied, the folder icon will be used.
        :param filters: the file extension filter. See class attributes.
        :param is_save: whether we are selecting a path for saving a file (changes the text of the validation button).
        :param multiple_files: whether WHEN LOADING, we accept multiple paths.
        :param is_directory: whether the target is a directory (instead of a file).
        """
        super().__init__()
        # Check for incompatibilities
        assert (filters is not None) ^ is_directory
        assert not (multiple_files and is_directory)
        assert not (multiple_files and is_save)

        self.current_path = current_path

        if text is None:
            self.setIcon(QIcon(folder_icon.as_posix()))
        else:
            self.setText(text)
        self.filters = filters
        self.is_save = is_save
        self.multiple_files = multiple_files
        self.is_directory = is_directory

        self.setToolTip("Open file selection window")
        self.clicked.connect(self._open_file_dialog)

    def _open_file_dialog(self):
        """Handler for setting up the file dialog window."""
        dialog = QFileDialog()

        # Changes the validation button text
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave if self.is_save else QFileDialog.AcceptMode.AcceptOpen)

        # Select the path to open the dialog to
        current_path = Path(self.current_path)
        # Mini QoL: if the path does not exist, check if the parent maybe does exist
        if not current_path.exists():
            current_path = current_path.parent
        # If the path exists, open to this folder (or the folder containing the file)
        if current_path.exists():
            if not current_path.is_dir():
                current_path = current_path.parent
            dialog.setDirectory(current_path.absolute().as_posix())
        else:
            dialog.setDirectory(os.getcwd())

        # Set filter for type of file expected: one, many, directory...
        if self.is_directory:
            dialog.setFileMode(QFileDialog.FileMode.Directory)
        elif self.is_save:
            dialog.setFileMode(QFileDialog.FileMode.AnyFile)
        elif not self.multiple_files:
            dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        else:
            dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)

        # Sets the file extension filter
        dialog.setViewMode(QFileDialog.ViewMode.Detail)
        if self.filters:
            dialog.setNameFilters(self.filters)

        # Emits the selected path(s) using the correct signal
        if dialog.exec() == QDialog.DialogCode.Accepted:
            if self.multiple_files:
                self.paths_selected.emit(dialog.selectedFiles())
            else:
                path = dialog.selectedFiles()[0]
                self.path_selected.emit(path)
