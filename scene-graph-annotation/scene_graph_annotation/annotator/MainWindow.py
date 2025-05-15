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

import json
import logging
import sys
from pathlib import Path

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSizePolicy, QGridLayout, \
    QPushButton, QMessageBox

from scene_graph_annotation.knowledge import KnowledgeGraph
from scene_graph_annotation.logging_handlers import RecordDisplayHandler
from scene_graph_annotation.ui_utils import QSelectFolderWidget, QSelectJsonFileWidget
from scene_graph_annotation.utils.asset_paths import logo_path
from scene_graph_annotation.utils.pathing import remove_suffixes
from scene_graph_api.utils.image_utils import get_image_paths
from scene_graph_api.utils.parsing import load_json_from_path
from .AnnotationWindow import AnnotationWindow
from ..utils import VERSION


def get_datadir() -> Path:
    """
    Returns a parent directory path
    where persistent application data can be stored.

    # linux: ~/.local/share
    # macOS: ~/Library/Application Support
    # windows: C:/Users/<USER>/AppData/Roaming
    """

    home = Path.home()
    if sys.platform == "win32":
        return home / "AppData/Roaming"
    elif sys.platform == "linux":
        return home / ".local/share"
    elif sys.platform == "darwin":
        return home / "Library/Application Support"
    else:
        return Path(".")


class MainWindow(QMainWindow):
    """
    Main window for the annotator.
    The last working path config (start succeeded) will be saved in a JSON file in the APPDATA folder.
    Components:
    - Welcome message with icon followed by short instructions
    - A Grid view with 2 columns, 4 rows:
        - On the first column, 3 QSelectFileWidget to select the path to the knowledge graph, the image folder,
        and the target folder for scene graphs.
        - On the second column, 2 validation buttons:
          - Validate chosen knowledge graph
          - Check name matching between image and target folder and display number of matches.
            The knowledge graph needs to have been validated to enable these buttons
    - Start button, that will validate the knowledge graph, do file pairing between images, and targets and
    open an annotation window.
    """

    def __init__(self):
        super().__init__()
        self._app_name = "Scene Graph Annotator"

        # Path config save path
        self._config_save_path = get_datadir() / "scene_graph_annotator_paths.json"
        self._knowledge_graph_folder_key = "knowledge_graph_path"
        self._img_folder_key = "images_path"
        self._graph_folder_key = "scene_graph_path"

        self._fixed_size = 530, 380
        self._logo_size = 60, 60
        self._welcome_msg = f"Welcome to the {self._app_name}! (v{VERSION})"
        self._instructions = "To begin, please select the image and segmentations folders. " \
                             "An image and an segmentation with the same filename will automatically be paired. " \
                             "Then select the target folder for saving scene graphs. " \
                             "Finally please select the scene graph knowledge graph file."

        # Path selection widgets
        self._knowledge_graph_file_widget = QSelectJsonFileWidget("Knowledge graph")
        self._image_folder_widget = QSelectFolderWidget("Image folder")
        self._graph_folder_widget = QSelectFolderWidget("Scene graph folder")

        # Validation buttons
        self._validate_knowledge_graph_button = QPushButton("Check\nknowledge\ngraph")
        self._validate_img_seg_folders_button = QPushButton("Check\ndata\nfolders")

        # Start button
        self._start_button = QPushButton("Start annotating")

        # Last knowledge graph loaded when validating
        self._provisional_knowledge_graph: KnowledgeGraph | None = None

        self._windows = []
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(self._app_name)
        window_icon = QIcon(logo_path.as_posix())
        self.setWindowIcon(window_icon)
        self.setFixedSize(*self._fixed_size)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        # Welcome part
        # Logo left and two labels on the right aligned vertically
        welcome_widget = QWidget()
        layout.addWidget(welcome_widget)
        welcome_layout = QHBoxLayout()
        welcome_widget.setLayout(welcome_layout)
        welcome_layout.setContentsMargins(0, 0, 0, 0)
        welcome_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        # Logo
        welcome_logo = QLabel()
        welcome_layout.addWidget(welcome_logo)
        welcome_logo.setPixmap(window_icon.pixmap(QSize(*self._logo_size)))

        # Texts
        welcome_text_widget = QWidget()
        welcome_layout.addWidget(welcome_text_widget)
        welcome_text_layout = QVBoxLayout()
        welcome_text_widget.setLayout(welcome_text_layout)
        welcome_text_layout.setContentsMargins(0, 0, 0, 0)

        welcome_message = QLabel()
        welcome_text_layout.addWidget(welcome_message)
        welcome_message.setText(self._welcome_msg)
        welcome_message.setStyleSheet("font-weight: bold")
        welcome_message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        welcome_message.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        welcome_message.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        # Instructions
        instructions = QLabel()
        welcome_text_layout.addWidget(instructions)
        instructions.setText(self._instructions)
        instructions.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        instructions.setWordWrap(True)
        instructions.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        # Grid layout with path selection widgets and validation buttons
        path_selection_widget = QWidget()
        layout.addWidget(path_selection_widget)
        path_selection_layout = QGridLayout()
        path_selection_widget.setLayout(path_selection_layout)
        path_selection_layout.setContentsMargins(0, 0, 0, 0)
        # First column
        path_selection_layout.addWidget(self._knowledge_graph_file_widget, 1, 1)
        path_selection_layout.addWidget(self._image_folder_widget, 2, 1)
        path_selection_layout.addWidget(self._graph_folder_widget, 3, 1)

        # Second column
        path_selection_layout.addWidget(self._validate_knowledge_graph_button, 1, 2)
        path_selection_layout.addWidget(self._validate_img_seg_folders_button, 2, 2, 2, 2)
        self._validate_knowledge_graph_button.clicked.connect(self._validate_knowledge_graph)
        self._validate_img_seg_folders_button.clicked.connect(self._validate_data_folders)
        self._validate_knowledge_graph_button.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Expanding)
        self._validate_img_seg_folders_button.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Expanding)
        self._validate_knowledge_graph_button.setToolTip("Check that the knowledge graph file can be read and "
                                                         "that its content is valid.")
        self._validate_img_seg_folders_button.setToolTip("Please validate a knowledge graph first, "
                                                         "so that the expected type of image is known and"
                                                         "we can compute matches.")
        self._validate_img_seg_folders_button.setEnabled(False)

        # Start button
        layout.addWidget(self._start_button)
        self._start_button.clicked.connect(self._validate_and_start)

        # Set tab ordering for easy switch between fields using the "Tab" keyboard key
        # by removing the validation buttons from the Tab Order list
        self._validate_img_seg_folders_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._validate_knowledge_graph_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        # Set paths from config if any present
        # FIXME if any other options need saving, implement a proper framework
        def set_text(widget, key):
            path = paths_config.get(key)
            if path is not None:
                widget.set_text(path)

        paths_config = load_json_from_path(self._config_save_path.as_posix())
        if paths_config is not None:
            set_text(self._knowledge_graph_file_widget, self._knowledge_graph_folder_key)
            set_text(self._image_folder_widget, self._img_folder_key)
            set_text(self._graph_folder_widget, self._graph_folder_key)

        self.show()

    # noinspection PyUnusedLocal
    def _validate_knowledge_graph(
            self,
            dummy=None,
            display: bool = True,
            logger: logging.Logger | None = None,
            handler: RecordDisplayHandler | None = None
    ) -> KnowledgeGraph | None:
        """
        Attempt to load the knowledge graph and to validate it.
        Dummy arg because of PyQt using the first arg if there is any.
        """
        if logger is None:
            logger = logging.Logger("annotator/MainWindow/_validate_knowledge_graph")
            handler = RecordDisplayHandler()
            logger.addHandler(handler)

        # Load knowledge graph from json and validate
        knowledge_graph = KnowledgeGraph.load(self._knowledge_graph_file_widget.get_path().as_posix(), logger)
        if knowledge_graph is not None:
            knowledge_graph.validate(logger)
        # Display any warning/error
        if display:
            handler.display_records(self)
        # Else display success message
        if not handler.has_errors() and display:
            # Update image folders button
            self._provisional_knowledge_graph = knowledge_graph
            self._validate_img_seg_folders_button.setToolTip(
                "Check that the folders can be found and that "
                "images and their corresponding Scene Graph can be found."
            )
            self._validate_img_seg_folders_button.setEnabled(True)

            if handler.has_warnings():
                msg = "Despite the warnings, the knowledge graph can be loaded."
            else:
                msg = "No errors or warnings found."
            QMessageBox.about(self, "Success", msg)

        return knowledge_graph

    # noinspection PyUnusedLocal
    def _validate_data_folders(
            self,
            dummy=None,
            display: bool = True,
            logger: logging.Logger | None = None,
            handler: RecordDisplayHandler | None = None
    ):
        """
        Validation of image / Scene Graph pairs found.
        Dummy arg because of PyQt using the first arg if there is any.
        :returns: for each patient string (img path, seg path).
        """
        if logger is None:
            logger = logging.Logger("annotator/MainWindow/_validate_data_folders")
            handler = RecordDisplayHandler()
            logger.addHandler(handler)

        # Check that the image folder exists
        img_path = self._image_folder_widget.get_path()
        if not img_path.is_dir():
            logger.error("The image folder does not exist.")
        # Check that the scene graph folder exists
        sg_path = self._graph_folder_widget.get_path()
        if not sg_path.is_dir():
            logger.error("The Scene Graph folder does not exist.")

        # If any error display and end validation
        if handler.has_errors():
            if display:
                handler.display_records(self)
            return

        # No display => no need to look for matches
        if not display:
            return

        # Remove folders and ext, also careful about files with multiple suffixes
        images = get_image_paths(self._provisional_knowledge_graph, img_path)
        scene_graphs = list(sg_path.glob("*.json"))

        img_names = remove_suffixes(images)
        sg_names = remove_suffixes(scene_graphs)
        matches = set(img_names).intersection(sg_names)

        # Display how many matches were found
        if not matches:
            logger.warning(f"{len(images)} images and {len(scene_graphs)} Scene Graphs were found "
                           f"and there is not a single name match.")
            handler.display_records(self)
        else:
            samples = " or ".join([pat for pat in img_names[:3]])
            QMessageBox.about(
                self,
                "Success",
                f"{len(images)} images and "
                f"{len(scene_graphs)} Scene Graphs were found. "
                f"{len(matches)} matches where found such as {samples}."
            )

    def _validate_and_start(self):
        """
        Check that image / segmentation folders exist.
        Check that the scene graph folder exists / can be created.
        Validate the knowledge graph.
        Open an annotation window.
        """
        logger = logging.Logger("annotator/MainWindow/_validate_and_start")
        handler = RecordDisplayHandler()
        logger.addHandler(handler)

        knowledge_graph = self._validate_knowledge_graph(display=False, logger=logger, handler=handler)
        self._validate_data_folders(display=False, logger=logger, handler=handler)

        if handler.has_errors():
            handler.display_records(self)
            return

        img_folder = self._image_folder_widget.get_path()
        scene_graph_folder = self._graph_folder_widget.get_path()

        # Save the path config
        config = {
            self._img_folder_key: img_folder.as_posix(),
            self._graph_folder_key: scene_graph_folder.as_posix(),
            self._knowledge_graph_folder_key: self._knowledge_graph_file_widget.get_path().as_posix(),
        }
        with open(self._config_save_path, "w") as f:
            json.dump(config, f)

        # Create new annotation window
        annotation_window = AnnotationWindow(
            knowledge_graph,
            img_folder,
            scene_graph_folder,
            parent=self
        )
        # Keep reference to avoid garbage collection by Qt
        self._windows.append(annotation_window)
