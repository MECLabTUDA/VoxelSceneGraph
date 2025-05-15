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

import logging
import shutil
import sys
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from PyQt6.QtWidgets import QApplication

from scene_graph_annotation.annotator import MainWindow
from scene_graph_annotation.knowledge import NaturalImageKG
from scene_graph_annotation.logging_handlers import TestingHandler


class TestMainWindow(TestCase):
    logger = logging.getLogger("annotator/MainWindow")
    handler = TestingHandler()
    app = QApplication(sys.argv)
    const_folder = Path(tempfile.mkdtemp())

    @classmethod
    def setUpClass(cls):
        cls.logger.addHandler(cls.handler)

    def setUp(self):
        self.handler.purge()

    @classmethod
    def tearDownClass(cls):
        cls.app.exit()
        shutil.rmtree(cls.const_folder)

    def _test_validate_data_folders(self, path_img: Path, path_seg: Path, display: bool, expected_errors: int):
        m = MainWindow()
        # Invalid path
        with patch.object(m._image_folder_widget, "get_path", lambda: path_img):
            # Existing folder
            with patch.object(m._graph_folder_widget, "get_path", lambda: path_seg):
                m._validate_data_folders(display=display, logger=self.logger, handler=self.handler)
        # Check that we open a QMessage box only when display is True
        self.assertEqual(display, self.handler.display_records_called)
        self.assertEqual(0, self.handler.get_warning_message_count())
        self.assertEqual(expected_errors, self.handler.get_error_message_count())

    def test_validate_data_folders_img_folder_not_exists(self):
        for display in [False, True]:
            self._test_validate_data_folders(Path(">>>>"), self.const_folder, display, 1)
            self.handler.purge()

    def test_validate_data_folders_seg_folder_not_exists(self):
        for display in [False, True]:
            self._test_validate_data_folders(self.const_folder, Path(">>>>"), display, 1)
            self.handler.purge()

    def test_validate_data_folders_both_not_exist(self):
        for display in [False, True]:
            self._test_validate_data_folders(Path(">>>>"), Path(">>>>2"), display, 2)
            self.handler.purge()

    def test_validate_data_folders_no_match(self):
        img_folder = Path(tempfile.mkdtemp())
        sg_folder = Path(tempfile.mkdtemp())
        try:
            for display in [False, True]:
                m = MainWindow()
                m._provisional_knowledge_graph = NaturalImageKG
                with patch.object(m._image_folder_widget, "get_path", lambda: img_folder):
                    # Existing folder
                    with patch.object(m._graph_folder_widget, "get_path", lambda: sg_folder):
                        m._validate_data_folders(display=display, logger=self.logger, handler=self.handler)
                        self.assertEqual(display, self.handler.display_records_called)
                        self.handler.purge()
        finally:
            shutil.rmtree(img_folder)
            shutil.rmtree(sg_folder)
