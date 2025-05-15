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
import tempfile
from pathlib import Path
from unittest import TestCase

from scene_graph_annotation.knowledge import RadiologyImageKG
from scene_graph_annotation.logging_handlers import TestingHandler
from scene_graph_annotation.utils.image_utils import find_patients_to_annotate


class TestPathing(TestCase):
    logger = logging.getLogger("parse_id_and_name")
    handler = TestingHandler()
    id_key = "id"
    name_key = "name"

    @classmethod
    def setUpClass(cls):
        cls.logger.addHandler(cls.handler)

    def setUp(self):
        self.handler.purge()

    def test_find_img_seg_pairs_no_match(self):
        img_folder = Path(tempfile.mkdtemp())
        ann_folder = Path(tempfile.mkdtemp())
        try:
            pairs = find_patients_to_annotate(RadiologyImageKG, img_folder, ann_folder)
            self.assertEqual(0, len(pairs))
        finally:
            shutil.rmtree(img_folder)
            shutil.rmtree(ann_folder)

    def test_find_img_seg_pairs_success(self):
        img_folder = Path(tempfile.mkdtemp())
        ann_folder = Path(tempfile.mkdtemp())
        # Create files
        for p in [
            img_folder / "1.nii",
            img_folder / "2.nii",
            img_folder / "3.not_valid",
            ann_folder / "2.json",
            ann_folder / "3.json",
        ]:
            with open(p, "w+"):
                pass

        try:
            pairs = find_patients_to_annotate(RadiologyImageKG, img_folder, ann_folder)
            self.assertEqual(1, len(pairs))
        finally:
            shutil.rmtree(img_folder)
            shutil.rmtree(ann_folder)
