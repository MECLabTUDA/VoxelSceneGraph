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
import shutil
import tempfile
from unittest import TestCase

import nibabel as nib
import numpy as np

# noinspection PyProtectedMember
from scene_graph_api.utils.nifti_io import NiftiImageWrapper


class TestNiftiIo(TestCase):
    temp_folder = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_folder)

    def test_nifti_image_to_gzipped_str(self):
        array = (np.random.random((100, 100, 10)) * 255).astype(np.uint8)
        img = NiftiImageWrapper(nib.Nifti1Image(array, np.random.random((4, 4))), True)
        path = os.path.join(self.temp_folder, "test_nifti_image_to_gzipped_str_no_corruption.nii.gz")
        with open(path, "wb") as f:
            f.write(img.to_str().encode(NiftiImageWrapper._encoding))
        img2 = nib.load(path)
        self.assertTrue(np.allclose(img.affine[:-1], img2.affine[:-1]))  # ignore last row as it's reset to [0, 0, 0, 1]
        # noinspection PyTypeChecker
        self.assertTrue(np.allclose(img.get_fdata(), img2.get_fdata()))

    def test_gzipped_str_to_nifti_image(self):
        array = np.random.randint(0, 100, (100, 100, 10), dtype=np.int32)
        affine = np.random.random((4, 4))
        img = NiftiImageWrapper(nib.Nifti1Image(array, affine), True)
        img2 = NiftiImageWrapper.from_str(img.to_str())
        self.assertTrue(np.allclose(img.affine[:-1], img2.affine[:-1]))  # ignore last row as it's reset to [0, 0, 0, 1]
        # noinspection PyTypeChecker
        self.assertTrue(np.allclose(img.get_fdata(), img2.get_fdata()))
