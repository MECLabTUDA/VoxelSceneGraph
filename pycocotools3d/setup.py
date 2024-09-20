"""
To compile and install locally run "python setup.py build_ext --inplace".
To install library to Python site-packages run "python -m pip install --use-feature=in-tree-build ."

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

import platform

import numpy as np
from setuptools import setup, Extension

ext_modules = [
    Extension(
        "pycocotools3d._masks.mask",
        sources=["./pycocotools3d/csrc/maskApi.c", "pycocotools3d/_masks/mask.pyx"],
        include_dirs=[np.get_include(), "./pycocotools3d/csrc"],
        extra_compile_args=[] if platform.system() == "Windows" else ["-Wno-cpp", "-Wno-unused-function", "-std=c99"],
    ),
    Extension(
        "pycocotools3d._masks.mask3d",
        sources=["./pycocotools3d/csrc/maskApi3d.c", "pycocotools3d/_masks/mask3d.pyx"],
        include_dirs=[np.get_include(), "./pycocotools3d/csrc"],
        extra_compile_args=[] if platform.system() == "Windows" else ["-Wno-cpp", "-Wno-unused-function", "-std=c99"],
    )
]

setup(
    name="pycocotools3D",
    description="Official APIs for the MS-COCO dataset adapted to 3D",
    url="",
    license="FreeBSD",
    packages=["pycocotools3d"],
    package_dir={"pycocotools3d": "pycocotools3d"},
    python_requires=">=3.10",
    install_requires=[
        "matplotlib>=2.1.0",
        "numpy<2",
        "cython"
    ],
    version="1.0",
    ext_modules=ext_modules
)
