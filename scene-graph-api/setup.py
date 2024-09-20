# !/usr/bin/env python
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

from setuptools import setup

setup(
    name="scene-graph-api",
    description="API library for Voxel Scene Graph",
    url="",
    license="Apache2.0",
    packages=["scene_graph_api"],
    package_dir={"scene_graph_api": "scene_graph_api"},
    python_requires=">=3.10",
    install_requires=[
        "numpy<2",
        "nibabel",
        "SimpleITK",
        "opencv-python",
        "typing-extensions",
        "tqdm",
        "referencing",
        "jsonpointer",
        "jsonschema",
        "Pillow",
        "click",
        "click-logging",
        "connected-components-3d",
        "tqdm"
    ],
    version="1.0",
    entry_points={
        "console_scripts": [
            "sgapi_format_conversion = scripts.format_conversion:main",
            "sgapi_graph_dataset_to_coco = scripts.graph_dataset_to_coco:main",
        ]
    }
)
