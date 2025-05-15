"""
To compile and install locally run "python setup.py build_ext --inplace".
To install library to Python site-packages run "python -m pip install --use-feature=in-tree-build ."

Copyright 2025 Antoine Sanner, Technical University of Darmstadt, Darmstadt, Germany

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
    name="scene-graph-annotation",
    description="Library for the interactive annotation of Scene Graphs",
    url="https://github.com/MECLabTUDA/VoxelSceneGraph",
    license="Apache2.0",
    packages=["scene_graph_annotation"],
    package_dir={"scene_graph_annotation": "scene_graph_annotation"},
    python_requires=">=3.10",
    install_requires=[
        "numpy<2",
        "matplotlib",
        "PyQt6",
        "nibabel",
        "scipy",
        "scikit-image",
        "pillow",
        "connected-components-3d",
        "opencv-python",
        "SimpleITK",
        "lru-dict",
        "click",
        "click-logging",
        "tqdm",
        "networkx",
        "pyvis",
        "PyQt6-WebEngine-qt6",
        "PyQt6-WebEngine"
    ],
    version="1.0",
    entry_points={
        "console_scripts": [
            "sgann_scene_graph_to_html = scripts.scene_graph_to_html:main",
            "sgann_knowledge_graph_to_html = scripts.knowledge_graph_to_html:main",
        ]
    }
)
