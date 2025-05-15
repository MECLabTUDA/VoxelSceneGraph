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
    url="https://github.com/MECLabTUDA/VoxelSceneGraph",
    license="Apache2.0",
    packages=["annotation_database"],
    package_dir={"annotation_database": "annotation_database"},
    python_requires=">=3.10",
    install_requires=[
        "click",
        "click-logging",
        "nibabel",
        "numpy",
        "sqlite-utils"
    ],
    version="1.0",
    entry_points={
        "console_scripts": [
            "sgann_database_rebuild = scripts.database_rebuild:main",
            "sgann_study_add = scripts.study_add:main",
            "sgann_study_delete = scripts.study_delete:main",
            "sgann_study_list = scripts.study_list:main",
            "sgann_study_progress_list = scripts.study_progress_list:main",
            "sgann_study_progress_update = scripts.study_progress_update:main",
        ]
    }
)
