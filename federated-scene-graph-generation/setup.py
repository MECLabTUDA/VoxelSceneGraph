# !/usr/bin/env python
"""
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
    name="federated-scene-graph-generation",
    description="Library for Federated Voxel Scene Graph experiments",
    url="https://github.com/MECLabTUDA/VoxelSceneGraph",
    license="Apache2.0",
    packages=["federated_scene_graph_prediction"],
    package_dir={"federated_scene_graph_prediction": "federated_scene_graph_prediction"},
    python_requires=">=3.10",
    install_requires=[],
    version="1.0",
    entry_points={
        "console_scripts": [
            "fsgpred_detector_one_stage_pretrain_roi_heads_separately_server = tools.detector_one_stage_pretrain_roi_heads_separately_server:main",
            "fsgpred_detector_pretrain_net = tools.detector_pretrain_net_server:main",
            "fsgpred_multiuse_client = tools.multiuse_client:main",
            "fsgpred_relation_train_net_server = tools.relation_train_net_server:main",
        ]
    }
)
