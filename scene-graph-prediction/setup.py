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
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.


import warnings
from pathlib import Path

import torch
from setuptools import setup
# noinspection PyProtectedMember
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME


def get_extensions():
    if CUDA_HOME is None:
        warnings.warn("CUDA_HOME environment variable is not set")

    csrc_dir = Path(__file__).parent.absolute() / "scene_graph_prediction" / "csrc"

    main_file = list(csrc_dir.glob("*.cpp"))
    source_cpu = list((csrc_dir / "cpu").glob("*.cpp"))
    source_cuda = list((csrc_dir / "cuda").glob("*.cu"))

    source = main_file + source_cpu

    extra_cflags = ["-I" + csrc_dir.as_posix()]

    if torch.cuda.is_available() and CUDA_HOME is not None:
        source.extend(source_cuda)
        extra_cflags += ["-DWITH_CUDA"]
        ext_type = CUDAExtension
    else:
        ext_type = CppExtension
    source = list(map(lambda p: p.as_posix(), source))

    ext_modules = [
        ext_type(
            "scene_graph_prediction.layers.c_layers.c_layers",
            source, extra_compile_args={"cxx": extra_cflags}
        )
    ]

    return ext_modules


# noinspection PyTypeChecker
setup(
    name="scene-graph-prediction",
    description="Library for Voxel Scene Graph experiments",
    url="",
    license="Apache2.0",
    packages=["scene_graph_prediction"],
    package_dir={"scene_graph_prediction": "scene_graph_prediction"},
    install_requires=[
        "ninja",
        "yacs",
        "cython",
        "matplotlib",
        "tqdm",
        "pandas",
        "numpy<2",
        "Pillow",
        "h5py",
        "scipy",
        "opencv-python",
        "nibabel",
        "connected-components-3d",
        "scikit-learn",
        "pyyaml"
    ],
    version="1.0",
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension},
    entry_points={
        "console_scripts": [
            "sgpred_boxlist_prediction_to_slicer_rois = tools.boxlist_prediction_to_slicer_rois:main",
            "sgpred_detector_offline_eval = tools.detector_offline_eval:main",
            "sgpred_detector_detector_one_stage_box_feature_extractor_pretrain_net = "
            "tools.detector_one_stage_box_feature_extractor_pretrain_net:main",
            "sgpred_detector_pretrain_net = tools.detector_pretrain_net:main",
            "sgpred_relation_train_net = tools.relation_train_net:main",
            "sgpred_test_net = tools.test_net:main",
        ]
    }
)
