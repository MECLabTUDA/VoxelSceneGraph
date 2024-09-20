# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Centralized catalog of paths."""

import os
from typing import Type

from yacs.config import CfgNode

from .datasets import Dataset, RelationDetectionDataset, Split, RatioSpliter, FixedSpliter
from .transforms import Compose


class DatasetCatalog:
    _DATASET_TYPE_KEY = "dataset_type"
    DATASETS_DIR = ""
    CACHE_DIR = os.path.join(DATASETS_DIR, ".cache")

    DATASETS = {
        # ICPR2024 2024 Paper
        "bleed_detec_enlarged_v3": {
            "img_dir": DATASETS_DIR + "bleed_detec/imagesTr",
            "annotation_dir": DATASETS_DIR + "bleed_detec_enlarged_v3/boxListsTr",
            "knowledge_graph_file": DATASETS_DIR + "bleed_detec/template.json",
            "spliter": RatioSpliter(val_ratio=.13, test_ratio=0.),
            _DATASET_TYPE_KEY: RelationDetectionDataset
        },
        "bleed_detec_shrunk_v3": {
            "img_dir": DATASETS_DIR + "bleed_detec/imagesTr",
            "annotation_dir": DATASETS_DIR + "bleed_detec_shrunk_v3/boxListsTr",
            "knowledge_graph_file": DATASETS_DIR + "bleed_detec/template.json",
            "spliter": RatioSpliter(val_ratio=.13, test_ratio=0.),
            _DATASET_TYPE_KEY: RelationDetectionDataset
        },
        "bleed_detec_removed_v3": {
            "img_dir": DATASETS_DIR + "bleed_detec/imagesTr",
            "annotation_dir": DATASETS_DIR + "bleed_detec_removed_v3/boxListsTr",
            "knowledge_graph_file": DATASETS_DIR + "bleed_detec/template.json",
            "spliter": RatioSpliter(val_ratio=.13, test_ratio=0.),
            _DATASET_TYPE_KEY: RelationDetectionDataset
        },
        "bleed_detec_moved_v3": {
            "img_dir": DATASETS_DIR + "bleed_detec/imagesTr",
            "annotation_dir": DATASETS_DIR + "bleed_detec_moved_v3/boxListsTr",
            "knowledge_graph_file": DATASETS_DIR + "bleed_detec/template.json",
            "spliter": RatioSpliter(val_ratio=.13, test_ratio=0.),
            _DATASET_TYPE_KEY: RelationDetectionDataset
        },
        "bleed_detec_added_v3": {
            "img_dir": DATASETS_DIR + "bleed_detec/imagesTr",
            "annotation_dir": DATASETS_DIR + "bleed_detec_added_v3/boxListsTr",
            "knowledge_graph_file": DATASETS_DIR + "bleed_detec/template.json",
            "spliter": RatioSpliter(val_ratio=.13, test_ratio=0.),
            _DATASET_TYPE_KEY: RelationDetectionDataset
        },
        "bleed_v3": {
            "img_dir": DATASETS_DIR + "bleed_detec/imagesTr",
            "annotation_dir": DATASETS_DIR + "bleed_detec_v3/boxListsTr",
            "knowledge_graph_file": DATASETS_DIR + "bleed_detec/template.json",
            "spliter": RatioSpliter(val_ratio=.13, test_ratio=0.),
            _DATASET_TYPE_KEY: RelationDetectionDataset
        },
        "bleed_test_v3": {
            "img_dir": DATASETS_DIR + "bleed_detec_test/imagesTr",
            "annotation_dir": DATASETS_DIR + "bleed_detec_test_v3/boxListsTr",
            "knowledge_graph_file": DATASETS_DIR + "bleed_detec_test/template.json",
            "spliter": RatioSpliter(val_ratio=0., test_ratio=1.),
            _DATASET_TYPE_KEY: RelationDetectionDataset
        },
        "bleed_MZ_v3": {
            "img_dir": DATASETS_DIR + "bleed_detec_MZ/imagesTs",
            "annotation_dir": DATASETS_DIR + "bleed_detec_MZ_v3/boxListsTs",
            "knowledge_graph_file": DATASETS_DIR + "bleed_detec_MZ/template.json",
            "spliter": RatioSpliter(val_ratio=0., test_ratio=1.),
            _DATASET_TYPE_KEY: RelationDetectionDataset
        },
    }

    @staticmethod
    def get(name: str, cfg: CfgNode, transforms: Compose, split: Split) -> Dataset:
        dataset = DatasetCatalog.DATASETS.get(name)
        if dataset is None:
            raise ValueError(f"Dataset not available: {name}")

        # Linter not smart enough
        # noinspection PyTypeChecker
        dataset_type: Type = dataset[DatasetCatalog._DATASET_TYPE_KEY]

        # Get construction arguments (but excluding dataset type)
        dataset_args = dataset.copy()
        del dataset_args[DatasetCatalog._DATASET_TYPE_KEY]

        # Linter not smart enough
        # noinspection PyCallingNonCallable
        return dataset_type(cfg, DatasetCatalog.DATASETS_DIR, transforms, split, **dataset_args)
