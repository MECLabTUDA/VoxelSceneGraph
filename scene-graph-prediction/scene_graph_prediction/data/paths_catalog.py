# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Centralized catalog of paths."""

import os
from typing import Type

from yacs.config import CfgNode

from .datasets import Dataset, RelationDetectionDataset, Split, RatioSpliter, FixedSpliter
from .transforms import Compose


class DatasetCatalog:
    _DATASET_TYPE_KEY = "dataset_type"
    DATASETS_DIR = r"C:\Users\asanner\PycharmProjects\scene-graph-prediction/datasets/"
    CACHE_DIR = os.path.join(DATASETS_DIR, ".cache")

    DATASETS = {
        "INSTANCE2022": {
            "img_dir": DATASETS_DIR + "fed_sgg_full_rot/INSTANCE2022/images",
            "annotation_dir": DATASETS_DIR + "fed_sgg_full_rot/INSTANCE2022/boxlists_wacv",
            "knowledge_graph_file": DATASETS_DIR + "fed_sgg_full_rot/knowledge_graph_fixed.json",
            "spliter": FixedSpliter(DATASETS_DIR + "fed_sgg_full_rot/INSTANCE2022/split.json"),
            _DATASET_TYPE_KEY: RelationDetectionDataset
        },
        "MZ": {
            "img_dir": DATASETS_DIR + "fed_sgg_full_rot/MZ/images",
            "annotation_dir": DATASETS_DIR + "fed_sgg_full_rot/MZ/boxlists_wacv",
            "knowledge_graph_file": DATASETS_DIR + "fed_sgg_full_rot/knowledge_graph_fixed.json",
            "spliter": FixedSpliter(DATASETS_DIR + "fed_sgg_full_rot/MZ/split.json"),
            _DATASET_TYPE_KEY: RelationDetectionDataset
        },
        "BHSD": {
            "img_dir": DATASETS_DIR + "fed_sgg_full_rot/BHSD/images",
            "annotation_dir": DATASETS_DIR + "fed_sgg_full_rot/BHSD/boxlists_wacv",
            "knowledge_graph_file": DATASETS_DIR + "fed_sgg_full_rot/knowledge_graph_fixed.json",
            "spliter": FixedSpliter(DATASETS_DIR + "fed_sgg_full_rot/BHSD/split.json"),
            _DATASET_TYPE_KEY: RelationDetectionDataset
        },
        "CQ500": {
            "img_dir": DATASETS_DIR + "fed_sgg_full_rot/CQ500/images",
            "annotation_dir": DATASETS_DIR + "fed_sgg_full_rot/CQ500/boxlists_wacv",
            "knowledge_graph_file": DATASETS_DIR + "fed_sgg_full_rot/knowledge_graph_fixed.json",
            "spliter": FixedSpliter(DATASETS_DIR + "fed_sgg_full_rot/CQ500/split.json"),
            _DATASET_TYPE_KEY: RelationDetectionDataset
        },
        "INSTANCE2022_rel": {
            "img_dir": DATASETS_DIR + "fed_sgg_full_rot/INSTANCE2022/images",
            "annotation_dir": DATASETS_DIR + "fed_sgg_full_rot/INSTANCE2022/boxlists_wacv",
            "knowledge_graph_file": DATASETS_DIR + "fed_sgg_full_rot/knowledge_graph_fixed.json",
            "spliter": FixedSpliter(DATASETS_DIR + "fed_sgg_full_rot/INSTANCE2022/split_rel_only.json"),
            _DATASET_TYPE_KEY: RelationDetectionDataset
        },
        "MZ_rel": {
            "img_dir": DATASETS_DIR + "fed_sgg_full_rot/MZ/images",
            "annotation_dir": DATASETS_DIR + "fed_sgg_full_rot/MZ/boxlists_wacv",
            "knowledge_graph_file": DATASETS_DIR + "fed_sgg_full_rot/knowledge_graph_fixed.json",
            "spliter": FixedSpliter(DATASETS_DIR + "fed_sgg_full_rot/MZ/split_rel_only.json"),
            _DATASET_TYPE_KEY: RelationDetectionDataset
        },
        "BHSD_rel": {
            "img_dir": DATASETS_DIR + "fed_sgg_full_rot/BHSD/images",
            "annotation_dir": DATASETS_DIR + "fed_sgg_full_rot/BHSD/boxlists_wacv",
            "knowledge_graph_file": DATASETS_DIR + "fed_sgg_full_rot/knowledge_graph_fixed.json",
            "spliter": FixedSpliter(DATASETS_DIR + "fed_sgg_full_rot/BHSD/split_rel_only.json"),
            _DATASET_TYPE_KEY: RelationDetectionDataset
        },
        "CQ500_rel": {
            "img_dir": DATASETS_DIR + "fed_sgg_full_rot/CQ500/images",
            "annotation_dir": DATASETS_DIR + "fed_sgg_full_rot/CQ500/boxlists_wacv",
            "knowledge_graph_file": DATASETS_DIR + "fed_sgg_full_rot/knowledge_graph_fixed.json",
            "spliter": FixedSpliter(DATASETS_DIR + "fed_sgg_full_rot/CQ500/split_rel_only.json"),
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
