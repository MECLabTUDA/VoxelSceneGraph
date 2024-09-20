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
        # Obj detect full (MICCAI)
        "detect_full_train": {
            "img_dir": DATASETS_DIR + "sgg_full_rot/images",
            "annotation_dir": DATASETS_DIR + "sgg_full_rot/boxLists",
            "knowledge_graph_file": DATASETS_DIR + "sgg_full_rot/template.json",
            "spliter": FixedSpliter(
                DATASETS_DIR + "sgg_full_rot/split.json"
            ),
            _DATASET_TYPE_KEY: RelationDetectionDataset
        },
        "detect_full_eval_INSTANCE": {
            "img_dir": DATASETS_DIR + "sgg_full_rot/images",
            "annotation_dir": DATASETS_DIR + "sgg_full_rot/boxLists",
            "knowledge_graph_file": DATASETS_DIR + "sgg_full_rot/template.json",
            "spliter": FixedSpliter(
                DATASETS_DIR + "sgg_full_rot/split.json",
                0
            ),
            _DATASET_TYPE_KEY: RelationDetectionDataset
        },
        "detect_full_eval_MZ": {
            "img_dir": DATASETS_DIR + "sgg_full_rot/images",
            "annotation_dir": DATASETS_DIR + "sgg_full_rot/boxLists",
            "knowledge_graph_file": DATASETS_DIR + "sgg_full_rot/template.json",
            "spliter": FixedSpliter(
                DATASETS_DIR + "sgg_full_rot/split.json",
                1
            ),
            _DATASET_TYPE_KEY: RelationDetectionDataset
        },

        "sgg_full_train": {
            "img_dir": DATASETS_DIR + "sgg_full_rot/images",
            "annotation_dir": DATASETS_DIR + "sgg_full_rot/boxLists",
            "knowledge_graph_file": DATASETS_DIR + "sgg_full_rot/template.json",
            "spliter": FixedSpliter(
                DATASETS_DIR + "sgg_full_rot/split_rel.json"
            ),
            _DATASET_TYPE_KEY: RelationDetectionDataset
        },
        "sgg_full_eval_INSTANCE": {
            "img_dir": DATASETS_DIR + "sgg_full_rot/images",
            "annotation_dir": DATASETS_DIR + "sgg_full_rot/boxLists",
            "knowledge_graph_file": DATASETS_DIR + "sgg_full_rot/template.json",
            "spliter": FixedSpliter(
                DATASETS_DIR + "sgg_full_rot/split_rel.json",
                0
            ),
            _DATASET_TYPE_KEY: RelationDetectionDataset
        },
        "sgg_full_eval_MZ": {
            "img_dir": DATASETS_DIR + "sgg_full_rot/images",
            "annotation_dir": DATASETS_DIR + "sgg_full_rot/boxLists",
            "knowledge_graph_file": DATASETS_DIR + "sgg_full_rot/template.json",
            "spliter": FixedSpliter(
                DATASETS_DIR + "sgg_full_rot/split_rel.json",
                1
            ),
            _DATASET_TYPE_KEY: RelationDetectionDataset
        },
        # ===========================================================================
        "detect_full_train_INST_ONLY": {
            "img_dir": DATASETS_DIR + "sgg_full_rot/images",
            "annotation_dir": DATASETS_DIR + "sgg_full_rot/boxLists",
            "knowledge_graph_file": DATASETS_DIR + "sgg_full_rot/template.json",
            "spliter": FixedSpliter(
                DATASETS_DIR + "sgg_full_rot/split_INST_only.json"
            ),
            _DATASET_TYPE_KEY: RelationDetectionDataset
        },
        "detect_full_eval_INST_ONLY": {
            "img_dir": DATASETS_DIR + "sgg_full_rot/images",
            "annotation_dir": DATASETS_DIR + "sgg_full_rot/boxLists",
            "knowledge_graph_file": DATASETS_DIR + "sgg_full_rot/template.json",
            "spliter": FixedSpliter(
                DATASETS_DIR + "sgg_full_rot/split_INST_only.json",
                0
            ),
            _DATASET_TYPE_KEY: RelationDetectionDataset
        },
        "detect_full_eval_MZ_ONLY": {
            "img_dir": DATASETS_DIR + "sgg_full_rot/images",
            "annotation_dir": DATASETS_DIR + "sgg_full_rot/boxLists",
            "knowledge_graph_file": DATASETS_DIR + "sgg_full_rot/template.json",
            "spliter": FixedSpliter(
                DATASETS_DIR + "sgg_full_rot/split_INST_only.json",
                1
            ),
            _DATASET_TYPE_KEY: RelationDetectionDataset
        },

        "sgg_full_train_INST_ONLY": {
            "img_dir": DATASETS_DIR + "sgg_full_rot/images",
            "annotation_dir": DATASETS_DIR + "sgg_full_rot/boxLists",
            "knowledge_graph_file": DATASETS_DIR + "sgg_full_rot/template.json",
            "spliter": FixedSpliter(
                DATASETS_DIR + "sgg_full_rot/split_rel_INST_only.json"
            ),
            _DATASET_TYPE_KEY: RelationDetectionDataset
        },
        "sgg_full_train_INST_ONLY_hard": {
            "img_dir": DATASETS_DIR + "sgg_full_rot/images",
            "annotation_dir": DATASETS_DIR + "sgg_full_rot/boxLists",
            "knowledge_graph_file": DATASETS_DIR + "sgg_full_rot/template.json",
            "spliter": FixedSpliter(
                DATASETS_DIR + "sgg_full_rot/split_rel_INST_only_hard.json"
            ),
            _DATASET_TYPE_KEY: RelationDetectionDataset
        },
        "sgg_full_eval_INSTANCE_INST_ONLY": {
            "img_dir": DATASETS_DIR + "sgg_full_rot/images",
            "annotation_dir": DATASETS_DIR + "sgg_full_rot/boxLists",
            "knowledge_graph_file": DATASETS_DIR + "sgg_full_rot/template.json",
            "spliter": FixedSpliter(
                DATASETS_DIR + "sgg_full_rot/split_rel_INST_only.json",
                0
            ),
            _DATASET_TYPE_KEY: RelationDetectionDataset
        },
        "sgg_full_eval_MZ_ONLY": {
            "img_dir": DATASETS_DIR + "sgg_full_rot/images",
            "annotation_dir": DATASETS_DIR + "sgg_full_rot/boxLists",
            "knowledge_graph_file": DATASETS_DIR + "sgg_full_rot/template.json",
            "spliter": FixedSpliter(
                DATASETS_DIR + "sgg_full_rot/split_rel_INST_only.json",
                1
            ),
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
