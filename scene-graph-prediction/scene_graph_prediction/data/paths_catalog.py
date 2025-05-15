# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Centralized catalog of paths."""
import os
from pathlib import Path
from typing import Type

from yacs.config import CfgNode

from .datasets import Dataset, RelationDetectionDataset, Split, FixedSpliter
from .transforms import Compose


class DatasetCatalog:
    _DATASET_TYPE_KEY = "dataset_type"
    DATASETS_DIR = Path(__file__).parent.parent.parent.as_posix() + "/datasets/"
    CACHE_DIR = os.path.join(DATASETS_DIR, ".cache")

    DATASETS = {
        "INSTANCE2022": {
            "img_dir": DATASETS_DIR + "BleedScene3D/normalized/INSTANCE2022/images",
            "annotation_dir": DATASETS_DIR + "BleedScene3D/normalized/INSTANCE2022/boxlists",
            "knowledge_graph_file": DATASETS_DIR + "BleedScene3D/knowledge_graph.json",
            "spliter": FixedSpliter(DATASETS_DIR + "BleedScene3D/normalized/INSTANCE2022/split.json"),
            _DATASET_TYPE_KEY: RelationDetectionDataset
        },
        "BHSD": {
            "img_dir": DATASETS_DIR + "BleedScene3D/normalized/BHSD/images",
            "annotation_dir": DATASETS_DIR + "BleedScene3D/normalized/BHSD/boxlists",
            "knowledge_graph_file": DATASETS_DIR + "BleedScene3D/knowledge_graph.json",
            "spliter": FixedSpliter(DATASETS_DIR + "BleedScene3D/normalized/BHSD/split.json"),
            _DATASET_TYPE_KEY: RelationDetectionDataset
        },
        "CQ500": {
            "img_dir": DATASETS_DIR + "BleedScene3D/normalized/CQ500/images",
            "annotation_dir": DATASETS_DIR + "BleedScene3D/normalized/CQ500/boxlists",
            "knowledge_graph_file": DATASETS_DIR + "BleedScene3D/knowledge_graph.json",
            "spliter": FixedSpliter(DATASETS_DIR + "BleedScene3D/normalized/CQ500/split.json"),
            _DATASET_TYPE_KEY: RelationDetectionDataset
        },
        "PhysioNet": {
            "img_dir": DATASETS_DIR + "BleedScene3D/normalized/PhysioNet/images",
            "annotation_dir": DATASETS_DIR + "BleedScene3D/normalized/PhysioNet/boxlists",
            "knowledge_graph_file": DATASETS_DIR + "BleedScene3D/knowledge_graph.json",
            "spliter": FixedSpliter(DATASETS_DIR + "BleedScene3D/normalized/PhysioNet/split.json"),
            _DATASET_TYPE_KEY: RelationDetectionDataset
        },
        "HemSeg200": {
            "img_dir": DATASETS_DIR + "BleedScene3D/normalized/HemSeg200/images",
            "annotation_dir": DATASETS_DIR + "BleedScene3D/normalized/HemSeg200/boxlists",
            "knowledge_graph_file": DATASETS_DIR + "BleedScene3D/knowledge_graph.json",
            "spliter": FixedSpliter(DATASETS_DIR + "BleedScene3D/normalized/HemSeg200/split.json"),
            _DATASET_TYPE_KEY: RelationDetectionDataset
        },

        "INSTANCE2022_rel": {
            "img_dir": DATASETS_DIR + "BleedScene3D/normalized/INSTANCE2022/images",
            "annotation_dir": DATASETS_DIR + "BleedScene3D/normalized/INSTANCE2022/boxlists",
            "knowledge_graph_file": DATASETS_DIR + "BleedScene3D/knowledge_graph.json",
            "spliter": FixedSpliter(DATASETS_DIR + "BleedScene3D/normalized/INSTANCE2022/split.json"),
            "keep_only_with_rel": True,
            _DATASET_TYPE_KEY: RelationDetectionDataset
        },
        "BHSD_rel": {
            "img_dir": DATASETS_DIR + "BleedScene3D/normalized/BHSD/images",
            "annotation_dir": DATASETS_DIR + "BleedScene3D/normalized/BHSD/boxlists",
            "knowledge_graph_file": DATASETS_DIR + "BleedScene3D/knowledge_graph.json",
            "spliter": FixedSpliter(DATASETS_DIR + "BleedScene3D/normalized/BHSD/split.json"),
            "keep_only_with_rel": True,
            _DATASET_TYPE_KEY: RelationDetectionDataset
        },
        "CQ500_rel": {
            "img_dir": DATASETS_DIR + "BleedScene3D/normalized/CQ500/images",
            "annotation_dir": DATASETS_DIR + "BleedScene3D/normalized/CQ500/boxlists",
            "knowledge_graph_file": DATASETS_DIR + "BleedScene3D/knowledge_graph.json",
            "spliter": FixedSpliter(DATASETS_DIR + "BleedScene3D/normalized/CQ500/split.json"),
            "keep_only_with_rel": True,
            _DATASET_TYPE_KEY: RelationDetectionDataset
        },
        "PhysioNet_rel": {
            "img_dir": DATASETS_DIR + "BleedScene3D/normalized/PhysioNet/images",
            "annotation_dir": DATASETS_DIR + "BleedScene3D/normalized/PhysioNet/boxlists",
            "knowledge_graph_file": DATASETS_DIR + "BleedScene3D/knowledge_graph.json",
            "spliter": FixedSpliter(DATASETS_DIR + "BleedScene3D/normalized/PhysioNet/split.json"),
            "keep_only_with_rel": True,
            _DATASET_TYPE_KEY: RelationDetectionDataset
        },
        "HemSeg200_rel": {
            "img_dir": DATASETS_DIR + "BleedScene3D/normalized/HemSeg200/images",
            "annotation_dir": DATASETS_DIR + "BleedScene3D/normalized/HemSeg200/boxlists",
            "knowledge_graph_file": DATASETS_DIR + "BleedScene3D/knowledge_graph.json",
            "spliter": FixedSpliter(DATASETS_DIR + "BleedScene3D/normalized/HemSeg200/split.json"),
            "keep_only_with_rel": True,
            _DATASET_TYPE_KEY: RelationDetectionDataset
        }
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
