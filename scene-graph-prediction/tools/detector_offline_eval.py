"""
Run evaluation from saved prediction.

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

import logging
import os
from pathlib import Path

from yacs.config import CfgNode

from scene_graph_api.utils.pathing import remove_suffixes
from scene_graph_prediction.data import build_test_data_loaders, build_val_data_loaders
from scene_graph_prediction.data.evaluation import evaluate, EvaluationType
from scene_graph_prediction.data.evaluation.sgg_eval import SGG_EVAL_RESULTS_FILE, SGG_EVAL_RESULTS_DICT_FILE
from scene_graph_prediction.engine.training_script_blobs import prepare_basics, build_evaluation_type_from_args
from scene_graph_prediction.structures import BoxList


def run_val(cfg: CfgNode, evaluation_type: EvaluationType, logger: logging.Logger):
    val_data_loaders = build_val_data_loaders(cfg, is_distributed=False)

    logger.info("Evaluation on val datasets\n")

    output_folders: list[str | None] = [None] * len(cfg.DATASETS.VAL)
    dataset_names = cfg.DATASETS.VAL
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "latest_validation", dataset_name)
            if os.path.exists(output_folder):
                output_folders[idx] = output_folder

    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, val_data_loaders):
        if output_folder is None:
            logger.warning(f"Missing folder for val dataset {dataset_name}")
            continue

        # Compute reverse mapping as we cannot rely on any name ordering
        mapping = {
            remove_suffixes(Path(data_loader_val.dataset.get_img_info(idx)["file_path"])): idx
            for idx in range(len(data_loader_val.dataset))
        }

        predictions = {
            mapping[remove_suffixes(path)]: BoxList.load(path)
            for idx, path in enumerate(sorted(
                [path for path in Path(output_folder).glob("*.pth")
                 if path.name not in [SGG_EVAL_RESULTS_FILE, SGG_EVAL_RESULTS_DICT_FILE]]
            ))
        }
        evaluate(
            cfg=cfg,
            dataset=data_loader_val.dataset,
            dataset_name=dataset_name,
            predictions=predictions,
            output_folder=None,  # As not to save the results again
            evaluation_type=evaluation_type,
            logger=logger
        )


# noinspection DuplicatedCode,PyShadowingNames
def run_test(cfg: CfgNode, evaluation_type: EvaluationType, logger: logging.Logger):
    logger.info("Evaluation on test datasets")

    if not os.path.exists(os.path.join(cfg.OUTPUT_DIR, "test")):
        logger.warning("The test dataset folder with predictions does not exist.")
        return

    output_folders: list[str | None] = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "test", dataset_name)
            if os.path.exists(output_folder):
                output_folders[idx] = output_folder

    data_loaders_test = build_test_data_loaders(cfg)

    for output_folder, dataset_name, data_loader_test in zip(output_folders, dataset_names, data_loaders_test):
        if output_folder is None:
            logger.warning(f"Missing folder for test dataset {dataset_name}")
            continue
        logger.info(f"\nEvaluate dataset {dataset_name}\n")

        # Compute reverse mapping as we cannot rely on any name ordering
        mapping = {
            remove_suffixes(Path(data_loader_test.dataset.get_img_info(idx)["file_path"])): idx
            for idx in range(len(data_loader_test.dataset))
        }

        predictions = {
            mapping[remove_suffixes(path)]: BoxList.load(path)
            for idx, path in enumerate(sorted(
                [path for path in Path(output_folder).glob("*.pth")
                 if path.name not in [SGG_EVAL_RESULTS_FILE, SGG_EVAL_RESULTS_DICT_FILE]]
            ))
        }

        evaluate(
            cfg=cfg,
            dataset=data_loader_test.dataset,
            dataset_name=dataset_name,
            predictions=predictions,
            output_folder=None,  # As not to save the results again
            evaluation_type=evaluation_type,
            logger=logger
        )


def main():
    args, cfg, logger = prepare_basics(parse_evaluation_types=True, log_to_file=True, filename="eval.txt")
    evaluation_type = build_evaluation_type_from_args(args)

    run_val(cfg, evaluation_type, logger)
    if not args.skip_test:
        run_test(cfg, evaluation_type, logger)


if __name__ == "__main__":
    main()
