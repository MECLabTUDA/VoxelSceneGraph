"""
Script for optimizing a few hyperparameters related to NMS.

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
import itertools
import logging

import numpy as np
from yacs.config import CfgNode

from scene_graph_prediction.data.evaluation import IouType
from scene_graph_prediction.engine.training_script_blobs import run_test, run_val, prepare_basics, \
    build_evaluation_type_from_args, build_val_data_loaders
from scene_graph_prediction.modeling.abstractions import AbstractDetector
from scene_graph_prediction.modeling.utils.misc import LossComputationCfg
from scene_graph_prediction.utils.checkpoint import DetectronCheckpointer


def metrics_to_score(cfg: CfgNode, per_ds_metrics: dict) -> float:
    """
    Convert a dict with per dataset metrics to a normalized score.
    Relevant metrics are selected based on the config.
    """

    def rec_key_lookup(d: dict, ks: list) -> float:
        ret = d
        for key in ks:
            ret = ret[key]
        return ret

    # Figure out the proper keys to look for
    if cfg.MODEL.RELATION_ON:
        metrics = [
            [IouType.Relations, "Overall", "Recall@8"],
            [IouType.Relations, "Overall", "MeanRecall@8"],
            [IouType.Relations, "Overall", "MeanAveragePrecision@8"],
        ]
    elif cfg.MODEL.MASK_ON:
        metrics = [
            [IouType.Segmentation, "Bleeding", "AP0.30[\"all\"]@25"],
            [IouType.Segmentation, "Bleeding", "AR0.30[\"all\"]@25"],
        ]
    else:
        metrics = [
            ([IouType.Relations, "Overall", "RecallUpperBound@8"], 1),
            ([IouType.Relations, "Overall", "MeanRecallUpperBound@8"], 1),
            ([IouType.BoundingBox, "Overall", "%TP"], 2),
        ]

    return np.mean([rec_key_lookup(per_ds_metrics[ds], m) * w for ds in per_ds_metrics for m, w in metrics])


def main():
    args, cfg, logger = prepare_basics(parse_evaluation_types=True, log_to_file=True, filename="optimization.txt")
    evaluation_type = build_evaluation_type_from_args(args)
    cfg.defrost()

    model = AbstractDetector.build(cfg)
    model.to(cfg.MODEL.DEVICE)

    checkpointer = DetectronCheckpointer(cfg, model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.load()

    logger_muted = logging.getLogger("")
    logger_muted.setLevel(logging.ERROR)
    val_data_loaders = build_val_data_loaders(cfg, args.distributed, )

    # Get score current config
    val_metrics, _ = run_val(
        model, val_data_loaders, evaluation_type, LossComputationCfg.none(), args.distributed, logger_muted
    )
    orig_val_score = best_val_score = metrics_to_score(cfg, val_metrics)
    orig_test_score = metrics_to_score(cfg, run_test(model, evaluation_type, args.distributed, logger_muted))

    # FIXME we need a better way to change parameters in a module
    post_processor = model.rpn.box_selector
    parameters_best = {
        "pre_nms_score_thresh": cfg.MODEL.RETINANET.INFERENCE_TH,
        "nms_thresh": cfg.MODEL.RPN.NMS_THRESH
    }
    parameters = {
        "pre_nms_score_thresh": [0.2, 0.4, 0.5, 0.6, 0.7, 0.8],
        "nms_thresh": [0.2, 0.3]
    }
    # Compute combinations
    keys, values = zip(*parameters.items())
    parameters_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # Use parameters config on predictions and evaluate
    for param_idx, param in enumerate(parameters_combinations):
        logger.info(f"\nTesting configuration ({param_idx + 1}/{len(parameters_combinations)}): {param}")
        for k, v in param.items():
            setattr(post_processor, k, v)

        val_metrics, _ = run_val(
            model, val_data_loaders, evaluation_type, LossComputationCfg.none(), args.distributed, logger_muted
        )

        this_val_score = metrics_to_score(cfg, val_metrics)
        logger.info(f"Config has score {this_val_score:.3f} compared to best score {best_val_score:.3f}")
        if this_val_score > best_val_score:
            logger.info("We have a new best!")
            best_val_score = this_val_score
            parameters_best = param

    # Compute score from dict on val
    for k, v in parameters_best.items():
        setattr(post_processor, k, v)
    final_test_score = metrics_to_score(cfg, run_test(model, evaluation_type, args.distributed, logger))

    # Save best cfg and results and metrics on test
    logger.info(f"Original val score: {orig_val_score:.3f}")
    logger.info(f"Original test score: {orig_test_score:.3f}")
    logger.info(f"Best configuration: {parameters_best}")
    logger.info(f"New val score: {best_val_score:.3f}")
    logger.info(f"New test score: {final_test_score:.3f}")


if __name__ == "__main__":
    main()
