"""
Training script for training a box-head feature extractor head when using a one-stage object detector.

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

import os

from scene_graph_prediction.data.evaluation import EvaluationType, IouType
from scene_graph_prediction.engine.training_script_blobs import build_training_basics, run_test, prepare_basics, \
    run_train
from scene_graph_prediction.utils.checkpoint import DetectronCheckpointer
from scene_graph_prediction.utils.miscellaneous import save_config


def main():
    args, cfg, logger = prepare_basics(log_to_file=True)

    output_config_path = os.path.join(cfg.OUTPUT_DIR, "config.yml")
    logger.info(f"Saving config into: {output_config_path}")
    # Save overloaded model config in the output directory
    save_config(cfg, output_config_path)

    if not cfg.MODEL.ROI_HEADS_ONLY:
        logger.error("cfg.MODEL.ROI_HEADS_ONLY needs to be True for this script to work.")
        exit(1)

    # Build model architecture, optimizer and scheduler
    model, optimizer, scheduler, device = build_training_basics(
        args.local_rank,
        args.distributed,
        logger,
        no_grad_list_generator=lambda mod: [mod.rpn, mod.backbone]
    )

    arguments = {"iteration": 0}

    # Load the model
    checkpointer = DetectronCheckpointer(cfg, model, optimizer, scheduler, cfg.OUTPUT_DIR)
    if checkpointer.has_checkpoint():
        extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT,
                                                  update_schedule=cfg.SOLVER.UPDATE_SCHEDULE_DURING_LOAD)
        arguments.update(extra_checkpoint_data)
    else:
        checkpointer.load(cfg.MODEL.PRETRAINED_ONE_STAGE_DETECTOR_CKPT, with_optim=False)

    # Figure out what kind of evaluation we need to do (we never care about segmentation)
    evaluation_type = EvaluationType.COCO
    if IouType.Segmentation in IouType.build_iou_types(cfg) or cfg.MODEL.REQUIRE_SEMANTIC_SEGMENTATION:
        evaluation_type |= EvaluationType.SemanticSegmentation
    if cfg.TEST.RELATION.COMPUTE_RELATION_UPPER_BOUND:
        evaluation_type |= EvaluationType.SGG

    # Do the main training loop
    model = run_train(
        arguments,
        model,
        optimizer,
        scheduler,
        checkpointer,
        device,
        evaluation_type,
        args.distributed,
        logger
    )

    # Save again after training to add parameters that are used there
    save_config(cfg, output_config_path)

    if not args.skip_test:
        run_test(model, evaluation_type, args.distributed, logger)

    # Save a final time after testing to add parameters that are used there
    save_config(cfg, output_config_path)


if __name__ == "__main__":
    main()
