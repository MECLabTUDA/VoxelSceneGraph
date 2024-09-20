"""
Basic training script for relation detection.

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

from scene_graph_prediction.data.evaluation import EvaluationType
from scene_graph_prediction.engine.training_script_blobs import build_training_basics, run_train, run_test, \
    prepare_basics
from scene_graph_prediction.modeling.abstractions import AbstractDetector
from scene_graph_prediction.utils.checkpoint import DetectronCheckpointer
from scene_graph_prediction.utils.miscellaneous import save_config


def main():
    args, cfg, logger = prepare_basics(log_to_file=True)

    output_config_path = os.path.join(cfg.OUTPUT_DIR, "config.yml")
    logger.info(f"Saving config into: {output_config_path}")
    # Save overloaded model config in the output directory
    save_config(cfg, output_config_path)

    # Note: we slow down the LR of the layers start with the names in slow_heads
    if cfg.MODEL.ROI_RELATION_HEAD.IS_SLOW_PREDICTOR_HEAD:
        slow_heads = [
            "roi_heads.relation.box_feature_extractor",
            "roi_heads.relation.union_feature_extractor.feature_extractor"
        ]
    else:
        slow_heads = []

    # Build model architecture, optimizer and scheduler
    def no_grad_list_generator(mod: AbstractDetector):
        # Modules that should be always set in eval mode
        # Their eval() method should be called after model.train() is called
        eval_modules = mod.rpn, mod.backbone
        # A one-stage mod will not have a box head
        if hasattr(mod, "box"):
            eval_modules += mod.box,
        if hasattr(mod, "mask"):
            eval_modules += mod.mask,
        return eval_modules

    # Build model architecture, optimizer and scheduler
    model, optimizer, scheduler, device = build_training_basics(
        args.local_rank,
        args.distributed,
        logger,
        slow_heads=slow_heads,
        no_grad_list_generator=no_grad_list_generator
    )
    arguments = {"iteration": 0}

    checkpointer = DetectronCheckpointer(cfg, model, optimizer, scheduler, cfg.OUTPUT_DIR)
    # If there is certain checkpoint in output_dir, load it, else load pretrained detector
    if checkpointer.has_checkpoint():
        extra_checkpoint_data = checkpointer.load(
            None,
            update_schedule=cfg.SOLVER.UPDATE_SCHEDULE_DURING_LOAD,
        )
        arguments.update(extra_checkpoint_data)
    else:
        # Load model from detection training
        # Note: load mapping allows to start training with the weights from the box_head's feature extractor
        load_mapping = {
            "roi_heads.relation.box_feature_extractor": "roi_heads.box.feature_extractor",
            "roi_heads.relation.union_feature_extractor.feature_extractor": "roi_heads.box.feature_extractor"
        }

        if cfg.MODEL.ATTRIBUTE_ON:
            load_mapping["roi_heads.relation.att_feature_extractor"] = "roi_heads.attribute.feature_extractor"
            load_mapping["roi_heads.relation.union_feature_extractor.att_feature_extractor"] = \
                "roi_heads.attribute.feature_extractor"

        checkpointer.load(cfg.MODEL.PRETRAINED_DETECTOR_CKPT, with_optim=False, load_mapping=load_mapping)

    # Do the main training loop
    model = run_train(
        arguments,
        model,
        optimizer,
        scheduler,
        checkpointer,
        device,
        EvaluationType.SGG,
        args.distributed,
        logger
    )

    # Save again after training to add parameters that are used there
    save_config(cfg, output_config_path)

    if not args.skip_test:
        run_test(model, EvaluationType.SGG, args.distributed, logger)

    # Save a final time after testing to add parameters that are used there
    save_config(cfg, output_config_path)


if __name__ == "__main__":
    main()
