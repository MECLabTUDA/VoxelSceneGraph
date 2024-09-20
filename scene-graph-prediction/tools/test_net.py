"""
Script for running the trained model on the test data. Works for both object, and relation detection.

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

from scene_graph_prediction.engine.training_script_blobs import run_test, prepare_basics, \
    build_evaluation_type_from_args
from scene_graph_prediction.modeling.abstractions import AbstractDetector
from scene_graph_prediction.utils.checkpoint import DetectronCheckpointer


def main():
    args, cfg, logger = prepare_basics(parse_evaluation_types=True, log_to_file=True, filename="test.txt")
    evaluation_type = build_evaluation_type_from_args(args)

    model = AbstractDetector.build(cfg)
    model.to(cfg.MODEL.DEVICE)

    checkpointer = DetectronCheckpointer(cfg, model, save_dir=cfg.OUTPUT_DIR)
    _ = checkpointer.load(cfg.MODEL.WEIGHT)

    run_test(model, evaluation_type, args.distributed, logger)


if __name__ == "__main__":
    main()
