import json
from argparse import Namespace
from pathlib import Path

import click

from fed_scene_graph_prediction.config import cfg
from fed_scene_graph_prediction.registries import AGGREGATORS
from fed_scene_graph_prediction.theoden_extensions.actions import InitGlobalModelFromServerAction
from fed_scene_graph_prediction.theoden_extensions.commands import InitModelObjectDetectionCommand, \
    LoadTrainValDatasetCommand, InitClientCommand, SceneGraphPredictionInstructionBundle
from scene_graph_prediction.engine.training_script_blobs import build_config
from theoden import start_server
from theoden.operations import ClosedDistribution, RequireNumberOfClientsCondition, ExitRunCommand, SequentialCommand
import shutil


@click.command()
@click.option("-c", "--config-file",
              type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path),
              required=True, help="Path to the config file.")
@click.option("-n", "--run-name",
              type=str, required=False, default="", help="Name for this experiment run.")
@click.option("--allow-deprecated-options", "allow_deprecated",
              is_flag=True, help="Allow unknown options when loading a (old) config. Use with care...")
def main(config_file: Path, run_name: str, allow_deprecated: bool) -> int:
    # Load the config (in-place)
    build_config(Namespace(config_file=config_file, opts=[], allow_deprecated=allow_deprecated))

    # Save the config in the target directory
    Path(cfg.OUTPUT_DIR).mkdir(exist_ok=True, parents=True)
    shutil.copy(config_file, Path(cfg.OUTPUT_DIR) / "config.yml")

    # Figure out how many clients we have
    n_clients = len(cfg.CLIENTS)
    if n_clients == 0:
        raise RuntimeError("No clients configured!")

    steps_per_round = cfg.FEDERATED_LEARNING.STEPS_PER_ROUND
    aggregator = AGGREGATORS[cfg.FEDERATED_LEARNING.AGGREGATOR]()

    start_server(
        instructions=[
            InitGlobalModelFromServerAction(cfg),
            ClosedDistribution(
                commands=[SequentialCommand([
                    InitClientCommand(json.loads(json.dumps(cfg))),  # Ugly recursive conversion to dict
                    InitModelObjectDetectionCommand(),
                    LoadTrainValDatasetCommand()
                ])]
            ),
            # FIXME we do not support TEST.DO_PRETRAIN_VAL or SOLVER.CHECKPOINT_PERIOD
            #  We also do not support optimizer / scheduler / sampler saving
            #  While continuing an existing training run, we load arguments and iteration, but it does not have an impact on the training commands issued
            SceneGraphPredictionInstructionBundle(
                n_rounds=cfg.SOLVER.MAX_ITER // steps_per_round,
                steps_per_round=steps_per_round,
                aggregator=aggregator,
                num_workers=cfg.DATALOADER.NUM_WORKERS,
                validate_every_n_rounds=cfg.TEST.VAL_PERIOD // steps_per_round,
                final_validation=True,
                only_grad=False
            ),
            ClosedDistribution(
                commands=[ExitRunCommand()]
            )
        ],
        permanent_conditions=[RequireNumberOfClientsCondition(n_clients)],
        run_name=run_name,
        global_context=cfg.FEDERATED_LEARNING.GLOBAL_CONTEXT if cfg.FEDERATED_LEARNING.GLOBAL_CONTEXT else None,
        communication_address=cfg.FEDERATED_LEARNING.COMMUNICATION_ADDRESS if cfg.FEDERATED_LEARNING.COMMUNICATION_ADDRESS else None,
        exit_on_finish=True,
        timeout=3600
    )
    return 0


if __name__ == "__main__":
    exit(main())
