import os

from fed_scene_graph_prediction.config import cfg as global_cfg
from scene_graph_prediction.utils.collect_env import collect_env_info
from scene_graph_prediction.utils.comm import get_rank
from scene_graph_prediction.utils.config import AccessTrackingCfgNode
from scene_graph_prediction.utils.logger import setup_logger
from scene_graph_prediction.utils.metric_logger import MetricLogger
from scene_graph_prediction.utils.miscellaneous import mkdir
from theoden import Transferable
from theoden.operations import Command


class InitClientCommand(Command, Transferable):
    def __init__(self, cfg: dict, uuid: str | None = None):
        super().__init__(uuid=uuid)
        self.cfg = cfg  # Main server config node as dict

    def execute(self):
        server_config = AccessTrackingCfgNode(self.cfg)
        # noinspection PyUnresolvedReferences
        client_name = self.client.username

        # Check that the client name can be found in the config
        if client_name not in server_config.CLIENTS:
            raise ValueError(
                f"Client name {client_name} cannot be found in the config ({list(server_config.CLIENTS.keys())})."
            )
        client_options = server_config.CLIENTS[client_name]

        # Build the client config
        server_config.DATASETS.TRAIN = client_options.DATASETS.TRAIN
        server_config.DATASETS.VAL = client_options.DATASETS.VAL
        server_config.DATASETS.TEST = client_options.DATASETS.TEST
        if "SOLVER" in client_options:
            server_config.SOLVER.IMS_PER_BATCH = client_options.SOLVER.IMS_PER_BATCH
        if "TEST" in client_options:
            server_config.TEST.IMS_PER_BATCH = client_options.TEST.IMS_PER_BATCH
        if "OUTPUT_DIR" in client_options:
            server_config.OUTPUT_DIR = client_options.OUTPUT_DIR
        if "MODEL" in client_options:
            server_config.MODEL.DEVICE = client_options.MODEL.DEVICE
        server_config.freeze()
        self.client_rm["cfg"] = server_config

        # Also propagate to scene_graph_prediction's global config
        global_cfg.merge_from_other_cfg(server_config)
        global_cfg.freeze()

        # Prepare the logger and output dir
        output_dir = server_config.OUTPUT_DIR
        if output_dir:
            mkdir(output_dir)

        logger = setup_logger("scene_graph_prediction", output_dir, get_rank(), "log.txt")
        logger.info(f"Using {int(os.environ.get('WORLD_SIZE', 1))} GPUs")

        logger.debug("Collecting env info (might take some time)")
        logger.debug("\n" + collect_env_info())

        self.client_rm["logger"] = logger
        self.client_rm["meters"] = MetricLogger(server_config, separator="\n")
