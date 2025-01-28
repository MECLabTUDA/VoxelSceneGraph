import os

from scene_graph_prediction.engine.training_script_blobs import build_training_basics
from scene_graph_prediction.utils.checkpoint import DetectronCheckpointer
from scene_graph_prediction.utils.collect_env import collect_env_info
from scene_graph_prediction.utils.config import AccessTrackingCfgNode
from scene_graph_prediction.utils.logger import setup_logger
from scene_graph_prediction.utils.miscellaneous import mkdir
from theoden.operations import Action
from theoden.resources import ResourceManager
from theoden.resources.meta import DictCheckpoint
from theoden.topology import Topology


class InitGlobalModelFromServerAction(Action):
    """Use the server config to load the global model saved on the server."""

    def __init__(self, cfg: AccessTrackingCfgNode) -> None:
        super().__init__()
        self.cfg = cfg

    def perform(self, topology: Topology, resource_manager: ResourceManager):
        """Set the state dict of all clients to a unified state dict.

        Return a successor instruction, that will be executed after the model is initialized.
        This Instruction will select a state dict and distribute it to all clients.

        Args:
            topology (Topology): The topology register.
            resource_manager (ResourceManager): The resource register.

        Returns:
            Instruction: The successor instruction.
        """
        # Prepare the logger and output dir
        cfg = self.cfg
        output_dir = cfg.OUTPUT_DIR
        if output_dir:
            mkdir(output_dir)

        logger = setup_logger("scene_graph_prediction", output_dir, 0, "log.txt")
        logger.info(f"Using {int(os.environ.get('WORLD_SIZE', 1))} GPUs")

        logger.debug("Collecting env info (might take some time)")
        logger.debug("\n" + collect_env_info())
        arguments = {"iteration": 0}

        # Use the config to load the model
        model, optimizer, scheduler, _ = build_training_basics(0, False, logger)
        checkpointer = DetectronCheckpointer(cfg, model, optimizer, scheduler, cfg.OUTPUT_DIR)
        if cfg.MODEL.RELATION_ON:
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
        else:
            # Simple object detection
            if cfg.MODEL.PRETRAINED_DETECTOR_CKPT != "":
                # Load when training the ROI heads separately
                checkpointer.load(cfg.MODEL.PRETRAINED_DETECTOR_CKPT, with_optim=False)
            else:
                extra_checkpoint_data = checkpointer.load(
                    cfg.MODEL.WEIGHT, update_schedule=cfg.SOLVER.UPDATE_SCHEDULE_DURING_LOAD
                )
                arguments.update(extra_checkpoint_data)
        resource_manager["checkpointer"] = checkpointer

        # We don't save those
        checkpointer.optimizer = None
        checkpointer.scheduler = None

        # Register model, optimizer and scheduler
        resource_manager.checkpoint_manager.register_checkpoint(
            resource_type="model",
            resource_key="model",
            checkpoint_key="__global__",
            checkpoint=DictCheckpoint(state_dict=model.state_dict())
        )
