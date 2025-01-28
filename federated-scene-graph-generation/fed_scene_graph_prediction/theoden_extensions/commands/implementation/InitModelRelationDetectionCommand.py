from scene_graph_prediction.data.evaluation import EvaluationType
from scene_graph_prediction.engine.training_script_blobs import build_training_basics
from scene_graph_prediction.modeling.abstractions import AbstractDetector
from scene_graph_prediction.modeling.utils.misc import LossComputationCfg
from scene_graph_prediction.utils.checkpoint import DetectronCheckpointer
from theoden import Transferable
from theoden.operations import Command
from theoden.resources import TorchModel


class InitModelRelationDetectionCommand(Command, Transferable):
    def execute(self):
        cfg = self.client_rm["cfg"]
        logger = self.client_rm["logger"]
        logger.info("Initializing model for relation prediction")

        # Build model architecture, optimizer and scheduler
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

        model, optimizer, scheduler, device = build_training_basics(
            0,
            False,
            logger,
            slow_heads=slow_heads,
            no_grad_list_generator=no_grad_list_generator
        )
        self.client_rm["arguments"] = {"iteration": 0}

        # Create checkpointer and load model
        checkpointer = DetectronCheckpointer(cfg, model, optimizer, scheduler, cfg.OUTPUT_DIR)
        self.client_rm["checkpointer"] = checkpointer

        # FIXME We have to use a wrapper
        _model = TorchModel()
        _model.set_model(model)

        self.client_rm["model"] = _model
        self.client_rm["optimizer"] = optimizer
        self.client_rm["scheduler"] = scheduler
        self.client_rm["device"] = device
        self.client_rm["evaluation_type"] = EvaluationType.SGG
        self.client_rm["compute_loss"] = LossComputationCfg(False, False, True)
