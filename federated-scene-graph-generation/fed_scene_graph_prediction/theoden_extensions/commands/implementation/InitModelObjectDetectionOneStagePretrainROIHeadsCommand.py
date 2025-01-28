from scene_graph_prediction.data.evaluation import EvaluationType, IouType
from scene_graph_prediction.engine.training_script_blobs import build_training_basics
from scene_graph_prediction.modeling.utils.misc import LossComputationCfg
from scene_graph_prediction.utils.checkpoint import DetectronCheckpointer
from theoden import Transferable
from theoden.operations import Command
from theoden.resources import TorchModel


class InitModelObjectDetectionOneStagePretrainROIHeadsCommand(Command, Transferable):
    def execute(self):
        cfg = self.client_rm["cfg"]
        logger = self.client_rm["logger"]
        logger.info("Initializing model for object detection")

        # Build model architecture, optimizer and scheduler
        model, optimizer, scheduler, device = build_training_basics(
            0, False, logger,
            no_grad_list_generator=lambda mod: [mod.rpn, mod.backbone]  # Freeze backbone and one-stage object detector
        )

        self.client_rm["arguments"] = {"iteration": 0}

        # Create checkpointer and load model
        checkpointer = DetectronCheckpointer(cfg, model, optimizer, scheduler, cfg.OUTPUT_DIR)
        self.client_rm["checkpointer"] = checkpointer

        # Figure out what kind of evaluation we need to do
        evaluation_type = EvaluationType.COCO
        if IouType.Segmentation in IouType.build_iou_types(cfg) or cfg.MODEL.REQUIRE_SEMANTIC_SEGMENTATION:
            evaluation_type |= EvaluationType.SemanticSegmentation
        if cfg.TEST.RELATION.COMPUTE_RELATION_UPPER_BOUND:
            evaluation_type |= EvaluationType.SGG

        # FIXME We have to use a wrapper
        _model = TorchModel()
        _model.set_model(model)

        self.client_rm["model"] = _model
        self.client_rm["optimizer"] = optimizer
        self.client_rm["scheduler"] = scheduler
        self.client_rm["device"] = device
        self.client_rm["evaluation_type"] = evaluation_type
        self.client_rm["compute_loss"] = LossComputationCfg(False, True, False)
