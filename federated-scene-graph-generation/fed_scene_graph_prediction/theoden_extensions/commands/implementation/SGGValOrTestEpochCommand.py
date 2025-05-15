from typing import Sequence

from scene_graph_prediction.engine.training_script_blobs import run_test, run_val
from scene_graph_prediction.modeling.utils.misc import LossComputationCfg
from scene_graph_prediction.scheduling.lr_scheduler import MetricsAwareScheduler
from theoden import MetricResponse
from theoden.operations import ValidateEpochCommand


class SGGValOrTestEpochCommand(ValidateEpochCommand, implements=ValidateEpochCommand):
    """Perform validation or testing based on the split."""

    def execute(self) -> MetricResponse:
        logger = self.client_rm["logger"]
        logger.info(f"Starting a {self.split} epoch")
        if self.split == "test":
            return self.execute_testing()
        return self.execute_validation()

    def execute_validation(self) -> MetricResponse:
        cfg = self.client_rm["cfg"]
        metrics, losses = run_val(
            self.client_rm["model"].model,
            self.client_rm["val_loaders"],
            self.client_rm["evaluation_type"],
            self.client_rm["compute_loss"] if cfg.TEST.TRACK_VAL_LOSS else LossComputationCfg.none(),
            False,
            self.client_rm["logger"]
        )
        flat_metrics = {}
        for dataset_name, all_metrics in metrics.items():
            for iou_type, all_metrics in all_metrics.items():
                for class_name, metric_dict in all_metrics.items():
                    for metric_name, value in metric_dict.items():
                        if isinstance(value, Sequence):
                            for idx, vv in enumerate(value):
                                flat_metrics[f"{iou_type.value}_{class_name}_{metric_name}{idx}"] = vv
                        else:
                            flat_metrics[f"{iou_type.value}_{class_name}_{metric_name}"] = value
        flat_metrics.update(losses)

        iteration = self.client_rm["arguments"]["iteration"]
        # Add losses for logging
        if cfg.TEST.TRACK_VAL_LOSS:
            self.client_rm["meters"].update(iteration, **losses)

        # Do scheduling
        # Scheduler should be called after optimizer.step() in pytorch>=1.1.0
        # https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
        scheduler = self.client_rm["scheduler"]
        logger = self.client_rm["logger"]
        if isinstance(scheduler, MetricsAwareScheduler):
            if not cfg.TEST.DO_VAL:
                logger.error(
                    f"Validation loss computation is disabled, but the {type(scheduler)} relies on it. Aborting..."
                )
                exit(-1)

            scheduler.step(iteration, -losses["loss"])
            if scheduler.stage_count >= cfg.SOLVER.SCHEDULE.MAX_DECAY_STEP:
                logger.info(f"Trigger MAX_DECAY_STEP at iteration {iteration}.")

        return MetricResponse(flat_metrics, metric_type="val", comm_round=self.communication_round)

    def execute_testing(self) -> MetricResponse:
        metrics = run_test(
            self.client_rm["model"].model,
            self.client_rm["evaluation_type"],
            False,
            self.client_rm["logger"]
        )
        flat_metrics = {
            f"{iou_type.value}_{class_name}_{metric_name}": value
            for dataset_name, all_metrics in metrics.items()
            for iou_type, all_metrics in all_metrics.items()
            for class_name, metric_dict in all_metrics.items()
            for metric_name, value in metric_dict.items()
        }

        return MetricResponse(flat_metrics, metric_type="test", comm_round=self.communication_round)
