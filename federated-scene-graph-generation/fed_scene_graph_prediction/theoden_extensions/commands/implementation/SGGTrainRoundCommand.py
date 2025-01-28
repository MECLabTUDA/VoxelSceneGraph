import datetime
import time

import numpy as np
import torch
from tqdm import tqdm

from scene_graph_prediction.utils.comm import reduce_dict
from theoden import MetricResponse
from theoden.operations import TrainRoundCommand


class SGGTrainRoundCommand(TrainRoundCommand, implements=TrainRoundCommand):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.losses_key = "loss"

    def execute(self) -> MetricResponse:
        model = self.client_rm["model"].model
        optimizer = self.client_rm["optimizer"]
        scheduler = self.client_rm["scheduler"]
        cfg = self.client_rm["cfg"]
        device = self.client_rm["device"]
        logger = self.client_rm["logger"]
        meters = self.client_rm["meters"]
        compute_loss = self.client_rm["compute_loss"]
        train_data_loader_iterator = self.client_rm["train_loader_iter"]
        arguments = self.client_rm["arguments"]
        iteration = arguments["iteration"]

        logger.info(f"Starting a training round (iter={iteration})")

        end = time.time()
        max_iter = cfg.SOLVER.MAX_ITER
        step_losses = []

        for _ in tqdm(range(self.num_steps)):
            model.train()
            iteration += 1

            try:
                images, targets, ids = next(train_data_loader_iterator)
            except StopIteration:
                logger.error(f"StopIteration at iteration {iteration} during training.")
                break

            if any(len(target) == 0 for target in targets):
                logger.error(
                    f"Iteration={iteration} || "
                    f"Image Ids used for training {ids} || "
                    f"targets Length={[len(target) for target in targets]}"
                )
                continue
            arguments["iteration"] = iteration

            if not cfg.MODEL.OPTIMIZED_ROI_HEADS_PIPELINE:
                # Defer the device-move to the model
                images = images.to(device)
                targets = [target.to(device) for target in targets]

            try:
                _, loss_dict = model(images, targets, compute_loss=compute_loss)

                # Filter out any information that might have been returned (see doc for LossDict)
                # noinspection PyTypeChecker
                losses: torch.Tensor = sum(
                    loss for key, loss in loss_dict.items()
                    if not key.startswith("_") and loss.numel() > 0
                )
                # Note: currently only used for federated training... use carefully
                # noinspection PyUnusedLocal
                additional_information = {key[1:]: val for key, val in loss_dict.items() if key.startswith("_")}

                # Reduce losses over all GPUs for logging purposes
                # and filter out additional information
                loss_dict_reduced = {
                    k: v
                    for k, v in reduce_dict(loss_dict, average=True).items()
                    if not k.startswith("_") and v.numel() > 0
                }
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                step_losses.append(losses_reduced.detach().cpu().item())

                meters.update(iteration, loss=losses_reduced, **loss_dict_reduced)

                # Catch NaN losses and warn if necessary
                if losses > 0 and not losses.isnan():
                    optimizer.zero_grad()
                    losses.backward()
                    optimizer.step()
                else:
                    logger.warning(f"Loss was NaN at iteration {iteration}:\n{loss_dict}\n")
                    del images
                    del targets
                    del loss_dict
                    del losses
                    del losses_reduced
                    del loss_dict_reduced
                    continue

                batch_time = time.time() - end
                end = time.time()
                meters.update(iteration, time=batch_time)

                # Clear GPU memory before computing any validation loss, predictions...
                del images
                del targets
                del loss_dict
                del losses
                del losses_reduced
                del loss_dict_reduced
                torch.cuda.empty_cache()

                # Because of MetricsAwareScheduler, the val_result should be None if no new validation occurred
                scheduler.step()

            except Exception as e:
                # If anything happens, we still want to save the model to avoid losing the progress
                # checkpointer.save(f"model_{iteration:07d}", **arguments)
                import traceback
                logger.error("".join(traceback.format_tb(e.__traceback__)))
                meters.update(time=time.time())

            # Log some metrics
            eta_seconds = meters.time.global_avg * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            if iteration % cfg.SOLVER.METERS_PERIOD == 0 or iteration == max_iter:
                logger.info(
                    meters.separator.join(
                        [
                            f"eta: {eta_string}",
                            f"iter: {iteration}",
                            f"{meters}",
                            f"lr: {optimizer.param_groups[0]['lr']:.6f}",
                            f"max mem: {torch.cuda.max_memory_allocated() / 1024. / 1024.:.0f}",
                        ]
                    ))

        # FIXME add more detailed losses later
        return MetricResponse(
            metrics={"loss": np.mean(step_losses)}, metric_type="train",
            comm_round=self.communication_round
        )
