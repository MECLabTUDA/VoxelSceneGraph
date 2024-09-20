import argparse
import datetime
import logging
import os
import time
from collections import defaultdict
from typing import Sequence, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from scene_graph_prediction.config import cfg
from scene_graph_prediction.data import build_test_data_loaders, build_val_data_loaders, build_train_data_loader, \
    save_split
from scene_graph_prediction.data.evaluation import evaluate, IouType, EvaluationType
from scene_graph_prediction.engine import inference
from scene_graph_prediction.modeling.abstractions.detector import AbstractDetector
from scene_graph_prediction.modeling.abstractions.loss import LossDict
from scene_graph_prediction.scheduling import build_lr_scheduler, build_optimizer
from scene_graph_prediction.scheduling.lr_scheduler import MetricsAwareScheduler
from scene_graph_prediction.utils.checkpoint import DetectronCheckpointer
from scene_graph_prediction.utils.collect_env import collect_env_info
from scene_graph_prediction.utils.comm import synchronize, get_rank, reduce_dict
from scene_graph_prediction.utils.config import AccessTrackingCfgNode
from scene_graph_prediction.utils.logger import setup_logger
from scene_graph_prediction.utils.metric_logger import MetricLogger
from scene_graph_prediction.utils.miscellaneous import mkdir


def parse_args(parse_evaluation_types: bool = False) -> argparse.Namespace:
    """Parse args and return them."""
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file", "-c",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    if parse_evaluation_types:
        parser.add_argument(
            "-c"
            "--coco-eval",
            dest="coco_eval_type",
            help="Evaluate COCO-style detection.",
            action="store_true",
        )
        parser.add_argument(
            "-s"
            "--segmentation-eval",
            dest="seg_eval_type",
            help="Evaluate semantic segmentation performance.",
            action="store_true",
        )
        parser.add_argument(
            "-g"
            "--sgg-eval",
            dest="sgg_eval_type",
            help="Evaluate the detection for SGG or the performance upper bound (according to cfg).",
            action="store_true",
        )

    return parser.parse_args()


def build_config(args: argparse.Namespace):
    """Build the config by merging the defaults with the config specified in the args."""

    def recursive_merge(cfg_path: str, base_cfgs_seen: list[str]):
        with open(cfg_path, "r") as f:
            cur_cfg = AccessTrackingCfgNode.load_cfg(f)

        path = cur_cfg.get("BASE_CFG", "")
        if path != "":
            path = os.path.abspath(path)

        if path not in base_cfgs_seen:
            base_cfgs_seen.append(path)
            recursive_merge(path, base_cfgs_seen)
        cfg.merge_from_other_cfg(cur_cfg)

    # Find out whether the BASE_CFG has been overridden by the command line args
    root_cfg_path = args.config_file
    if "BASE_CFG" in args.opts:
        override_path_idx = args.opts.index("BASE_CFG") + 1
        if override_path_idx < len(args.opts):
            root_cfg_path = args.opts[override_path_idx]

    # Recursive load changes from defaults
    recursive_merge(root_cfg_path, ["", os.path.abspath(args.config_file)])
    # Finally override options with commandline arguments
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    cfg.reset_accessed_keys()
    # We're using the main CfgNode with the defaults anyway...
    # return cfg


def build_evaluation_type_from_args(args: argparse.Namespace) -> EvaluationType:
    """Build the appropriate evaluation type based on flags in the args."""
    evaluation_type = EvaluationType(0)
    if args.coco_eval_type:
        evaluation_type |= EvaluationType.COCO
    if args.seg_eval_type:
        evaluation_type |= EvaluationType.SemanticSegmentation
    if args.sgg_eval_type:
        evaluation_type |= EvaluationType.SGG

    return evaluation_type


def prepare_basics(
        parse_evaluation_types: bool = False,
        log_to_file: bool = True,
        filename: str = "log.txt"
) -> tuple[argparse.Namespace, AccessTrackingCfgNode, logging.Logger]:
    """
    Prepare the args, config, and logger.
    Will also create folders as needed, check the #GPUs, collect the env info....
    Note: some scripts require the user to pass evaluation types.
    """

    args = parse_args(parse_evaluation_types)
    num_gpus = int(os.environ.get("WORLD_SIZE", 1))
    args.distributed = num_gpus > 1

    torch.autograd.set_detect_anomaly(True)

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        # noinspection PyUnresolvedReferences
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    build_config(args)
    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("scene_graph_prediction", output_dir if log_to_file else "", get_rank(), filename)
    logger.info(f"Using {num_gpus} GPUs")
    logger.info(args)

    logger.debug("Collecting env info (might take some time)")
    logger.debug("\n" + collect_env_info())

    return args, cfg, logger


def build_training_basics(
        local_rank: int,
        distributed: bool,
        logger: logging.Logger,
        slow_heads: list[str] | None = None,
        no_grad_list_generator: Callable[[AbstractDetector], Sequence[torch.nn.Module]] | None = None
) -> tuple[AbstractDetector, torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler, torch.device]:
    """
    Build the model (freezes parts if required), optimizer, scheduler and target device.
    :param local_rank:
    :param distributed:
    :param logger:
    :param slow_heads: list of names of slow heads (with reduced learning rate).
    :param no_grad_list_generator: if specified, will set the corresponding modules to not requiring gradients.
    """
    model = AbstractDetector.build(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    # We need to set the correct parts of the model as requiring no grad before building the optimizer
    # Otherwise it will not be taken into account
    if no_grad_list_generator:
        set_no_grad(no_grad_list_generator(model))

    # GeneralizedRCC is a Module
    optimizer = build_optimizer(cfg, model, logger, slow_heads=slow_heads, rl_factor=float(cfg.SOLVER.IMS_PER_BATCH))
    scheduler = build_lr_scheduler(cfg, optimizer)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank],
            output_device=local_rank,
            # This should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )

    return model, optimizer, scheduler, device


def run_train(
        arguments: dict,
        model: AbstractDetector,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        checkpointer: DetectronCheckpointer,
        device: torch.device,
        evaluation_type: EvaluationType,
        distributed: bool,
        logger: logging.Logger
) -> AbstractDetector:
    """
    Perform the main training data loading and loop.
    Note: basically everything after the model loading has occurred.
    :param model:
    :param optimizer:
    :param scheduler:
    :param checkpointer:
    :param device:
    :param arguments:
    :param evaluation_type: What kind of evaluation should be performed. Training script dependent.
    :param distributed:
    :param logger:
    :returns: the trained model
    """

    # Check where we are in the training
    max_iter = cfg.SOLVER.MAX_ITER
    start_iter = arguments["iteration"]

    # There's no training and no validation to do, so we skip loading all that useless data
    if start_iter >= max_iter and not cfg.TEST.DO_PRETRAIN_VAL:
        return model

    train_data_loader = build_train_data_loader(cfg, is_distributed=distributed, start_iter=arguments["iteration"])
    val_data_loaders = build_val_data_loaders(cfg, is_distributed=distributed)
    save_split(train_data_loader, val_data_loaders, None, cfg.OUTPUT_DIR, False)

    if cfg.TEST.DO_PRETRAIN_VAL:
        logger.info("Validate before training")
        run_val(model, val_data_loaders, evaluation_type, False, distributed, logger)
        # Check whether we should abort again
        if start_iter >= max_iter:
            return model

    logger.info("Start training")
    meters = MetricLogger(cfg, separator="\n")
    start_training_time = time.time()
    end = time.time()

    update_bar = tqdm(total=(cfg.TEST.VAL_PERIOD if cfg.TEST.DO_VAL else max_iter) - start_iter)
    for iteration, (images, targets, ids) in enumerate(train_data_loader, start_iter):
        model.train()
        iteration += 1

        if any(len(target) == 0 for target in targets):
            logger.error(
                f"Iteration={iteration} || "
                f"Image Ids used for training {ids} || "
                f"targets Length={[len(target) for target in targets]}"
            )
            continue
        arguments["iteration"] = iteration

        # Skip any empty batch
        # Note: this can happen due to knowledge-based sampling
        if len(targets) == 0:
            continue

        if not cfg.MODEL.OPTIMIZED_ROI_HEADS_PIPELINE:
            # Defer the device-move to the model
            images = images.to(device)
            targets = [target.to(device) for target in targets]

        try:
            _, loss_dict = model(images, targets)

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

            meters.update(iteration, loss=losses_reduced, **loss_dict_reduced)

            # Catch NaN losses and warn if necessary
            if losses > 0 and not losses.isnan():
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
            else:
                # logger.warning(f"Loss was NaN at iteration {iteration}:\n{loss_dict}\n")
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
            update_bar.update()

            # Clear GPU memory before computing any validation loss, predictions...
            del images
            del targets
            del loss_dict
            del losses
            del losses_reduced
            del loss_dict_reduced
            torch.cuda.empty_cache()

            # Because of MetricsAwareScheduler, the val_result should be None if no new validation occurred
            val_result = None
            if cfg.TEST.DO_VAL and iteration % cfg.TEST.VAL_PERIOD == 0:
                update_bar.close()  # Close the update bar before starting validation as it will open a new one
                _, val_losses = run_val(
                    model, val_data_loaders, evaluation_type, cfg.TEST.TRACK_VAL_LOSS, distributed, logger
                )
                # If enabled, will compute the loss on the validation data
                if cfg.TEST.TRACK_VAL_LOSS:
                    val_result = -val_losses["loss"]
                    meters.update(iteration, **val_losses)
                # Check if the next training session is shorter
                if iteration + cfg.TEST.VAL_PERIOD > max_iter:
                    update_bar = tqdm(total=max_iter - iteration)
                else:
                    update_bar = tqdm(total=cfg.TEST.VAL_PERIOD)

            # Do scheduling
            # Scheduler should be called after optimizer.step() in pytorch>=1.1.0
            # https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
            if isinstance(scheduler, MetricsAwareScheduler):
                if not cfg.TEST.DO_VAL:
                    logger.error(
                        f"Validation loss computation is disabled, but the {type(scheduler)} relies on it. Aborting..."
                    )
                    exit(-1)

                scheduler.step(iteration, val_result)
                if scheduler.stage_count >= cfg.SOLVER.SCHEDULE.MAX_DECAY_STEP:
                    logger.info(f"Trigger MAX_DECAY_STEP at iteration {iteration}.")
                    break
            else:
                scheduler.step()
        except Exception as e:
            # If anything happens, we still want to save the model to avoid losing the progress
            # checkpointer.save(f"model_{iteration:07d}", **arguments)
            import traceback
            logger.error("".join(traceback.format_tb(e.__traceback__)))
            raise e

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

        # Save the model if needed
        if iteration % cfg.SOLVER.CHECKPOINT_PERIOD == 0:
            checkpointer.save(f"model_{iteration:07d}", **arguments)

    checkpointer.save("model_final", **arguments)
    total_training_time = time.time() - start_training_time
    update_bar.close()
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(f"Total training time: {total_time_str} ({total_training_time / max_iter:.4f} s / it)")

    return model


def run_val(
        model: AbstractDetector,
        val_data_loaders: list[DataLoader],
        evaluation_type: EvaluationType,
        compute_loss: bool,
        distributed: bool,
        logger: logging.Logger
) -> tuple[dict[str, dict[IouType, dict[str, dict[str, float]]]], LossDict]:
    """Run validation given a model and a list of dataloaders."""
    logger.info("Start validating")
    model.eval()

    if distributed:
        model = model.module
    # torch.cuda.empty_cache()  # Emptying the cache causes a crash...

    output_folders: list[str | None] = [None] * len(cfg.DATASETS.VAL)
    dataset_names = cfg.DATASETS.VAL
    val_result = {}
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "latest_validation", dataset_name)
            # noinspection PyTypeChecker
            mkdir(output_folder)
            output_folders[idx] = output_folder

    all_datasets_losses = defaultdict(dict)
    with torch.no_grad():
        for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, val_data_loaders):
            predictions, dataset_losses = inference(
                cfg,
                model,
                data_loader_val,
                dataset_name=dataset_name,
                compute_loss=compute_loss,
                device=cfg.MODEL.DEVICE
            )
            dataset_result = evaluate(
                cfg=cfg,
                dataset=data_loader_val.dataset,
                dataset_name=dataset_name,
                predictions=predictions,
                evaluation_type=evaluation_type,
                output_folder=output_folder,
                logger=logger
            )
            synchronize()

            val_result[dataset_name] = dataset_result
            # Add and rename all losses for the dataset
            for key in dataset_losses.keys():
                if not key.startswith("_"):
                    all_datasets_losses[key][dataset_name] = dataset_losses[key]

    # Aggregate losses across datasets and rename per-dataset losses
    # noinspection PyTypeChecker
    all_losses = {key: np.mean(list(all_datasets_losses[key].values())) for key in all_datasets_losses}
    for key, per_dataset in all_datasets_losses.items():
        for dataset_name, val in per_dataset.items():
            all_losses[f"{dataset_name}_{key}"] = val

    return val_result, all_losses


def run_test(
        model: AbstractDetector,
        evaluation_type: EvaluationType,
        distributed: bool,
        logger: logging.Logger
) -> dict[str, dict[IouType, dict[str, dict[str, float]]]]:
    """
    Run test given a model.
    Note: compared to validation, we'll be doing this only once, so we can load all the data here.
    """
    logger.info("Start of the final test phase\n")
    model.eval()

    if distributed:
        model = model.module
    # torch.cuda.empty_cache()  # Emptying the cache causes a crash...

    output_folders: list[str | None] = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "test", dataset_name)
            # noinspection PyTypeChecker
            mkdir(output_folder)
            output_folders[idx] = output_folder

    data_loaders_test = build_test_data_loaders(cfg, is_distributed=distributed)

    test_result = {}
    with torch.no_grad():
        for output_folder, dataset_name, data_loader_test in zip(output_folders, dataset_names, data_loaders_test):
            save_split(None, None, [data_loader_test], cfg.OUTPUT_DIR, True)
            predictions, _ = inference(
                cfg,
                model,
                data_loader_test,
                dataset_name=dataset_name,
                device=cfg.MODEL.DEVICE,
            )
            dataset_result = evaluate(
                cfg=cfg,
                dataset=data_loader_test.dataset,
                dataset_name=dataset_name,
                predictions=predictions,
                evaluation_type=evaluation_type,
                output_folder=output_folder,
                logger=logger
            )
            synchronize()
            test_result[dataset_name] = dataset_result

    return test_result


def set_no_grad(modules: Sequence[torch.nn.Module]):
    """DO NOT use module.eval(), otherwise all self.training condition will be set to False."""
    for module in modules:
        for _, param in module.named_parameters():
            param.requires_grad = False
