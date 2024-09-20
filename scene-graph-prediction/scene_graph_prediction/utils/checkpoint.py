# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
from typing import Any

import torch
from yacs.config import CfgNode

from scene_graph_prediction.utils.c2_model_loading import load_c2_format
from scene_graph_prediction.utils.model_serialization import load_state_dict
from scene_graph_prediction.utils.model_zoo import cache_url
from .comm import is_main_process
from pathlib import Path


class Checkpointer:
    LAST_CKPT = "last_checkpoint"

    def __init__(
            self,
            model: torch.nn.Module,
            optimizer: torch.optim.Optimizer | None = None,
            scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
            save_dir: str | Path = "",
            logger: logging.Logger | None = None,
    ):

        if logger is None:
            logger = logging.getLogger(__name__)

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = Path(save_dir)

        self.logger = logger

    def save(self, filename: str, **kwargs):
        """
        Does nothing if self.save_dir == ""
        :param filename: the target filename (witt or without the ".pth" at the end)
        :param kwargs: any other key/value pairs that need to be saved
        """
        if not self.save_dir or not is_main_process():
            return

        data = {"model": self.model.state_dict()}
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.state_dict()
        data.update(kwargs)

        if not filename.endswith(".pth"):
            filename += ".pth"

        save_file = (self.save_dir / filename).absolute()
        self.logger.info(f"Saving checkpoint to {save_file.as_posix()}")
        torch.save(data, save_file)
        self.tag_last_checkpoint(save_file)

    def load(
            self,
            filename: Path | str | None = None,
            with_optim: bool = True,
            update_schedule: bool = False,
            load_mapping: dict | None = None
    ) -> dict:
        """
        :param filename: is optional and is overriden if self.has_checkpoint() is True
        :param load_mapping:
        :param update_schedule:
        :param with_optim:
        """
        if not load_mapping:
            load_mapping = {}

        if self.has_checkpoint() and filename is None:
            # override argument with existing checkpoint
            filename = self.get_checkpoint_file()
        if not filename:
            # no checkpoint could be found
            self.logger.info("No checkpoint found. Initializing model from scratch.")
            return {}
        self.logger.info(f"Loading checkpoint from {filename}")
        checkpoint = self._load_file(filename)
        self._load_model(checkpoint, load_mapping)
        if with_optim:
            if "optimizer" in checkpoint and self.optimizer:
                self.logger.info(f"Loading optimizer")
                self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
            if "scheduler" in checkpoint and self.scheduler:
                self.logger.info(f"Loading scheduler")
                # noinspection PyTestUnpassedFixture
                self.scheduler.load_state_dict(checkpoint.pop("scheduler"))
            if self.scheduler and update_schedule:
                # Note: all PyTorch schedulers actually have the "last_epoch" attribute
                self.scheduler.last_epoch = checkpoint["iteration"]

        # return any further checkpoint data
        return checkpoint

    def has_checkpoint(self) -> bool:
        return (self.save_dir / self.LAST_CKPT).exists()

    def get_checkpoint_file(self) -> str | None:
        """Returns None if there was an error."""
        try:
            with open(self.save_dir / self.LAST_CKPT, "r") as f:
                last_saved = Path(f.read().strip())
            # For backward compatibility with older path formats (which are already absolute rather than relative)
            if not last_saved.is_absolute():
                last_saved = self.save_dir / last_saved
            last_saved = last_saved.as_posix()
        except IOError:
            # If the file doesn't exist, maybe because it has just been deleted by a separate process
            last_saved = None
        return last_saved

    def tag_last_checkpoint(self, last_filename: str | Path):
        """Writes to self.save_dir / self.LAST_CKPT the name of the last checkpoint file."""
        with open(self.save_dir / self.LAST_CKPT, "w") as f:
            f.write(Path(last_filename).relative_to(self.save_dir.resolve()).as_posix())

    def _load_file(self, f):
        return torch.load(f, map_location=torch.device("cpu"))

    def _load_model(self, checkpoint, load_mapping):
        load_state_dict(self.model, checkpoint.pop("model"), load_mapping)


class DetectronCheckpointer(Checkpointer):
    def __init__(
            self,
            cfg: CfgNode,
            model: torch.nn.Module,
            optimizer: torch.optim.Optimizer | None = None,
            scheduler: Any = None,
            save_dir: str = "",
            logger: logging.Logger | None = None
    ):
        super().__init__(model, optimizer, scheduler, save_dir, logger)
        self.cfg = cfg.clone()

    def _load_file(self, path: str) -> dict:
        # download url files
        if path.startswith("http"):
            # if the file is an url path, download it and cache it
            cached_f = cache_url(path)
            self.logger.info(f"Irl {path} cached in {cached_f}")
            path = cached_f
        # convert Caffe2 checkpoint from pkl
        if path.endswith(".pkl"):
            return load_c2_format(self.cfg, path)
        # load native detectron.pytorch checkpoint
        loaded = super()._load_file(path)
        if "model" not in loaded:
            loaded = dict(model=loaded)
        return loaded
