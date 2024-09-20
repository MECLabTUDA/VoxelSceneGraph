# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import os
import sys

DEBUG_PRINT_ON = False


def setup_logger(name: str, save_dir: str, distributed_rank: int, filename: str = "log.txt") -> logging.Logger:
    logger = logging.getLogger(name)
    # We have to do this because some library used by Theoden adds a handler to the RootLogger. Fuckers...
    logger.propagate = False
    if logger.hasHandlers():
        # Logger is already set up
        return logger

    if DEBUG_PRINT_ON:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    # Don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
