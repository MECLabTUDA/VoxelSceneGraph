import json
import math
from abc import ABC, abstractmethod
from enum import Enum
from functools import reduce
from pathlib import Path


class Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class DatasetSpliter(ABC):
    """Given a set of keys and a split, compute the corresponding subset of keys."""

    @abstractmethod
    def __call__(self, keys: list[str], split: Split) -> list[str]:
        """Compute and return the subset of keys corresponding to the split."""


class KFoldSpliter(DatasetSpliter):
    """
    Compute a K-fold cross-validation split.
    Note: 10% of the training data are reserved for validation.
    Note: not a stochastic process.
    """

    def __init__(self, current_fold: int, k: int = 5):
        self.current_fold = current_fold
        self.k = k

    def __call__(self, keys: list[str], split: Split):
        val_percent = .1
        k = self.k

        # Compute test split for each fold
        test_splits = [keys[idx::k] for idx in range(k)]
        # Compute train split for each fold by concatenating all test splits but leaving one out
        train_splits = [
            reduce(
                lambda a, b: a + b,
                # Note: Since the validation cases are taken from the last elements in this list,
                #       we add an offset + modulo such that we get a different validation split for each fold
                #       (and not just 4 times the last elements of fold 4 and 1 time the last elements of fold 5)
                [test_splits[train_idx % k] for train_idx in range(test_idx, k + test_idx) if train_idx != test_idx]
            )
            for test_idx in range(k)
        ]
        # Compute the size of each validation split
        val_sizes = [math.ceil(len(train_splits[idx]) * val_percent) for idx in range(k)]
        # Combine everything
        fold_keys = [
            {
                Split.TRAIN: train_splits[idx][:-val_sizes[idx]],
                Split.VAL: train_splits[idx][-val_sizes[idx]:],
                Split.TEST: test_splits[idx],
            }
            for idx in range(k)
        ]

        return fold_keys[self.current_fold][split]


class RatioSpliter(DatasetSpliter):
    """
    Compute splits based on ratios.
    Note: not a stochastic process.
    """

    def __init__(self, val_ratio: float, test_ratio: float = 0.):
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def __call__(self, keys: list[str], split: Split):
        n = len(keys)
        n_test = int(round(n * self.test_ratio))
        n_val = int(round(n * self.val_ratio))

        match split:
            case Split.TRAIN:
                if n_test + n_val == n:
                    return []
                else:
                    return keys[:-(n_test + n_val)]
            case Split.VAL:
                if n_val == 0:
                    return []
                elif n_test == 0:
                    return keys[-(n_test + n_val):]
                else:
                    return keys[-(n_test + n_val): -n_test]
            case Split.TEST:
                if n_test == 0:
                    return []
                else:
                    return keys[-n_test:]


class FixedSpliter(DatasetSpliter):
    """
    Compute splits based on ratios.
    Note: not a stochastic process.
    """

    def __init__(self, split_path: Path | str, split_list_idx: int | list[int] | None = None):
        self.split_path = split_path
        # Optionally if the split is val or test,
        # the index(es) of the relevant list of paths in a list of list as saved by save_split
        self.split_list_idx = split_list_idx

    def __call__(self, keys: list[str], split: Split):
        if not Path(self.split_path).exists():
            raise FileNotFoundError(f"Could not find the split file {self.split_path}.")

        with open(self.split_path, "r") as f:
            splits = json.load(f)
        splits = splits[split.value]

        if self.split_list_idx is not None:
            # Might have a list of list for val and test
            assert isinstance(splits, list) and splits != Split.TRAIN
            if isinstance(self.split_list_idx, list):
                # Concatenate multiple lists
                splits = reduce(lambda a, b: a + b, [splits[idx] for idx in self.split_list_idx])
            else:
                # Select the correct one
                splits = splits[self.split_list_idx]

        return splits
