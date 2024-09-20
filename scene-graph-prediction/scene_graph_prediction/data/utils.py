import json
import logging
import os
from pathlib import Path

from scene_graph_api.utils.pathing import remove_suffixes
from torch.utils.data import DataLoader

from .datasets import Dataset, Split


def save_labels(dataset_list: list[Dataset], output_dir: str):
    from scene_graph_prediction.utils.comm import is_main_process
    if not is_main_process():
        return

    logger = logging.getLogger(__name__)

    ids_to_labels = {}
    for dataset in dataset_list:  # type: Dataset
        cat_dict = {idx: dataset.categories[idx] for idx in range(len(dataset.categories))}
        ids_to_labels.update(cat_dict)

    if ids_to_labels:
        labels_file = os.path.join(output_dir, 'labels.json')
        logger.info(f"Saving labels mapping into {labels_file}")
        with open(labels_file, 'w') as f:
            json.dump(ids_to_labels, f, indent=2)


def save_split(
        train_dataset: Dataset | DataLoader | None,
        val_datasets: list[Dataset] | list[DataLoader] | None,
        test_datasets: list[Dataset] | list[DataLoader] | None,
        output_dir: str,
        append: bool = False
):
    from scene_graph_prediction.utils.comm import is_main_process

    if not is_main_process():
        return

    def get_names(datasets: list[Dataset] | list[DataLoader] | None) -> list[list[str]]:
        imgs = []
        if datasets is not None:
            for dataset in datasets:
                if dataset is None:
                    imgs.append([])
                    continue

                if isinstance(dataset, DataLoader):
                    dataset = dataset.dataset
                imgs.append([
                    remove_suffixes(Path(dataset.get_img_info(idx)["file_path"]))
                    for idx in range(len(dataset))
                ])
        return imgs

    logger = logging.getLogger(__name__)
    split = {
        # We always only have one train dataset or None
        Split.TRAIN.value: get_names([train_dataset])[0],
        Split.VAL.value: get_names(val_datasets),
        Split.TEST.value: get_names(test_datasets)
    }
    split_file = os.path.join(output_dir, "split.json")

    if append:
        try:
            with open(split_file, "r") as f:
                loaded_split = json.load(f)
            for key in split:
                split[key] += loaded_split.get(key, [])
        except (IOError, FileNotFoundError, json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Did not find or could not read split file {split_file} when appending split content: {e}")

    logger.info(f"{'Appending' if append else 'Saving'} split into {split_file}.")
    with open(split_file, "w") as f:
        json.dump(split, f)
