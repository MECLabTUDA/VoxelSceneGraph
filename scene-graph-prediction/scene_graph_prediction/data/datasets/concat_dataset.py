# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import bisect
from typing import Iterable

import torch
from pycocotools3d.coco import COCO3d
from pycocotools3d.coco.abstractions.relation_detection import SSGDataset
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset
from functools import reduce
from .Dataset import Dataset, COCOEvaluableDataset, SGGEvaluableDataset, ImgInfo, DatasetStatistics
from ...structures import BoxList, BoxListConverter


# Note: order of super-classes is important because torch.ConcatDataset calls the super().__init__ method with no args
class ConcatDataset(SGGEvaluableDataset, _ConcatDataset):
    """
    Same as torch.utils.data.dataset.ConcatDataset, but exposes an extra method for querying the sizes of the image.
    """

    def __init__(self, datasets: Iterable[COCOEvaluableDataset | SGGEvaluableDataset]):
        _ConcatDataset.__init__(self, datasets)
        # Assert datasets not empty and all have images with the same number of dimensions
        # noinspection PyTypeChecker
        datasets: list[COCOEvaluableDataset | SGGEvaluableDataset] = self.datasets
        assert datasets
        self.n_dim = datasets[0].n_dim
        # Assert n_dim
        for ds in datasets[1:]:
            assert ds.n_dim == self.n_dim
        # Find out which API(s) the datasets implement
        self._all_coco = all(isinstance(ds, COCOEvaluableDataset) for ds in datasets)
        self._all_sgg = all(isinstance(ds, SGGEvaluableDataset) for ds in datasets)

        # We mostly assume that all datasets share the same set of categories, etc...
        self.categories: list[str] = datasets[0].categories
        self.predicates: list[str] = datasets[0].predicates
        self.attributes: list[str] = datasets[0].attributes
        self.contiguous_category_id_to_json_id: dict[int, int] = datasets[0].contiguous_category_id_to_json_id

        # Use cumulative_sizes to update the mapping
        self.offsets = [0] + self.cumulative_sizes
        self.contiguous_image_id_to_json_id: dict[int, int] = {
            k + offset: v
            for dataset, offset in zip(datasets, self.offsets)
            for k, v in dataset.contiguous_image_id_to_json_id.items()
        }

        # TODO we only support 3D because we are constructing the COCO dataset from BoxLists
        #  and it's only supported for 3D
        assert self.n_dim == 3
        self._coco = COCO3d()

        # noinspection PyTypeChecker
        self.filenames: list[str] = reduce(lambda a, b: a+b, [dataset.filenames for dataset in datasets])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, BoxList, int]:
        return _ConcatDataset.__getitem__(self, idx)

    @property
    def coco(self) -> COCO3d:
        if self._coco is not None:
            return self._coco

        use_cats = not self._cfg.MODEL.RPN_ONLY
        # No inspect because NotRequired does not seem supported on import
        if not use_cats:
            # noinspection PyTypeChecker
            anns: SSGDataset = {
                "info": {},
                "categories": [{"id": 1, "name": "Foreground"}],
                "images": [],
                "annotations": [],
                # Note: no relation loading in binary mode (as objects are not classified)
                "predicates": []
            }
        else:
            # noinspection PyTypeChecker
            anns: SSGDataset = {
                "info": {},
                "categories": [{"id": idx, "name": name} for idx, name in enumerate(self.categories)][1:],  # No bg
                "images": [],
                "annotations": [],
                "predicates": [{"id": idx, "name": name} for idx, name in enumerate(self.predicates)][1:]  # No bg
            }
        for index in range(len(self)):
            # Add target to COCO
            # TODO: disable the addition of attributes since we don't use COCO code to evaluate attributes
            BoxListConverter.add_to_coco_annotation(self.get_groundtruth(index), anns, index, use_cats=use_cats)
        self._coco = COCO3d()
        self._coco.dataset = anns
        self._coco.createIndex()
        return self._coco

    def get_img_info(self, idx: int) -> ImgInfo:
        dataset_idx, sample_idx = self._get_indexes(idx)
        dataset: Dataset = self.datasets[dataset_idx]
        return dataset.get_img_info(sample_idx)

    def _get_indexes(self, idx: int) -> tuple[int, int]:
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return dataset_idx, sample_idx

    def get_groundtruth(self, idx: int) -> BoxList:
        if not self._all_coco:
            raise RuntimeError("Attempting to use a COCOEvaluableDataset method "
                               "when not all of datasets implement this interface.")

        dataset_idx, sample_idx = self._get_indexes(idx)
        # noinspection PyTypeChecker
        dataset: COCOEvaluableDataset = self.datasets[dataset_idx]
        return dataset.get_groundtruth(sample_idx)

    def get_statistics(self) -> DatasetStatistics:
        if not self._all_sgg:
            raise RuntimeError("Attempting to use a SGGEvaluableDataset method "
                               "when not all of datasets implement this interface.")

        if len(self.datasets) == 1:
            # noinspection PyTypeChecker
            dataset: SGGEvaluableDataset = self.datasets[0]
            return dataset.get_statistics()

        all_statistics = [dataset.get_statistics() for dataset in self.datasets]
        # noinspection PyTypeChecker
        sum_fg_matrix: torch.LongTensor = sum([stat["fg_matrix"] for stat in all_statistics])

        concat_statistics = all_statistics[0]
        concat_statistics["fg_matrix"] = sum_fg_matrix
        concat_statistics["pred_dist"] = torch.log(sum_fg_matrix / (sum_fg_matrix.sum(2)[:, :, None] + 1e-5))

        return concat_statistics

    def __len__(self) -> int:
        return sum(len(dataset) for dataset in self.datasets)
