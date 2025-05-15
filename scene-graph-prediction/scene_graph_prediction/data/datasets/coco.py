# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os

import torch
from PIL import Image
from pycocotools3d.coco import COCO
from yacs.config import CfgNode

from scene_graph_prediction.structures import BoxList, PersonKeypoints, PolygonList
from .Dataset import COCOEvaluableDataset, ImgInfo
from .Split import Split
from ..transforms import Compose


class COCODataset(COCOEvaluableDataset):

    def __init__(
            self,
            cfg: CfgNode,
            datasets_dir: str,
            transforms: Compose,
            split: Split,
            img_dir: str,
            ann_file: str,
    ):
        """
        `MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.
        It requires the `COCO API to be installed <https://github.com/pdollar/coco/tree/master/PythonAPI>`_.

        :param ann_file: Path to json annotation file.
        :param transforms: A function/transform that takes input sample and its target as entry and
                           returns a transformed version.
        """
        super().__init__(cfg, datasets_dir, transforms, split)
        # TODO add 3D support? Mostly need to use the proper Coco class?
        self.n_dim = 2

        self._root = os.path.expanduser(os.path.join(datasets_dir, img_dir))
        ann_file = os.path.join(datasets_dir, ann_file)
        self._coco = COCO(ann_file)
        self._ids: list[int] = list(sorted(self.coco.imgs.keys()))

        # Filter images without detection annotations
        if split == Split.TRAIN:
            ids = []
            for img_id in self._ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if self._has_valid_annotation(anno):
                    ids.append(img_id)
            self._ids = ids

        # Build a list of labels from a potentially sparse dict of ids
        cat_dict: dict[int, str] = {cat["id"]: cat["name"] for cat in self.coco.cats.values()}
        cat_list: list[str] = [""] * max(cat_dict.keys())
        for k, v in cat_dict.items():
            cat_list[k] = v
        self.categories = cat_list

        self.json_category_id_to_contiguous_id = {v: i + 1 for i, v in enumerate(self.coco.getCatIds())}
        self.contiguous_category_id_to_json_id = {v: k for k, v in self.json_category_id_to_contiguous_id.items()}
        self.contiguous_image_id_to_json_id = {k: v for k, v in enumerate(self._ids)}

    @property
    def coco(self) -> COCO:
        return self._coco

    def __len__(self) -> int:
        return len(self._ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, BoxList, int]:
        id_ = self._ids[idx]
        img: Image.Image = self._load_image(id_)
        target = self.get_groundtruth(idx)

        img, target = self._transforms(img, target)  # type: torch.Tensor, BoxList

        return img, target, idx

    def get_img_info(self, idx: int) -> ImgInfo:
        # noinspection PyTypeChecker
        return self.coco.imgs[self.contiguous_image_id_to_json_id[idx]]

    def get_groundtruth(self, index: int) -> BoxList:
        id_ = self._ids[index]
        anno: list = self._load_target(id_)

        img_info = self.get_img_info(index)
        if "depth" in img_info:
            img_size = img_info["depth"], img_info["height"], img_info["width"]
        else:
            img_size = img_info["height"], img_info["width"]

        # Filter crowd annotations
        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 2 * self.n_dim)  # Guard against no boxes
        target = BoxList(boxes, tuple(reversed(img_size)), mode=BoxList.Mode.zyxdhw).convert(BoxList.Mode.zyxzyx)

        classes = [obj["category_id"] for obj in anno]
        # TODO use the new reindex_groundtruth method
        # TODO check if everything needs to be contiguous for pycocotools
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.LABELS = classes

        if anno and "segmentation" in anno[0]:
            masks = [obj["segmentation"] for obj in anno]
            masks = PolygonList(masks, img_size)
            target.MASKS = masks

        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = PersonKeypoints(keypoints, img_size)
            target.KEYPOINTS = keypoints

        return target.clip_to_image(remove_empty=True)

    def _has_valid_annotation(self, anno: list) -> bool:
        _min_keypoints_per_image = 10

        # If it's empty, there is no annotation
        if len(anno) == 0:
            return False

        # If all boxes have close to zero area, there is no annotation
        if all(any(o <= 1 for o in obj["bbox"][self.n_dim:]) for obj in anno):
            return False

        # Keypoints task have a slightly different criteria for considering if an annotation is valid
        if "keypoints" not in anno[0]:
            return True

        # For keypoint detection tasks, only consider valid images those containing at least min_keypoints_per_image
        count = sum(sum(1 for v in ann["keypoints"][self.n_dim::self.n_dim + 1] if v > 0) for ann in anno)
        return count >= _min_keypoints_per_image

    def _load_image(self, idx: int) -> Image.Image:
        path = self.coco.loadImgs(idx)[0]["file_name"]
        return Image.open(os.path.join(self._root, path)).convert("RGB")

    def _load_target(self, idx: int) -> list:
        return self.coco.loadAnns(self.coco.getAnnIds(idx))
