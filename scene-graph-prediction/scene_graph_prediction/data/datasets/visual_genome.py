import json
import os
import random
from collections import defaultdict
from typing import Literal

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from yacs.config import CfgNode

from scene_graph_prediction.structures import BoxList, BoxListOps
from .Dataset import Dataset, ImgInfo, DatasetStatistics
from .Split import Split
from ..transforms import Compose


class VGDataset(Dataset):
    BOX_SCALE = 1024  # Scale at which we have the boxes

    def __init__(
            self,
            cfg: CfgNode,
            datasets_dir: str,
            transforms: Compose,
            split: Split,
            img_dir: str,
            roi_db_file: str,
            dict_file: str,
            image_file: str,
            num_im: int = -1,
            num_val_im: int = 5000,
            filter_duplicate_rels: bool = True
    ):
        """
        Torch dataset for VisualGenome
        :param split: Must be either train, test, or val
        :param img_dir: folder containing all vg images
        :param roi_db_file: HDF5 containing the GT boxes, classes, and relationships
        :param dict_file: JSON Contains mapping of classes/relationships to words
        :param image_file: HDF5 containing image filenames
        :param filter_duplicate_rels: Whenever we see a duplicate relationship we'll sample instead
        :param num_im: Number of images in the entire dataset. -1 for all images.
        :param num_val_im: Number of images in the validation set (must be less than num_im unless num_im is -1).
        """
        super().__init__(cfg, datasets_dir, transforms, split)

        self.n_dim = 2
        self._flip_aug = cfg.MODEL.FLIP_AUG
        self._img_dir = os.path.join(datasets_dir, img_dir)
        self._roi_db_file = os.path.join(datasets_dir, roi_db_file)
        self._dict_file = os.path.join(datasets_dir, dict_file)
        self._image_file = os.path.join(datasets_dir, image_file)
        self._filter_non_overlap = (not cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX and
                                    cfg.MODEL.RELATION_ON and
                                    cfg.MODEL.ROI_RELATION_HEAD.REQUIRE_BOX_OVERLAP and
                                    split == Split.TRAIN)
        self._filter_duplicate_rels = filter_duplicate_rels and split == Split.TRAIN

        # Contiguous 151, 51 containing "Background"
        self.categories, self.predicates, self.attributes = self._load_info(dict_file)

        # IF MODEL.RELATION_ON is True, filter images with empty rels,
        # else set filter to False because we need all images for pretraining detector
        split_mask, self._gt_boxes, self._gt_classes, self._gt_attributes, self._relationships = self._load_graphs(
            roi_db_file, split.value, num_im, num_val_im=num_val_im,
            filter_empty_rels=cfg.MODEL.RELATION_ON,
            filter_non_overlap=self._filter_non_overlap,
        )

        # Length equals to split_mask
        filenames, img_info = self._load_image_filenames(img_dir, image_file)
        self.filenames = [filenames[i] for i in np.where(split_mask)[0]]
        self._img_info: list[ImgInfo] = [img_info[i] for i in np.where(split_mask)[0]]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, BoxList, int]:
        img = Image.open(self.filenames[idx]).convert("RGB")
        if img.size[0] != self._img_info[idx]['width'] or img.size[1] != self._img_info[idx]['height']:
            print('=' * 20 +
                  f' ERROR idx {idx} {img.size} {self._img_info[idx]["width"]} {self._img_info[idx]["height"]} ' +
                  '=' * 20)
            raise RuntimeError

        flip_img = random.random() > 0.5 and self._flip_aug and self._split == Split.TRAIN
        target = self.get_groundtruth(idx, flip_img)

        if flip_img:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)

        img, target = self._transforms(img, target)

        return img, target, idx

    def get_statistics(self) -> DatasetStatistics:
        # TODO do idx remapping
        fg_matrix, bg_matrix = self._get_VG_statistics(must_overlap=True)
        eps = 1e-3
        bg_matrix += 1
        fg_matrix[:, :, 0] = bg_matrix
        pred_dist = np.log(fg_matrix / fg_matrix.sum(2)[:, :, None] + eps)

        # Numpy array is np.uint64
        # noinspection PyTypeChecker
        fg_matrix: torch.LongTensor = torch.from_numpy(fg_matrix)
        return {
            'fg_matrix': fg_matrix,
            'pred_dist': torch.from_numpy(pred_dist).float(),
            'obj_classes': self.categories,
            'rel_classes': self.predicates,
            'att_classes': self.attributes,
        }

    def get_img_info(self, index: int) -> ImgInfo:
        # WARNING: original image_file.json has several pictures with false image size
        # Use correct function to check the validity before training: it will take a while, you only need to do it once
        # correct_img_info(self.img_dir, self.image_file)
        return self._img_info[index]

    def get_groundtruth(self, index: int, flip_img: bool = False) -> BoxList:
        """:returns: a RelationHeadTarget"""
        img_info = self.get_img_info(index)
        w, h = img_info["width"], img_info["height"]
        # Important: recover original box from BOX_SCALE
        # noinspection PyTypeChecker
        box = self._gt_boxes[index] / self.BOX_SCALE * max(w, h)
        box = torch.from_numpy(box).reshape(-1, 4)  # guard against no boxes

        if flip_img:
            new_x_min = w - box[:, 2]
            new_x_max = w - box[:, 0]
            box[:, 0] = new_x_min
            box[:, 2] = new_x_max
        target = BoxList(box, (h, w), BoxList.Mode.zyxzyx)

        target.add_field(BoxList.AnnotationField.LABELS, torch.from_numpy(self._gt_classes[index]))
        target.add_field(BoxList.AnnotationField.ATTRIBUTES, torch.from_numpy(self._gt_attributes[index]))

        # noinspection PyTypeChecker
        relation = self._relationships[index].copy()  # (num_rel, 3)
        if self._filter_duplicate_rels:
            # Filter out dupes!
            assert self._split == Split.TRAIN
            all_rel_sets = defaultdict(list)
            for (o0, o1, r) in relation:
                all_rel_sets[(o0, o1)].append(r)
            relation = [(k[0], k[1], np.random.choice(v)) for k, v in all_rel_sets.items()]
            relation = np.array(relation, dtype=np.int32)

        # Add relation to target
        num_box = len(target)
        relation_map = torch.zeros((num_box, num_box), dtype=torch.int64)
        for i in range(relation.shape[0]):
            if relation_map[int(relation[i, 0]), int(relation[i, 1])] > 0:
                if random.random() > 0.5:
                    relation_map[int(relation[i, 0]), int(relation[i, 1])] = int(relation[i, 2])
            else:
                relation_map[int(relation[i, 0]), int(relation[i, 1])] = int(relation[i, 2])
        target.add_field(BoxList.AnnotationField.RELATIONS, relation_map, indexing_power=2)

        return target.clip_to_image(remove_empty=True)

    def __len__(self) -> int:
        return len(self.filenames)

    # noinspection PyPep8Naming
    def _get_VG_statistics(self, must_overlap: bool = True) -> tuple[np.ndarray, np.ndarray]:
        # We need to create a new dataset because we always need the stats for the train split
        train_data = VGDataset(
            self._cfg, self._datasets_dir, Compose([]), Split.TRAIN,  # Dummy transforms
            img_dir=self._img_dir, roi_db_file=self._roi_db_file,
            dict_file=self._dict_file, image_file=self._image_file, num_val_im=5000,
            filter_duplicate_rels=False
        )
        num_obj_classes = len(train_data.categories)
        num_rel_classes = len(train_data.predicates)
        fg_matrix = np.zeros((num_obj_classes, num_obj_classes, num_rel_classes), dtype=np.int64)
        bg_matrix = np.zeros((num_obj_classes, num_obj_classes), dtype=np.int64)

        for ex_ind in tqdm(range(len(train_data))):
            gt_classes = train_data._gt_classes[ex_ind].copy()
            gt_relations = train_data._relationships[ex_ind].copy()
            gt_boxes = train_data._gt_boxes[ex_ind].copy()

            # For the foreground, we'll just look at everything
            # noinspection PyTypeChecker
            o1o2 = gt_classes[gt_relations[:, :2]]
            for (o1, o2), gtr in zip(o1o2, gt_relations[:, 2]):
                fg_matrix[o1, o2, gtr] += 1
            # For the background, get all the things that overlap.
            # noinspection PyTypeChecker
            o1o2_total = gt_classes[np.array(VGDataset._box_filter(gt_boxes, must_overlap=must_overlap), dtype=int)]
            for (o1, o2) in o1o2_total:
                bg_matrix[o1, o2] += 1

        return fg_matrix, bg_matrix

    @staticmethod
    def _box_filter(boxes: np.ndarray, must_overlap: bool = False) -> np.ndarray:
        """Only include boxes that overlap as possible relations. If no overlapping boxes, use all of them."""
        dummy_size = 1, 1
        overlaps = BoxListOps.iou(
            BoxList(boxes.astype(np.float32), dummy_size),
            BoxList(boxes.astype(np.float32), dummy_size)
        ) > 0
        np.fill_diagonal(overlaps, 0)

        all_possibilities = np.ones_like(overlaps, dtype=bool)
        np.fill_diagonal(all_possibilities, 0)

        if must_overlap:
            possible_boxes = np.column_stack(np.where(overlaps))
            if possible_boxes.size == 0:
                possible_boxes = np.column_stack(np.where(all_possibilities))
            return possible_boxes

        return np.column_stack(np.where(all_possibilities))

    @staticmethod
    def _correct_img_info(img_dir: str, image_file: str):
        with open(image_file, "r") as f:
            data = json.load(f)
        for i in range(len(data)):
            img = data[i]
            basename = f"{img['image_id']}.jpg"
            filename = os.path.join(img_dir, basename)
            img_data = Image.open(filename).convert("RGB")
            if img["width"] != img_data.size[0] or img["height"] != img_data.size[1]:
                print(f"--------- False id: {i} ---------")
                print(img_data.size)
                print(img)
                data[i]["width"] = img_data.size[0]
                data[i]["height"] = img_data.size[1]
        with open(image_file, "w") as outfile:
            json.dump(data, outfile)

    @staticmethod
    def _load_info(dict_file: str, add_bg: bool = True) -> tuple[list, list, list]:
        """Loads the file containing the visual genome label meanings."""
        info = json.load(open(dict_file, "r"))
        if add_bg:
            info["label_to_idx"]["__background__"] = 0
            info["predicate_to_idx"]["__background__"] = 0
            info["attribute_to_idx"]["__background__"] = 0

        class_to_ind = info["label_to_idx"]
        predicate_to_ind = info["predicate_to_idx"]
        attribute_to_ind = info["attribute_to_idx"]
        categories = sorted(class_to_ind, key=lambda k: class_to_ind[k])
        predicates = sorted(predicate_to_ind, key=lambda k: predicate_to_ind[k])
        attributes = sorted(attribute_to_ind, key=lambda k: attribute_to_ind[k])

        return categories, predicates, attributes

    @staticmethod
    def _load_image_filenames(img_dir: str, image_file: str) -> tuple[list[str], list]:
        """
        Loads the image filenames from visual genome from the JSON file that contains them.
        This matches the preprocessing in scene-graph-TF-release/data_tools/vg_to_imdb.py.
        :param image_file: JSON file. Elements contain the param "image_id".
        :param img_dir: directory where the VisualGenome images are located
        :return: List of filenames corresponding to the good images
        """
        with open(image_file, "r") as f:
            im_data = json.load(f)

        corrupted_ims = ["1592.jpg", "1722.jpg", "4616.jpg", "4617.jpg"]
        fns = []
        img_info = []
        for i, img in enumerate(im_data):
            basename = f"{img['image_id']}.jpg"
            if basename in corrupted_ims:
                continue

            filename = os.path.join(img_dir, basename)
            if os.path.exists(filename):
                fns.append(filename)
                img_info.append(img)
        assert len(fns) == 108073
        assert len(img_info) == 108073
        return fns, img_info

    @staticmethod
    def _load_graphs(roi_db_file: str,
                     split: Literal["train", "test", "val"],
                     num_im: int,
                     num_val_im: int,
                     filter_empty_rels: bool,
                     filter_non_overlap: bool) \
            -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        """
        Load the file containing the GT boxes and relations, as well as the dataset split.
        :param roi_db_file: HDF5
        :param split: (train, val, or test)
        :param num_im: Number of images we want
        :param num_val_im: Number of validation images
        :param filter_empty_rels: will not be filtered otherwise
        :param filter_non_overlap: If training, filter images that don't overlap.
        :return:
            image_index: numpy array corresponding to the index of images we're using.
            boxes: List where each element is a [num_gt, 4] array of ground truth boxes (x1, y1, x2, y2).
            gt_classes: List where each element is a [num_gt] array of classes
            gt_attributes: List where each element is a [num_att, int64] array
            relationships: List where each element is a [num_r, 3] array of (box_ind_1, box_ind_2, predicate) rels
        """
        import h5py
        roi_h5 = h5py.File(roi_db_file, "r")
        data_split = roi_h5["split"][:]
        split_flag = 2 if split == "test" else 0
        split_mask = data_split == split_flag

        # Filter out images without bounding boxes
        split_mask &= roi_h5["img_to_first_box"][:] >= 0
        if filter_empty_rels:
            split_mask &= roi_h5["img_to_first_rel"][:] >= 0

        image_index = np.where(split_mask)[0]
        if num_im > -1:
            image_index = image_index[:num_im]
        if num_val_im > 0:
            if split == "val":
                image_index = image_index[:num_val_im]
            elif split == "train":
                image_index = image_index[num_val_im:]

        split_mask = np.zeros_like(data_split).astype(bool)
        split_mask[image_index] = True

        # Get box information
        all_labels = roi_h5["labels"][:, 0]
        all_attributes = roi_h5["attributes"][:, :]
        all_boxes = roi_h5[f"boxes_{VGDataset.BOX_SCALE}"][:]  # cx,cy,w,h

        assert np.all(all_boxes[:, :2] >= 0)  # Sanity check
        assert np.all(all_boxes[:, 2:] > 0)  # No empty box

        # Convert from xc, yc, w, h to x1, y1, x2, y2
        all_boxes[:, :2] = all_boxes[:, :2] - all_boxes[:, 2:] / 2
        all_boxes[:, 2:] = all_boxes[:, :2] + all_boxes[:, 2:]

        im_to_first_box = roi_h5["img_to_first_box"][split_mask]
        im_to_last_box = roi_h5["img_to_last_box"][split_mask]
        im_to_first_rel = roi_h5["img_to_first_rel"][split_mask]
        im_to_last_rel = roi_h5["img_to_last_rel"][split_mask]

        # Load relation labels
        relations = roi_h5["relationships"][:]
        relation_predicates = roi_h5["predicates"][:, 0]
        assert im_to_first_rel.shape[0] == im_to_last_rel.shape[0]
        assert relations.shape[0] == relation_predicates.shape[0]  # sanity check

        # Get everything by image.
        boxes = []
        gt_classes = []
        gt_attributes = []
        relationships = []
        for i in range(len(image_index)):
            i_obj_start = im_to_first_box[i]
            i_obj_end = im_to_last_box[i]
            i_rel_start = im_to_first_rel[i]
            i_rel_end = im_to_last_rel[i]

            boxes_i = all_boxes[i_obj_start: i_obj_end + 1, :]
            gt_classes_i = all_labels[i_obj_start: i_obj_end + 1]
            gt_attributes_i = all_attributes[i_obj_start: i_obj_end + 1, :]

            if i_rel_start >= 0:
                predicates = relation_predicates[i_rel_start: i_rel_end + 1]
                obj_idx = relations[i_rel_start: i_rel_end + 1] - i_obj_start  # range is [0, num_box)
                assert np.all(obj_idx >= 0)
                assert np.all(obj_idx < boxes_i.shape[0])
                rels = np.column_stack((obj_idx, predicates))  # (num_rel, 3), representing sub, obj, and pred
            else:
                assert not filter_empty_rels
                rels = np.zeros((0, 3), dtype=np.int32)

            if filter_non_overlap:
                assert split == "train"
                # Construct a BoxList object to apply the iou method (with a dummy size)
                boxes_i_obj = BoxList(boxes_i, (1000, 1000), BoxList.Mode.zyxzyx)
                inters = BoxListOps.iou(boxes_i_obj, boxes_i_obj)
                rel_overs = inters[rels[:, 0], rels[:, 1]]
                inc = np.where(rel_overs > 0.0)[0]

                if inc.size > 0:
                    rels = rels[inc]
                else:
                    split_mask[image_index[i]] = 0
                    continue

            boxes.append(boxes_i)
            gt_classes.append(gt_classes_i)
            gt_attributes.append(gt_attributes_i)
            relationships.append(rels)

        return split_mask, boxes, gt_classes, gt_attributes, relationships
