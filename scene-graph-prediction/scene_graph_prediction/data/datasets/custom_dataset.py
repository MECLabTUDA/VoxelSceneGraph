from pathlib import Path

import torch
from pycocotools3d.coco import COCO3d
from pycocotools3d.coco.abstractions.relation_detection import SSGDataset
from scene_graph_api.knowledge import KnowledgeGraph
from scene_graph_api.utils.nifti_io import NiftiImageWrapper
from scene_graph_api.utils.pathing import remove_suffixes
from yacs.config import CfgNode

from scene_graph_prediction.structures import BoxList, FieldExtractor, BoxListOps, BoxListConverter
from scene_graph_prediction.utils.logger import setup_logger
from .Dataset import DatasetStatistics, SGGEvaluableDataset
from .Dataset import ImgInfo
from .Split import DatasetSpliter
from .Split import Split
from ..transforms import Compose


class RelationDetectionDataset(SGGEvaluableDataset):
    """
    Dataset for Scene Graph Generation with data structures from scene_graph_api.
    """

    def __init__(
            self,
            cfg: CfgNode,
            datasets_dir: str,
            transforms: Compose,
            split: Split,
            img_dir: str,
            annotation_dir: str,
            knowledge_graph_file: str,
            spliter: DatasetSpliter
    ):
        """
        Torch dataset for object detection based on annotation from scene_graph_annotation.
        :param transforms: List of transforms.
                           Make sure that at least one produces the required masks when learning to segment,
                           e.g. RandomAffine or PrepareMasks.
        :param img_dir: folder containing all input images.
        :param annotation_dir: folder containing serialized BoxLists (with compressed masks).
        :param knowledge_graph_file: path to the knowledge graph used for annotating images.
        """
        super().__init__(cfg, datasets_dir, transforms, split)
        # TODO add support for 2D?
        self.n_dim = 3
        self._fold = cfg.DATASETS.FOLD
        self._img_dir = img_dir
        self._annotation_dir = annotation_dir

        self._logger = setup_logger(__file__, "", 1)
        self.knowledge = KnowledgeGraph.load(knowledge_graph_file, self._logger)
        if self.knowledge is None:
            raise FileNotFoundError(knowledge_graph_file)

        image_paths = sorted(list(Path(img_dir).glob("*")))
        assert image_paths
        # Path to image for each graph
        self._all_filenames = {remove_suffixes(p): p.as_posix() for p in image_paths}

        self._annotation_paths = sorted(list(Path(annotation_dir).glob("*.pth")))
        assert self._annotation_paths

        # Assert that the filenames in the annotation folder is a subset of filenames in the image folder
        image_names = {remove_suffixes(p) for p in image_paths}
        annotation_names = {remove_suffixes(p) for p in self._annotation_paths}
        assert annotation_names.issubset(image_names), annotation_names.difference(image_names)

        # Keys i.e. filenames without any suffix
        self._keys = [remove_suffixes(p) for p in self._annotation_paths]

        self._key_to_annotation_path = {k: p for k, p in zip(self._keys, self._annotation_paths)}
        # Keys to use for this dataset instance
        self._fold_split_keys = spliter(self._keys, self._split)

        # Load targets for split and remove small boxes
        self._compressed_targets: dict[str, BoxList] = {}
        min_size = cfg.INPUT.MIN_SIZE
        for index, key in enumerate(self._fold_split_keys):
            path = self._key_to_annotation_path[key]
            target = BoxList.load(path)
            target = BoxListOps.remove_small_boxes(target, min_size)

            # Check if weighted box training is enabled and whether the weights are supplied
            if self._cfg.MODEL.WEIGHTED_BOX_TRAINING and not target.has_field(target.AnnotationField.IMPORTANCE):
                # Default to a uniform weighting of boxes
                target.IMPORTANCE = torch.ones(len(target), dtype=torch.float32, device=target.boxes.device)

            self._compressed_targets[remove_suffixes(path)] = target

        # Dataset Interface
        self.categories = [self.BACKGROUND_CLASS_NAME] + [obj_class.name for obj_class in self.knowledge.classes]
        assert cfg.INPUT.N_OBJ_CLASSES == len(self.categories)

        # COCOEvaluableDataset Interface
        self._coco = None

        json_category_id_to_contiguous_id = {v: i + 1 for i, v in enumerate(self.coco.getCatIds())}
        self.contiguous_category_id_to_json_id = {v: k for k, v in json_category_id_to_contiguous_id.items()}
        self.contiguous_image_id_to_json_id = {v: v for v in range(len(self._fold_split_keys))}
        self.contiguous_image_id_to_json_name = {k: v for k, v in enumerate(self._fold_split_keys)}

        # COCO also supports relations now (even if none is annotated or planned)
        self.predicates = [self.BACKGROUND_CLASS_NAME] + [rule.name for rule in self.knowledge.rules]
        self.filenames = [self._all_filenames[key] for key in self._fold_split_keys]
        # TODO support attributes
        self.attributes = []

        # Assert that the len of predicates and attributes matches the number in the config file
        assert len(self.predicates) == cfg.INPUT.N_REL_CLASSES
        assert not cfg.MODEL.ATTRIBUTE_ON or len(self.attributes) == cfg.INPUT.N_ATT_CLASSES

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, BoxList, int]:
        """
        Note: targets have only their default fields (i.e. those that they were saved with).
        Note: we expect to have at least the LABELMAP (or binary masks), and the LABELS.
        Note: label maps are the preferred way to store instance segmentation as it's much more compact.
        """
        img_path = self.filenames[idx]
        # The data should always be saved with the original ordering
        nifti = NiftiImageWrapper.load_depth_first(img_path)

        # Convert to tensor
        img = torch.from_numpy(nifti.get_fdata())
        target = self._compressed_targets[self._fold_split_keys[idx]]
        transformed_img, transformed_target = self._transforms(img, target)  # type: torch.Tensor, BoxList

        # Field selection is handled by the RandomAffine transform
        # # Delete mask fields that are not necessary anymore and that could take up space (esp. when sampling)
        # if not self._cfg.MODEL.MASK_ON:
        #     transformed_target.del_field(transformed_target.AnnotationField.MASKS)
        #     transformed_target.del_field(transformed_target.AnnotationField.LABELMAP)
        # if not self._cfg.MODEL.REQUIRE_SEMANTIC_SEGMENTATION:
        #     transformed_target.del_field(transformed_target.AnnotationField.SEGMENTATION)

        # Add affine matrix to targets
        transformed_target.AFFINE_MATRIX = torch.tensor(nifti.affine)
        transformed_target = transformed_target.to(dtype=torch.float32)

        # Add the IMG_PATH field to the target
        transformed_target.IMG_PATH = img_path

        return transformed_img.to(dtype=torch.float32), transformed_target, idx

    def get_img_info(self, index: int) -> ImgInfo:
        key = self._fold_split_keys[index]
        target = self._compressed_targets[key]
        shape = target.size  # (d, h, w) i.e. sitk-shaped
        return {
            "file_path": self.filenames[index],
            "depth": shape[0],
            "height": shape[1],
            "width": shape[2]
        }

    def __len__(self) -> int:
        return len(self._fold_split_keys)

    def get_groundtruth(self, index: int) -> BoxList:
        """
        Note: targets have the needed mask fields as defined in the config, i.e.:
              - MASKS if MODEL.MASK_ON
              - SEGMENTATION if MODEL.REQUIRE_SEMANTIC_SEGMENTATION
        """
        key = self._fold_split_keys[index]
        target = self._compressed_targets[key]

        if self._cfg.MODEL.MASK_ON:
            if not target.has_field(BoxList.AnnotationField.MASKS):
                target.MASKS = FieldExtractor.masks(target)

        if self._cfg.MODEL.REQUIRE_SEMANTIC_SEGMENTATION:
            if not target.has_field(BoxList.AnnotationField.SEGMENTATION):
                target.SEGMENTATION = FieldExtractor.segmentation(target)

        return target

    @property
    def coco(self) -> COCO3d:
        """
        Fill the COCO object with the ground truth annotation.
        Note: does not load relations when in binary classification mode.
        """
        if self._coco is not None:
            return self._coco

        # Scene Graphs to COCO annotation file
        # No need to add relations, as we use our own framework for relation metrics computation
        # TODO maybe expand pycocotools3d at some point?

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
                "predicates": [],
                "relations": []
            }
        else:
            # TODO: disable the addition of attributes since we don't use COCO code to evaluate attributes
            # noinspection PyTypeChecker
            anns: SSGDataset = self.knowledge.to_coco()
        for index, target in enumerate(self._compressed_targets.values()):
            # Add target to COCO
            BoxListConverter.add_to_coco_annotation(target, anns, index, use_cats=use_cats)
        self._coco = COCO3d()
        self._coco.dataset = anns
        self._coco.createIndex()
        return self._coco

    def _compute_background_matrix(self, bg_matrix: torch.LongTensor):
        """
        It can be very useful to have statistics of objects not having any relations.
        Traditionally we would consider pairs of overlapping objects not being related.
        For 3D, it can be more challenging... so here we consider that all objects may be related.
        :param bg_matrix: the background matrix to fill in-place.
        """
        for key in self._fold_split_keys:
            target: BoxList = self._compressed_targets[key]  # No need to decompress masks
            relations = target.RELATIONS
            labels = target.LABELS

            subj_ids, obj_ids = torch.where(relations == 0)  # Tuple of coordinates, one tensor for each dim i.e. 2
            for subj_id, obj_id in zip(subj_ids, obj_ids):
                if subj_id == obj_id:
                    # Avoid counting relations of an object to itself
                    continue
                subj_class_id = labels[subj_id]
                obj_class_id = labels[obj_id]
                bg_matrix[subj_class_id, obj_class_id] += 1

    def get_statistics(self, force: bool = False) -> DatasetStatistics:
        """
        Compute statistics about relations.
        Note: this method can only work  if relation are present in the annotations.
        """

        if self._split != Split.TRAIN and not force:
            raise RuntimeError(
                f"The statistics can only be computed on the train split, "
                f"this dataset uses the {self._split.value} split..."
            )

        # Ids start at 1 and need to include the background
        num_obj_classes = len(self.knowledge.classes) + 1
        num_rel_classes = len(self.knowledge.rules) + 1

        fg_matrix = torch.zeros((num_obj_classes, num_obj_classes, num_rel_classes), dtype=torch.int64)

        for key in self._fold_split_keys:
            target: BoxList = self._compressed_targets[key]  # No need to decompress masks
            relations = target.RELATIONS
            labels = target.LABELS

            subj_ids, obj_ids = torch.where(relations != 0)  # Tuple of coordinates, one tensor for each dim i.e. 2
            for subj_id, obj_id in zip(subj_ids, obj_ids):
                subj_class_id = labels[subj_id]
                obj_class_id = labels[obj_id]
                rel_label = relations[subj_id, obj_id]
                fg_matrix[subj_class_id, obj_class_id, rel_label] += 1

        self._compute_background_matrix(fg_matrix[:, :, 0])
        pred_dist = torch.log(fg_matrix / (fg_matrix.sum(2)[:, :, None] + 1e-5))

        return {
            "fg_matrix": fg_matrix,
            "pred_dist": pred_dist,
            "obj_classes": self.categories,
            "rel_classes": self.predicates,
            "att_classes": self.attributes,
        }
