from functools import partial

import cc3d
import numpy as np
import torch
from yacs.config import CfgNode

from scene_graph_prediction.structures import BoxList, BoxListOps
from scene_graph_prediction.utils.miscellaneous import torch_bbox_2d, torch_bbox_3d
from ..retinaunet import RetinaUNetModule
from ...abstractions.backbone import AnchorStrides
from ...abstractions.box_head import BoxHeadTestProposals
from ...abstractions.loss import RPNLossDict
from ...abstractions.region_proposal import RPNProposals


class RetinaUNetHybridModule(RetinaUNetModule):
    """
    Customized RetinaUNet, which uses segmentation masks to predict the bounding box of objects of classes,
    where we know that there is exactly one object per image.
    Note: support 2D and 3D.
    """
    MINUS_INF = -10

    def __init__(self, cfg: CfgNode, in_channels: int, anchor_strides: AnchorStrides):
        self.num_unique_fg_classes = cfg.INPUT.N_UNIQUE_OBJ_CLASSES
        assert self.num_unique_fg_classes > 0, "No class with unique object, use RetinaUNet instead."

        self.num_normal_fg_classes = cfg.INPUT.N_OBJ_CLASSES - self.num_unique_fg_classes - 1

        # List of classes where we only keep the largest connected components
        self.keep_largest_cc_classes = cfg.MODEL.RETINANET.KEEP_LARGEST_CC_CLASSES
        assert all(c > self.num_normal_fg_classes for c in self.keep_largest_cc_classes), \
            "Cannot keep largest island for objects not detected purely through segmentation."

        # Per-class list of thresholds for dusting (padding done for convenience)
        self.dusting_thrs = [0] * (self.num_normal_fg_classes + 1) + cfg.MODEL.RETINANET.DUST_THRESHOLDS
        self.dusting_thrs += [0] * (cfg.INPUT.N_OBJ_CLASSES - len(self.dusting_thrs))

        super().__init__(cfg, in_channels, anchor_strides, override_detec_num_fg_classes=self.num_normal_fg_classes)

        assert self.n_dim in [2, 3], "Only 2D or 3D."
        assert not self.is_binary_classification, "This module cannot be used as a two-state model."

    def post_process_predictions(
            self, args, targets: list[BoxList] | None = None, _: bool = False
    ) -> RPNProposals | BoxHeadTestProposals:
        """
        Notes on fields added to the proposals:
        - during training: "labels", "attributes", "pred_logits"
        - during testing: "pred_logits", "pred_scores", "pred_labels", "boxes_per_cls"
        """
        # Filter targets to remove classes with unique object
        # The SEGMENTATION field used to learn the segmentation should be preserved
        if targets is not None:
            super_targets = [target[target.LABELS <= self.num_normal_fg_classes] for target in targets]
        else:
            super_targets = None
        # Still cannot annotate type hints perfectly...
        # noinspection PyTypeChecker
        proposals = super().post_process_predictions(args, super_targets, keep_seg_logits=True)

        # Just avoid doing this if we're only training a one-stage detector
        if not self.training or self.training_requires_full_processing:
            self._pad_pred_logits_field(proposals)
            self._pad_boxes_per_cls_field(proposals)
            self._detect_from_segmentation(proposals)
            # Then sort objects again by score
            for idx, prop in enumerate(proposals):
                proposals[idx] = prop[torch.argsort(prop.PRED_SCORES, descending=True)]

        return proposals

    def loss(self, args, targets: list[BoxList]) -> RPNLossDict:
        # We filter out boxes with labels that are hidden away from pure object detection
        # The segmentation loss can be computed as usual as the semantic segmentation field is unchanged
        super_targets = [target[target.LABELS <= self.num_normal_fg_classes] for target in targets]
        return super().loss(args, super_targets)

    def _pad_pred_logits_field(self, proposals: list[BoxList]):
        """
        Since the detection head does not know about object classes detected by segmentation,
        the logits tensor does not have the right shape (i.e. missing logits for extra classes).
        """
        for proposal in proposals:  # Iterating over image batch
            pred_logits = proposal.PRED_LOGITS
            padding = torch.full((pred_logits.shape[0], self.num_unique_fg_classes), self.MINUS_INF).to(pred_logits)
            proposal.PRED_LOGITS = torch.column_stack([pred_logits, padding])

    def _pad_boxes_per_cls_field(self, proposals: list[BoxList]):
        """
        Since the detection head does not know about object classes detected by segmentation,
        the boxes_per_cls tensor does not have the right shape (i.e. missing boxes for extra classes).
        """
        for proposal in proposals:  # Iterating over image batch
            boxes_per_cls = proposal.BOXES_PER_CLS
            extra_boxes = torch.tile(boxes_per_cls[:, :2 * self.n_dim], (1, self.num_unique_fg_classes))
            proposal.BOXES_PER_CLS = torch.cat([boxes_per_cls, extra_boxes], 1)

    def _denoise_seg(self, seg: torch.LongTensor, class_id: int) -> torch.LongTensor:
        """
        Apply different strategies to remove noise in the mask before computing the objects' bounds.
        E.g. keep the largest component or do some dusting.
        The segmentation is also updated such that final segmentation metrics are accurate.
        :param seg: the predicted segmentation.
        :param class_id: the class id being handled.
        :return: the de-noised mask.
        """
        # noinspection PyTypeChecker
        obj_mask: torch.LongTensor = seg == class_id

        # Find the proper processing function
        if class_id in self.keep_largest_cc_classes:
            proc_func = partial(cc3d.largest_k, k=1)
        elif dust_thr := self.dusting_thrs[class_id]:
            proc_func = partial(cc3d.dust, threshold=dust_thr)
        else:
            return obj_mask

        obj_mask_np = obj_mask.detach().cpu().numpy()
        new_mask_np = proc_func(obj_mask_np)
        new_mask = torch.tensor(new_mask_np.astype(np.uint8)).to(obj_mask)

        # Also update the predicted segmentation, to compute the proper metrics
        seg[obj_mask != new_mask] = 0

        return new_mask

    def _detect_from_segmentation(self, proposals: list[BoxList]):
        """
        Detect the relevant objects from the predicted segmentation mask and concatenate these to the proposals.
        """
        # Finally for evaluation: add the boxes for the unique objects, based on the predicted segmentation
        # To do that, we can a new BoxList with the following fields:
        #  "pred_labels", "pred_logits", "pred_scores", "pred_segmentation"
        num_classes = self.num_normal_fg_classes + self.num_unique_fg_classes + 1
        for idx in range(len(proposals)):  # Iterating over image batch
            proposal = proposals[idx]
            seg = proposal.PRED_SEGMENTATION
            seg_logits = proposal.PRED_SEGMENTATION_LOGITS  # CxDxHxW

            # List of tensors to cat
            pred_boxes = []
            pred_labels = []
            pred_logits = []
            pred_scores = []

            # For each class
            for class_id in range(self.num_normal_fg_classes + 1, num_classes):
                pred_labels.append(torch.tensor([class_id]).to(proposal.PRED_LABELS))
                # Note: there is a bg class logit
                logits = torch.full((num_classes,), self.MINUS_INF).to(proposal.pred_logits)

                obj_mask = self._denoise_seg(seg, class_id)
                if not torch.any(obj_mask):
                    # Add dummy box to produce consistent proposals
                    pred_boxes.append(torch.tensor([0, 0] * self.n_dim).to(proposal.boxes))
                    logits[class_id] = 0  # Not confident prediction
                    pred_logits.append(logits)
                    pred_scores.append(torch.tensor([0]).to(proposal.PRED_SCORES))
                    continue

                if self.n_dim == 2:
                    bbox = torch_bbox_2d(obj_mask).to(proposal.boxes)
                else:
                    bbox = torch_bbox_3d(obj_mask).to(proposal.boxes)
                pred_boxes.append(bbox)
                # Compute the score by taking the max logit for this class in the mask
                class_logit = torch.max(seg_logits[class_id][obj_mask])
                pred_scores.append(torch.sigmoid(class_logit))
                # Compute logits by setting all other classes' logits to -INF
                logits[class_id] = class_logit
                pred_logits.append(logits)

            # Build the BoxList object with additional objects
            unique_objects = BoxList(torch.row_stack(pred_boxes), proposal.size)
            unique_objects.PRED_LABELS = torch.row_stack(pred_labels).view(-1).long()
            unique_objects.PRED_LOGITS = torch.row_stack(pred_logits)
            unique_objects.PRED_SCORES = torch.row_stack(pred_scores).view(-1)
            # Create dummy boxes per class by just copying the predicted ones for each class
            unique_objects.BOXES_PER_CLS = torch.tile(unique_objects.boxes, (1, num_classes))
            unique_objects.PRED_SEGMENTATION = proposal.PRED_SEGMENTATION  # We need the same set of fields

            # Check whether we also need to add a LABELS field for relation training
            if proposal.has_field(BoxList.AnnotationField.LABELS):
                # Note: this does not take into account that
                # the segmentation may not be good enough to deserve a positive match
                # But this should be quite rare
                unique_objects.LABELS = unique_objects.PRED_LABELS

            # Clean-up field that is not otherwise used anywhere else
            proposal.del_field(proposal.PredictionField.PRED_SEGMENTATION_LOGITS)

            # Concatenate
            proposals[idx] = BoxListOps.cat([proposal, unique_objects])
