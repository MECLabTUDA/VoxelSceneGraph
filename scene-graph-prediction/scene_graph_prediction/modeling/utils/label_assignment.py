import torch

from scene_graph_prediction.structures import BoxList
from .matching import IoUMatcher


def assign_label_to_proposals(proposals: list[BoxList], targets: list[BoxList], fg_iou_thr: float):
    """Utility function for one-stage detectors to perform label assignment on proposals."""
    if len(proposals) == 0:
        return proposals

    rel_object_matcher = IoUMatcher(fg_iou_thr, fg_iou_thr)

    # For each image in the batch, match proposals to groundtruth and add LABELS field
    # Note: we're doing part of the sampling for relation detection
    #       I.e. we rely on the detector to find out the groundtruth label of the detection,
    #       But we need to keep in mind that the classification may not be final yet
    for img_idx, (target, proposal) in enumerate(zip(targets, proposals)):
        if len(proposal) == 0:
            # Still need to add empty attributes
            proposal.LABELS = torch.zeros(0, device=proposal.boxes.device, dtype=torch.int64)
            proposal.MATCHED_IDXS = torch.zeros(0, device=proposal.boxes.device, dtype=torch.int64)

        matched_indexes = rel_object_matcher(target, proposal)
        proposal.MATCHED_IDXS = matched_indexes
        proposal.LABELS = target.LABELS.long()[matched_indexes.clamp(min=0)]
        proposal.LABELS[matched_indexes < 0] = 0


def assign_label_to_proposals_always_match_special(
        proposals: list[BoxList],
        targets: list[BoxList],
        fg_iou_thr: float,
        always_match_class_idxs: int
):
    """
    Utility function for one-stage detectors to perform label assignment on proposals.
    However, all objects with predicted label >= always_match_class_idxs are always matched to the GT object
    with the same label (we assume that there is only one possible match).
    These objects are also excluded from matching with other objects
    Note: this is used in the hybrid Retina U-Net.
    """
    if len(proposals) == 0:
        return

    # Do the usual matching, but prevent unique objects in targets from matching

    # The following legacy code only works if the unique objects are last in the list
    # Note: it used to work because, we didn't compute MATCHED_IDXS back then...
    # super_targets = [target[target.LABELS <= always_match_class_idxs] for target in targets]
    # Instead create copies of targets and set the relevant boxes as empty
    super_targets = []
    for target in targets:
        target = target.copy_with_fields([BoxList.AnnotationField.LABELS])
        target.boxes = torch.clone(target.boxes)
        target.boxes[target.LABELS > always_match_class_idxs] = 0
        super_targets.append(target)
    assign_label_to_proposals(proposals, super_targets, fg_iou_thr)

    # Overwrite LABELS and MATCHED_IDXS of unique objects (detected through segmentation)
    for proposal, target in zip(proposals, targets):
        unique_obj_mask = proposal.PRED_LABELS > always_match_class_idxs
        # Match label independently of detection quality (since it should be very robust)
        proposal.LABELS[unique_obj_mask] = proposal.PRED_LABELS[unique_obj_mask]
        # Compute corresponding indices
        # noinspection PyTypeChecker
        _, matched_idxs = torch.nonzero(proposal.PRED_LABELS[unique_obj_mask][:, None] == target.LABELS, as_tuple=True)
        proposal.MATCHED_IDXS[unique_obj_mask] = matched_idxs
