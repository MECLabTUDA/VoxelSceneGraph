import torch
import torchvision.transforms.functional as func

from scene_graph_prediction.config import cfg
from scene_graph_prediction.modeling.abstractions.box_head import BoxHeadTestProposal
from scene_graph_prediction.modeling.abstractions.detector import AbstractDetector
from scene_graph_prediction.modeling.abstractions.loss import LossDict
from scene_graph_prediction.modeling.roi_heads.box_head.default.inference import build_roi_box_postprocessor
from scene_graph_prediction.structures import ImageList, BoxList, BoxListOps

_SIZE_T = int | tuple[int, ...]


def im_detect_bbox_aug(
        model: AbstractDetector,
        images: ImageList,
        compute_loss: bool,
        device: torch.device,
        targets: list[BoxList] | None
) -> tuple[list[BoxHeadTestProposal], LossDict]:
    """
    For given images, predicts bounding boxes and augments these predictions.
    For the return type, check out the PostProcessor for box head inference...
    Note: this is test-time prediction augmentation to ensure that we detect as many objects as possible...
    Note: targets are not required but are passed for easier debugging.
    """
    # Collect detections computed under different transformations
    boxlists_ts: list[list[BoxList]] = [[] for _ in range(len(images))]
    # Compute detections for the original image (identity transform)
    # Note: we only compute losses for this case
    boxlists_i, losses = model(images.to(device), targets, compute_loss)
    for idx, boxlist_t in enumerate(boxlists_i):
        # The first one (func call) is identity transform, no need to resize the boxlist
        boxlists_ts[idx].append(boxlist_t)

    # Perform detection on the horizontally flipped image
    if cfg.TEST.BBOX_AUG.H_FLIP:
        boxlists_hf = _im_detect_bbox_h_flip(model, images, device, targets)
        for idx, boxlist_t in enumerate(boxlists_hf):
            # Resize the boxlist as the first one
            boxlists_ts[idx].append(BoxListOps.resize(boxlist_t, boxlists_ts[idx][0].size))

    # Compute detections at different scales
    # Note: this feature needs to be fixed to be used as images currently not rescaled to the proper scale
    # for _ in cfg.TEST.BBOX_AUG.SCALES:
    #     boxlists_scl = _im_detect_bbox(model, images, device)
    #     for idx, boxlist_t in enumerate(boxlists_scl):
    #         # Resize the boxlist as the first one
    #         boxlists_ts[idx].append(BoxListOps.resize(boxlist_t,boxlists_ts[idx][0].size))
    #
    #     if cfg.TEST.BBOX_AUG.SCALE_H_FLIP:
    #         boxlists_scl_hf = _im_detect_bbox_h_flip(model, images, device)
    #         for idx, boxlist_t in enumerate(boxlists_scl_hf):
    #             # Resize the boxlist as the first one
    #             boxlists_ts[idx].append(BoxListOps.resize(boxlist_t, boxlists_ts[idx][0].size))

    # Merge boxlists detected by different bbox aug params
    boxlists = [BoxListOps.cat(boxlist_ts) for boxlist_ts in boxlists_ts]

    # Apply NMS and limit the final detections
    post_processor = build_roi_box_postprocessor(cfg)
    return [post_processor.filter_results(boxlist, cfg.INPUT.N_OBJ_CLASSES)[0] for boxlist in boxlists], losses


def _im_detect_bbox_h_flip(
        model: AbstractDetector,
        images: ImageList,
        device: torch.device,
        targets: list[BoxList] | None
) -> list[BoxList]:
    """
    Performs bbox detection on the horizontally flipped image.
    Function signature is the same as for im_detect_bbox.
    Note: targets are not required but are passed for easier debugging.
    """
    # H-flip images in the ImageList and build it again
    images = [func.hflip(image) for image in images.tensors]
    images = ImageList.to_image_list(images, cfg.INPUT.N_DIM, size_divisible=cfg.DATALOADER.SIZE_DIVISIBILITY)
    # H-flip targets if available
    if targets:
        targets = [BoxListOps.flip(box, BoxList.FlipDim.WIDTH) for box in targets]
    boxlists, _ = model(images.to(device), targets)

    # Invert the detections computed on the flipped image
    return [BoxListOps.flip(boxlist, BoxList.FlipDim.WIDTH) for boxlist in boxlists]
