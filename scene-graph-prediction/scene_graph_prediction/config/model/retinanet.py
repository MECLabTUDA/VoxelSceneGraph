from scene_graph_prediction.utils.config import AccessTrackingCfgNode

# -------------------------------------------------------------------------------------------------------------------- #
# RetinaNet Options (Follow the Detectron version)
# -------------------------------------------------------------------------------------------------------------------- #
RETINANET = AccessTrackingCfgNode()


# Weight for bbox_regression loss
RETINANET.BBOX_REG_WEIGHT = 4.0

# Smooth L1 loss beta for bbox regression
RETINANET.BBOX_REG_BETA = 0.11

# Inference cls score threshold, anchors with score > INFERENCE_TH are considered for inference
RETINANET.INFERENCE_TH = 0.05

# Whether RetinaNet is used as an RPN for a two-stage method e.g. Faster-RCNN.
RETINANET.TWO_STAGE = False

# Selected feature-maps (from P0 to P5; see thr FPN backbone)
# Note: the entire list of feature maps is still used for ROI heads (if any)
RETINANET.SELECTED_FEATURE_MAPS = [2, 3, 4, 5]

# -------------------------------------------------------------------------------------------------------------------- #
# RetinaUNet-Stroke Options (Follow the Detectron version)
# -------------------------------------------------------------------------------------------------------------------- #
# Optionally, for classes which have exactly one object per image,
# whether to keep the largest connected component only (before computing the mask)
RETINANET.KEEP_LARGEST_CC_CLASSES = []

# Optionally, for classes which have exactly one object per image,
# whether to apply some dusting, i.e. remove components with less than N voxels (before computing the mask)
# Note: One threshold per unique class
# Note: If a class idx is larger than the length of this list or N=0, the operation is not done
RETINANET.DUST_THRESHOLDS = []
