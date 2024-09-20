from scene_graph_prediction.utils.config import AccessTrackingCfgNode

# -------------------------------------------------------------------------------------------------------------------- #
# ROI HEADS options
# -------------------------------------------------------------------------------------------------------------------- #
ROI_HEADS = AccessTrackingCfgNode()
# Overlap threshold for an RoI to be considered foreground (if >= FG_IOU_THRESHOLD)
ROI_HEADS.FG_IOU_THRESHOLD = 0.5
# Overlap threshold for an RoI to be considered background
# (class = 0 if overlap in [0, BG_IOU_THRESHOLD))
ROI_HEADS.BG_IOU_THRESHOLD = 0.3

# Default weights on (dx, dy, dw, dh) for normalizing bbox regression targets
# These are empirically chosen to approximately lead to unit variance targets
ROI_HEADS.BBOX_REG_WEIGHTS = (10., 10., 5., 5.)
# RoI minibatch size *per image* (number of regions of interest [ROIs])
# Total number of RoIs per training minibatch = TRAIN.BATCH_SIZE_PER_IM * TRAIN.IMS_PER_BATCH
# E.g. a common configuration is: 512 * 2 * 8 = 8192
ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
# Target fraction of RoI minibatch that is labeled foreground (i.e. class > 0)
ROI_HEADS.POSITIVE_FRACTION = 0.25

# Only used on test mode

# Minimum score threshold (assuming scores in a [0, 1] range); a value chosen to
# balance obtaining high recall with not having too many low precision
# detections that will slow down inference post-processing steps (like NMS)
ROI_HEADS.SCORE_THRESH = 0.01
# Overlap threshold used for non-maximum suppression (suppress boxes with
# IoU >= this threshold)
ROI_HEADS.NMS = 0.3
ROI_HEADS.POST_NMS_PER_CLS_TOPN = 300
# Remove duplicated assigned labels for a single bbox in nms
ROI_HEADS.NMS_FILTER_DUPLICATES = False
# Maximum number of detections to return per image (100 is based on the limit established for the COCO dataset)
ROI_HEADS.DETECTIONS_PER_IMG = 100
