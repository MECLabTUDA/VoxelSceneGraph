from scene_graph_prediction.utils.config import AccessTrackingCfgNode

# -------------------------------------------------------------------------------------------------------------------- #
# RPN options
# -------------------------------------------------------------------------------------------------------------------- #
RPN = AccessTrackingCfgNode()
RPN.RPN_MID_CHANNEL = 512
# Base RPN anchor sizes given in absolute pixels
# Note: tuple[int, ...] in RPN mode of length n_anchors
# Note: tuple[tuple[int, ...], ...] in FPN mode of shape n_features_maps x n_anchors_per_lvl
RPN.ANCHOR_SIZES = (16, 32, 64, 128, 256, 512)
# RPN anchor aspect ratios for anchor sizes
RPN.ASPECT_RATIOS = (1.0, .5, 2.0)
# Base RPN anchor depth given in absolute pixels (only used for 3D)
# Note: tuple[int, ...] in RPN mode of length n_anchors
# Note: tuple[tuple[int, ...], ...] in FPN mode of shape n_features_maps x n_anchors_per_lvl
#       It is critical that there is a depth for each anchor size
# Note: There is currently no support for only defining custom anchors and no anchor size
RPN.ANCHOR_DEPTHS = (2, 4, 8, 16, 32)
# RPN anchor shapes as is (no ratios applied)
# Note: tuple[tuple[int, ...], ...] of shape n_anchors x n_dim
# Note: tuple[tuple[tuple[int, ...], ...], ...] in FPN mode of shape n_features_maps x n_anchors_per_lvl x n_dim
RPN.CUSTOM_ANCHORS = ()
# Remove RPN anchors that go outside the image by RPN_STRADDLE_THRESH pixels
# Set to -1 or a large value, e.g. 100000, to disable pruning anchors
RPN.STRADDLE_THRESH = 0

# ======================================================================================================================
# Matching
# Name of the matcher class to use: IoUMatcher or ATSSMatcher
RPN.MATCHER = "ATSSMatcher"

# IoU Matcher
# Minimum overlap required between an anchor and ground-truth box for the
# (anchor, gt box) pair to be a positive example (IoU >= FG_IOU_THRESHOLD ==> positive RPN example)
RPN.FG_IOU_THRESHOLD = 0.7
# Maximum overlap allowed between an anchor and ground-truth box for the
# (anchor, gt box) pair to be a negative examples (IoU < BG_IOU_THRESHOLD ==> negative RPN example)
RPN.BG_IOU_THRESHOLD = 0.3

# ATSS Matcher
# Max number of candidate anchors per feature map level
RPN.ATSS_NUM_CANDIDATES = 4
# ======================================================================================================================

# Total number of RPN examples per image
RPN.BATCH_SIZE_PER_IMAGE = 256
# Target fraction of foreground (positive) examples per RPN minibatch
RPN.POSITIVE_FRACTION = 0.5
# Number of top scoring RPN proposals to keep before applying NMS
# When FPN is used, this is *per FPN level* (not total)
RPN.PRE_NMS_TOP_N_TRAIN = 12000
RPN.PRE_NMS_TOP_N_TEST = 6000
# Number of top scoring RPN proposals to keep after applying NMS
# Also used for FPN
RPN.POST_NMS_TOP_N_TRAIN = 2000
RPN.POST_NMS_TOP_N_TEST = 1000
# NMS threshold used on RPN proposals
RPN.NMS_THRESH = 0.7
# Custom rpn head, empty to use default conv or separable conv
RPN.RPN_HEAD = "SingleConvRPNHead"

# Add gt boxes to RPN proposals
# Note: Used only for training the relation head.
# Note: To make sure the detector won't be missing any ground truth bbox,
#       we add ground truth box to the output of RPN proposals during training
RPN.ADD_GTBOX_TO_PROPOSAL_IN_TRAIN = False
