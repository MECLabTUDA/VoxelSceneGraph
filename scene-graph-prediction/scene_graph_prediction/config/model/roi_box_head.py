from scene_graph_prediction.utils.config import AccessTrackingCfgNode

# -------------------------------------------------------------------------------------------------------------------- #
# ROI Box HEADS options
# -------------------------------------------------------------------------------------------------------------------- #

ROI_BOX_HEAD = AccessTrackingCfgNode()
# Feature extractor
ROI_BOX_HEAD.FEATURE_EXTRACTOR = "FPN2MLPFeatureExtractor"
# Predictor class (box classification and regression)
ROI_BOX_HEAD.PREDICTOR = "BoxPredictor"
# Loss to use for box regression
ROI_BOX_HEAD.REGRESSION_LOSS = "L1Loss"
# Width/height resolution for extracted box patches
ROI_BOX_HEAD.POOLER_RESOLUTION = 16
# Depth resolution for extracted box patches
ROI_BOX_HEAD.POOLER_RESOLUTION_DEPTH = 4
# Sampling ratio for ROIAlign layers
ROI_BOX_HEAD.POOLER_SAMPLING_RATIO = 0

# Whether to add GT boxes to predicted ones for relation detection training
ROI_BOX_HEAD.ADD_GTBOX_TO_PROPOSAL_IN_TRAIN = False
# Whether to remove predicted objects that have no matching groundtruth object
ROI_BOX_HEAD.REMOVE_OBJ_NO_MATCH = False
# Siwe of the 1D feature representation produced by the feature extractor and fed to the predictor
ROI_BOX_HEAD.FEATURE_REPRESENTATION_SIZE = 2048

# Whether to add a group norm layer to feature extractors that support it
ROI_BOX_HEAD.USE_GN = False
# Dilation
ROI_BOX_HEAD.DILATION = 1
ROI_BOX_HEAD.CONV_HEAD_DIM = 256
ROI_BOX_HEAD.NUM_STACKED_CONVS = 4
