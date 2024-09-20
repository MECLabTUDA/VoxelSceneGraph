from scene_graph_prediction.utils.config import AccessTrackingCfgNode

# -------------------------------------------------------------------------------------------------------------------- #
# ROI Keypoint HEADS options
# -------------------------------------------------------------------------------------------------------------------- #

ROI_KEYPOINT_HEAD = AccessTrackingCfgNode()
ROI_KEYPOINT_HEAD.FEATURE_EXTRACTOR = "KeypointRCNNFeatureExtractor"
ROI_KEYPOINT_HEAD.PREDICTOR = "KeypointRCNNPredictor"
ROI_KEYPOINT_HEAD.POOLER_RESOLUTION = 16
ROI_KEYPOINT_HEAD.POOLER_RESOLUTION_DEPTH = 2
ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO = 0
ROI_KEYPOINT_HEAD.FEATURE_REPRESENTATION_SIZE = 1024
ROI_KEYPOINT_HEAD.CONV_LAYERS = tuple(512 for _ in range(8))
ROI_KEYPOINT_HEAD.RESOLUTION = 14
ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR = False
