from scene_graph_prediction.utils.config import AccessTrackingCfgNode

# -------------------------------------------------------------------------------------------------------------------- #
# ROI Attribute HEADS options
# -------------------------------------------------------------------------------------------------------------------- #

ROI_ATTRIBUTE_HEAD = AccessTrackingCfgNode()
ROI_ATTRIBUTE_HEAD.FEATURE_EXTRACTOR = "FPN2MLPFeatureExtractor"
ROI_ATTRIBUTE_HEAD.PREDICTOR = "FPNPredictor"
ROI_ATTRIBUTE_HEAD.SHARE_BOX_FEATURE_EXTRACTOR = False
# Add attributes to each box
# Note: Choose binary, because cross_entropy loss deteriorate the box head, even with 0.1 weight
ROI_ATTRIBUTE_HEAD.USE_BINARY_LOSS = True
ROI_ATTRIBUTE_HEAD.ATTRIBUTE_LOSS_WEIGHT = 0.1
ROI_ATTRIBUTE_HEAD.ATTRIBUTE_BGFG_SAMPLE = True
ROI_ATTRIBUTE_HEAD.ATTRIBUTE_BGFG_RATIO = 3
ROI_ATTRIBUTE_HEAD.POS_WEIGHT = 5.0
