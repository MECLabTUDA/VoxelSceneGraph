from scene_graph_prediction.utils.config import AccessTrackingCfgNode

# -------------------------------------------------------------------------------------------------------------------- #
# ROI Mask HEADS options
# -------------------------------------------------------------------------------------------------------------------- #

ROI_MASK_HEAD = AccessTrackingCfgNode()
ROI_MASK_HEAD.FEATURE_EXTRACTOR = "ResNet50Conv5ROIFeatureExtractor"
ROI_MASK_HEAD.PREDICTOR = "MaskRCNNC4Predictor"
ROI_MASK_HEAD.POOLER_RESOLUTION = 16  # ND Patch length for mask loss computation
ROI_MASK_HEAD.POOLER_RESOLUTION_DEPTH = 4
ROI_MASK_HEAD.POOLER_SAMPLING_RATIO = 0
ROI_MASK_HEAD.FEATURE_REPRESENTATION_SIZE = 1024
ROI_MASK_HEAD.CONV_LAYERS = (256, 256, 256, 256)
ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR = False
# Whether to resize and translate masks to the input image.
ROI_MASK_HEAD.POSTPROCESS_MASKS = False
ROI_MASK_HEAD.POSTPROCESS_MASKS_THRESHOLD = 0.5
# Dilation
ROI_MASK_HEAD.DILATION = 1
# GN
ROI_MASK_HEAD.USE_GN = False
