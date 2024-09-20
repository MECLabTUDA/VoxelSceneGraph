from scene_graph_prediction.utils.config import AccessTrackingCfgNode

from .backbone import BACKBONE
from .fbnet import FBNET
from .fpn import FPN
from .group_norm import GROUP_NORM
from .resnets import RESNETS
from .retinanet import RETINANET
from .roi_attribute_head import ROI_ATTRIBUTE_HEAD
from .roi_box_head import ROI_BOX_HEAD
from .roi_heads import ROI_HEADS
from .roi_keypoint_head import ROI_KEYPOINT_HEAD
from .roi_mask_head import ROI_MASK_HEAD
from .roi_relation_head import ROI_RELATION_HEAD
from .rpn import RPN
from .vgg import VGG

# -------------------------------------------------------------------------------------------------------------------- #
# Config related to model architectures
# -------------------------------------------------------------------------------------------------------------------- #
# See other files

MODEL = AccessTrackingCfgNode()
MODEL.BACKBONE = BACKBONE
MODEL.FPN = FPN
MODEL.FBNET = FBNET
MODEL.GROUP_NORM = GROUP_NORM
MODEL.RESNETS = RESNETS
MODEL.RETINANET = RETINANET
MODEL.RPN = RPN
MODEL.VGG = VGG
MODEL.ROI_HEADS = ROI_HEADS
MODEL.ROI_ATTRIBUTE_HEAD = ROI_ATTRIBUTE_HEAD
MODEL.ROI_BOX_HEAD = ROI_BOX_HEAD
MODEL.ROI_KEYPOINT_HEAD = ROI_KEYPOINT_HEAD
MODEL.ROI_MASK_HEAD = ROI_MASK_HEAD
MODEL.ROI_RELATION_HEAD = ROI_RELATION_HEAD

# -------------------------------------------------------------------------------------------------------------------- #
# Config related to the model
# -------------------------------------------------------------------------------------------------------------------- #

# Whether we're only training the anchor detector
# Note: this will also trigger the RetinaNet to behave like a two-stage method
MODEL.RPN_ONLY = False

# Whether we're only training the roi heads
# Note: detectors using one-stage object detectors will assume that they have a box head.
# Note: this can be useful when training a one-stage RetinaNet and wanting to train a box head later
#       to have a pretrained feature extractor for relation detection.
# Note: go check MODEL.PRETRAINED_ONE_STAGE_DETECTOR_CKPT.
MODEL.ROI_HEADS_ONLY = False

# When training ROI heads, we only need to crop relevant parts of the feature maps
# So, when we're not training the RPN (or one-stage detector), there is no gradient for the features maps
# and these can be discarded after being cropped.
# So by, computing the feature maps of images one-by-one, and only keeping relevant features,
# we can significantly increase the batch size.
MODEL.OPTIMIZED_ROI_HEADS_PIPELINE = False

# Which region proposal method to use ("RPN", "RetinaNet", "RetinaUNet"...)
# Note: the region proposal choice may be fixed with some detectors.
MODEL.REGION_PROPOSAL = "RPN"
MODEL.MASK_ON = False

# Which Box Head to use.
MODEL.BOX_HEAD = "ROIBoxHead"

# Whether the segmentation should be provided as semantic segmentation
# Note: does not require MASK_ON to be True to enable segmentation-based detection.
# Note: if False and MASK_ON is True, then the segmentation is provided as binary masks
MODEL.REQUIRE_SEMANTIC_SEGMENTATION = False
MODEL.KEYPOINT_ON = False
MODEL.ATTRIBUTE_ON = False
MODEL.RELATION_ON = False
MODEL.DEVICE = "cuda"
MODEL.FLIP_AUG = False
# Choice of detector class
MODEL.META_ARCHITECTURE = "GeneralizedRCNN"
MODEL.CLS_AGNOSTIC_BBOX_REG = False

# If the WEIGHT starts with a catalog://, like :R-50, the code will look for the path in paths_catalog.
# Else, it will use it as the specified absolute path
MODEL.WEIGHT = ""

# Path to the .pth checkpoint file from the object detection training
MODEL.PRETRAINED_DETECTOR_CKPT = ""

# Path to the .pth checkpoint file from the one-stage object detection training
# Note: go check MODEL.ROI_HEADS_ONLY.
MODEL.PRETRAINED_ONE_STAGE_DETECTOR_CKPT = ""

# Whether the IMPORTANCE field should be used to weight the detection of boxes.
# Note: the importance is defaulted to (1 + #rels implicating this object) if not supplied.
MODEL.WEIGHTED_BOX_TRAINING = False
