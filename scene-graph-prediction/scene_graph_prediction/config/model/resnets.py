from scene_graph_prediction.utils.config import AccessTrackingCfgNode

# -------------------------------------------------------------------------------------------------------------------- #
# ResNe[X]t options (ResNets = {ResNet, ResNeXt}
# Note: that parts of a ResNet may be used for both the backbone and the head
# These options apply to both
# -------------------------------------------------------------------------------------------------------------------- #
RESNETS = AccessTrackingCfgNode()

# Number of groups to use; 1 ==> ResNet; > 1 ==> ResNeXt
RESNETS.NUM_GROUPS = 1

# Baseline width of each group (2D: 64, 3D: 16)
RESNETS.WIDTH_PER_GROUP = 64

# Place the stride 2 conv on the 1x1 filter
# Use True only for the original MSRA ResNet; use False for C2 and Torch models
RESNETS.STRIDE_IN_1X1 = True

# Residual transformation function
RESNETS.TRANS_FUNC = "BottleneckWithFixedBatchNorm"  # "BottleneckWithFixedBatchNorm" or "BottleneckWithGN"
# ResNet's stem function (conv1 and pool1)
RESNETS.STEM_FUNC = "StemWithFixedBatchNorm"  # "StemWithFixedBatchNorm" or "StemWithGN"

# Apply dilation in stage "res5"
RESNETS.RES5_DILATION = 1

# (2D: 256 * 4, 3D: 256)
RESNETS.BACKBONE_OUT_CHANNELS = 256 * 4
# (2D: 256, 3D: 64)
RESNETS.RES2_OUT_CHANNELS = 256
# (2D: 64, 3D: 16)
RESNETS.STEM_OUT_CHANNELS = 64

RESNETS.STAGE_WITH_DCN = (False, False, False, False)  # Where to use DCN
RESNETS.WITH_MODULATED_DCN = False  # Use DeformConv or ModulatedDeformConv
RESNETS.DEFORMABLE_GROUPS = 1
