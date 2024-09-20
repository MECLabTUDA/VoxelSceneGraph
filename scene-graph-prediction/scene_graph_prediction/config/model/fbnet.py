from scene_graph_prediction.utils.config import AccessTrackingCfgNode

# -------------------------------------------------------------------------------------------------------------------- #
# FBNet options
# -------------------------------------------------------------------------------------------------------------------- #
FBNET = AccessTrackingCfgNode()
FBNET.ARCH = "default"
# custom arch
FBNET.BN_TYPE = "bn"
FBNET.SCALE_FACTOR = 1.0
# the output channels will be divisible by WIDTH_DIVISOR
FBNET.WIDTH_DIVISOR = 1
FBNET.DW_CONV_SKIP_BN = True
FBNET.DW_CONV_SKIP_RELU = True

# > 0 scale, == 0 skip, < 0 same dimension
FBNET.DET_HEAD_LAST_SCALE = 1.0
FBNET.DET_HEAD_BLOCKS = []
# overwrite the stride for the head, 0 to use original value
FBNET.DET_HEAD_STRIDE = 0

# > 0 scale, == 0 skip, < 0 same dimension
FBNET.KPTS_HEAD_LAST_SCALE = 0.0
FBNET.KPTS_HEAD_BLOCKS = []
# overwrite the stride for the head, 0 to use original value
FBNET.KPTS_HEAD_STRIDE = 0

# > 0 scale, == 0 skip, < 0 same dimension
FBNET.MASK_HEAD_LAST_SCALE = 0.0
FBNET.MASK_HEAD_BLOCKS = []
# overwrite the stride for the head, 0 to use original value
FBNET.MASK_HEAD_STRIDE = 0

# 0 to use all blocks defined in arch_def
FBNET.RPN_HEAD_BLOCKS = 0
FBNET.RPN_BN_TYPE = ""
