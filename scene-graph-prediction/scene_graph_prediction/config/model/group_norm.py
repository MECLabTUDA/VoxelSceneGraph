from scene_graph_prediction.utils.config import AccessTrackingCfgNode

# -------------------------------------------------------------------------------------------------------------------- #
# Group Norm options
# -------------------------------------------------------------------------------------------------------------------- #
GROUP_NORM = AccessTrackingCfgNode()
# Number of dimensions per group in GroupNorm (-1 if using NUM_GROUPS)
GROUP_NORM.DIM_PER_GP = -1
# Number of groups in GroupNorm (-1 if using DIM_PER_GP)
# Default 2D: 32; Default 3D: 16
GROUP_NORM.NUM_GROUPS = 16
# GroupNorm's small constant in the denominator
GROUP_NORM.EPSILON = 1e-5
