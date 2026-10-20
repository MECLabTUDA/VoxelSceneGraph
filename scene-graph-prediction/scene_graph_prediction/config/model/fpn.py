from scene_graph_prediction.utils.config import AccessTrackingCfgNode

# -------------------------------------------------------------------------------------------------------------------- #
# FPN options
# -------------------------------------------------------------------------------------------------------------------- #
# Note: the FPN also uses channel counts from Resnet. Be careful when working with 3D data.

FPN = AccessTrackingCfgNode()
FPN.USE_GN = False
FPN.USE_RELU = False
# Apply the post NMS per batch (default) or per image during training
# (default is True to be consistent with Detectron, see Issue #672)
FPN.POST_NMS_PER_BATCH = True
