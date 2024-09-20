from scene_graph_prediction.utils.config import AccessTrackingCfgNode

# -------------------------------------------------------------------------------------------------------------------- #
# Backbone options
# -------------------------------------------------------------------------------------------------------------------- #
BACKBONE = AccessTrackingCfgNode()

# The backbone conv body to use
# The string must match a function that is imported in modeling.model_builder
# (e.g., "FPN.add_fpn_ResNet101_conv5_body" to specify a ResNet-101-FPN
# backbone)
BACKBONE.CONV_BODY = "R-50-C4"

# Add StopGrad at a specified stage so the bottom layers are frozen
BACKBONE.FREEZE_CONV_BODY_AT = 2
