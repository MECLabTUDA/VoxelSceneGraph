from scene_graph_prediction.utils.config import AccessTrackingCfgNode

# -------------------------------------------------------------------------------------------------------------------- #
# Solver
# -------------------------------------------------------------------------------------------------------------------- #

SOLVER = AccessTrackingCfgNode()
SOLVER.MAX_ITER = 40000
SOLVER.METERS_PERIOD = 200

SOLVER.BASE_LR = 0.002
SOLVER.BIAS_LR_FACTOR = 2

SOLVER.MOMENTUM = 0.9

SOLVER.WEIGHT_DECAY = 0.0005
SOLVER.WEIGHT_DECAY_BIAS = 0.0
SOLVER.CLIP_NORM = 5.0

SOLVER.GAMMA = 0.1
SOLVER.STEPS = (30000,)

SOLVER.WARMUP_FACTOR = 1.0 / 3
SOLVER.WARMUP_ITERS = 500
SOLVER.WARMUP_METHOD = "linear"

SOLVER.SCHEDULE = AccessTrackingCfgNode()
SOLVER.SCHEDULE.TYPE = "WarmupMultiStepLR"  # "WarmupReduceLROnPlateau"
# The following parameters are only used for WarmupReduceLROnPlateau
SOLVER.SCHEDULE.PATIENCE = 2
SOLVER.SCHEDULE.THRESHOLD = 1e-4
SOLVER.SCHEDULE.COOLDOWN = 1
SOLVER.SCHEDULE.FACTOR = 0.5
SOLVER.SCHEDULE.MAX_DECAY_STEP = 7

SOLVER.CHECKPOINT_PERIOD = 2500

SOLVER.GRAD_NORM_CLIP = 5.0

SOLVER.PRINT_GRAD_FREQ = 5000

# Update schedule when load from a previous model:
# If set to True only maintain the iteration number
SOLVER.UPDATE_SCHEDULE_DURING_LOAD = True

# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will see 2 images per batch
SOLVER.IMS_PER_BATCH = 16
