from scene_graph_prediction.utils.config import AccessTrackingCfgNode

# -------------------------------------------------------------------------------------------------------------------- #
# DataLoader
# -------------------------------------------------------------------------------------------------------------------- #
DATALOADER = AccessTrackingCfgNode()
# Number of data loading threads
DATALOADER.NUM_WORKERS = 4
# If > 0, this enforces that each image in the collated batch should have a size divisible by SIZE_DIVISIBILITY
DATALOADER.SIZE_DIVISIBILITY = (0, 0, 0)

# -------------------------------------------------------------------------------------------------------------------- #
# Options for sampling, these are mutually exclusive

# If True, each batch should contain only images for which the aspect ratio is compatible.
# This groups portrait images together, and landscape images are not batched with portrait images.
# Note: only implemented for 2D
DATALOADER.ASPECT_RATIO_GROUPING = False

# If True, a KnowledgeGuidedObjectSampler will be used.
# Check its doc to understand the expected BoxList fields.
# Note: can only be used for training object detectors
# Note: cannot be used with distributed training (not tested at least)
# Requires:
# - not cfg.MODEL.RELATION_ON
# - cfg.MODEL.WEIGHTED_BOX_TRAINING
# - cfg.INPUT.N_IMG_ATT_CLASSES > 1
# - cfg.INPUT.N_ATT_CLASSES == cfg.INPUT.N_IMG_ATT_CLASSES if STRICT_SAMPLING
DATALOADER.KNOWLEDGE_GUIDED_BOX_GROUPING = False

# If True, a KnowledgeGuidedRelationSampler will be used.
# Note: can only be used for training relation predictors
# Note: cannot be used with distributed training (not tested at least)
# Requires:
# - cfg.MODEL.RELATION_ON
# - cfg.INPUT.N_REL_CLASSES > 2
DATALOADER.KNOWLEDGE_GUIDED_RELATION_GROUPING = False

# -------------------------------------------------------------------------------------------------------------------- #
# Has to be used with either KNOWLEDGE_GUIDED_BOX_GROUPING or KNOWLEDGE_GUIDED_RELATION_GROUPING.
# If True, we only train on objects/relation for the selected group rather than on the entire image annotation.
DATALOADER.STRICT_SAMPLING = False

# Has to be used with either KNOWLEDGE_GUIDED_BOX_GROUPING or KNOWLEDGE_GUIDED_RELATION_GROUPING.
# How many batches should be generated before selecting the next group
DATALOADER.ITER_PER_GROUP = 10
