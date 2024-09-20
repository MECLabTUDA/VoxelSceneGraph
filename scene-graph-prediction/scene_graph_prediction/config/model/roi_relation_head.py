from scene_graph_prediction.utils.config import AccessTrackingCfgNode

# -------------------------------------------------------------------------------------------------------------------- #
# ROI Relation HEADS options
# -------------------------------------------------------------------------------------------------------------------- #

ROI_RELATION_HEAD = AccessTrackingCfgNode()

# Select the Relationship Model
ROI_RELATION_HEAD.PREDICTOR = "MotifPredictor"
# Whether the learning rate should be reduced for the predictor
# Note: the only predictor that used that was "IMPPredictor"
ROI_RELATION_HEAD.IS_SLOW_PREDICTOR_HEAD = False

# Model for feature extraction, usually the same as for the Box head
ROI_RELATION_HEAD.FEATURE_EXTRACTOR = "RelationFeatureExtractor"
# Model for feature extraction from subject+object mask pairs
ROI_RELATION_HEAD.MASK_FEATURE_EXTRACTOR = "Factor444RelationMaskFeatureExtractor"
# Whether to use all feature levels to do pooling (and extract object features)
ROI_RELATION_HEAD.POOLING_ALL_LEVELS = True

# Batch size used for loss computation
ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE = 64

# Weight of positive cases compared to false positive ones
ROI_RELATION_HEAD.POSITIVE_WEIGHT = .25

# Whether to use groundtruth boxes for relation prediction
ROI_RELATION_HEAD.USE_GT_BOX = True
# Whether to use groundtruth box labels for relation prediction
# Note: Requires USE_GT_BOX=True
ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL = True

# Whether to use the LabelSmoothingRegression loss instead of the cross-entropy loss
ROI_RELATION_HEAD.LABEL_SMOOTHING_LOSS = False

ROI_RELATION_HEAD.PREDICT_USE_BIAS = True
# Note: For sgdet, during training, only train pairs with overlap
ROI_RELATION_HEAD.REQUIRE_BOX_OVERLAP = True

# When sampling learning examples from detections,
# we might find multiple detected object pairs that could correspond to a given groundtruth relation
# This is the max number of matches for a groundtruth relation
ROI_RELATION_HEAD.NUM_SAMPLE_PER_GT_REL = 4

# When computing features for a relation (ROIRelationFeatureExtractor),
# we normally use rectangular masks to locate both objects.
# However, one can choose to use predicted binary masks instead (as it's more precise).
ROI_RELATION_HEAD.PREDICT_USE_MASKS = False

# By default, the relation head will reclassify objects
# This option is here to disable that, e.g. because this head gets confused by significant object overlap
ROI_RELATION_HEAD.DISABLE_RECLASSIFICATION = False
# For sampling, which pair of objects might have a relation,
# enforce that the label combination has to be present in the training data.
# I.e. no zero-shot (sub, ob) pairs.
# Note: to learn sub/ob ordering, also allow pairs where the reverse ordering is found in the groundtruth
# Note: requires DISABLE_RECLASSIFICATION=True.
ROI_RELATION_HEAD.REQUIRE_REL_IN_TRAIN = False

# Embedding size, needs to match the vector size of the embedding data used
# E.g. GloVe6b has a vector size of 200
ROI_RELATION_HEAD.EMBED_DIM = 200
ROI_RELATION_HEAD.CONTEXT_DROPOUT_RATE = 0.2
ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM = 512
ROI_RELATION_HEAD.CONTEXT_POOLING_DIM = 4096
ROI_RELATION_HEAD.CONTEXT_OBJ_LAYER = 1  # assert >= 1
ROI_RELATION_HEAD.CONTEXT_REL_LAYER = 1  # assert >= 1

# -------------------------------------------------------------------------------------------------------------------- #
# Transformer options
# -------------------------------------------------------------------------------------------------------------------- #
ROI_RELATION_HEAD.TRANSFORMER = AccessTrackingCfgNode()

ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE = 0.1
ROI_RELATION_HEAD.TRANSFORMER.OBJ_LAYER = 4
ROI_RELATION_HEAD.TRANSFORMER.REL_LAYER = 2
ROI_RELATION_HEAD.TRANSFORMER.NUM_HEAD = 8
ROI_RELATION_HEAD.TRANSFORMER.INNER_DIM = 2048
ROI_RELATION_HEAD.TRANSFORMER.KEY_DIM = 64
ROI_RELATION_HEAD.TRANSFORMER.VAL_DIM = 64

# -------------------------------------------------------------------------------------------------------------------- #
# MOTIFS options
# -------------------------------------------------------------------------------------------------------------------- #
ROI_RELATION_HEAD.MOTIFS = AccessTrackingCfgNode()
# We need to order boxes in a sequence, so we need a strategy:
# - "width": sort by increasing center width
# - "height": sort by increasing center height
# - "depth": sort by increasing center depth
# - "volume:  sort by increasing volume
ROI_RELATION_HEAD.MOTIFS.SORTING_STRATEGY = "width"

# -------------------------------------------------------------------------------------------------------------------- #
# Causal Analysis options
# -------------------------------------------------------------------------------------------------------------------- #
ROI_RELATION_HEAD.CAUSAL = AccessTrackingCfgNode()
# Direct and indirect effect analysis
ROI_RELATION_HEAD.CAUSAL.EFFECT_ANALYSIS = False
# Fusion
ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE = "sum"  # "sum", "gate"
# Causal context feature layer
ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER = "motifs"  # "motifs", "vctree, "vtranse"

ROI_RELATION_HEAD.CAUSAL.SPATIAL_FOR_VISION = False

ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE = "none"  # "TDE", "TIE", "TE", "none"
