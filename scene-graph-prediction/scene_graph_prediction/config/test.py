from scene_graph_prediction.utils.config import AccessTrackingCfgNode

# -------------------------------------------------------------------------------------------------------------------- #
# Specific test options
# -------------------------------------------------------------------------------------------------------------------- #
TEST = AccessTrackingCfgNode()

# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will have 2 images per batch
TEST.IMS_PER_BATCH = 8

# Whether validate and validate period
TEST.DO_VAL = True
TEST.VAL_PERIOD = 2500

# Do an evaluation round when starting (or resuming training)
TEST.DO_PRETRAIN_VAL = True

# Whether to compute a loss for the validation data when validating
TEST.TRACK_VAL_LOSS = False

# Whether to save predicted BoxLists
# Note: these are not compressed and can be heavy. So, one might want to disable saving the predictions.
TEST.SAVE_BOXLISTS = True
# Whether to save predicted segmentations (validation and test set) as Nifti (.nii.gz) files
# Note: one could extract them from BoxLists, but this is more convenient. Also, these files weigh next to nothing.
TEST.SAVE_SEGMENTATIONS = True

# -------------------------------------------------------------------------------------------------------------------- #
# Test-time augmentations
# -------------------------------------------------------------------------------------------------------------------- #
TEST.BBOX_AUG = AccessTrackingCfgNode()

# Enable test-time augmentation for bounding box detection if True
# Note: not supported if RELATION_ON
TEST.BBOX_AUG.ENABLED = False

# Horizontal flip at the original scale (id transform)
# Note: requires TEST.BBOX_AUG.ENABLED to be True
TEST.BBOX_AUG.H_FLIP = False

# Each scale is the pixel size of an image's shortest side
# Note: requires TEST.BBOX_AUG.ENABLED to be True
# TEST.BBOX_AUG.SCALES = ()

# Horizontal flip augmentation at test time
# Note: requires TEST.BBOX_AUG.ENABLED to be True
TEST.BBOX_AUG.SCALE_H_FLIP = False

# -------------------------------------------------------------------------------------------------------------------- #
# Settings for object-detection testing testing
# -------------------------------------------------------------------------------------------------------------------- #
TEST.COCO = AccessTrackingCfgNode()

# Whether to also output metrics for individual object classes for COCO eval
TEST.COCO.PER_CATEGORY_METRICS = False

# -------------------------------------------------------------------------------------------------------------------- #
# Settings for relation testing
# -------------------------------------------------------------------------------------------------------------------- #
TEST.RELATION = AccessTrackingCfgNode()

# IoU threshold for detection individual objects in a relation. Use thr=-1 to do Phrase Detection.
TEST.RELATION.IOU_THRESHOLD = 0.5

# List of top-Ks to use for the evaluation
TEST.RELATION.RECALL_AT_K = [10, 20, 50, 100]

# For no graph constraint, we can consider upto this number of predictions per (sub, ob) pair
TEST.RELATION.NO_GRAPH_CONSTRAINT_TOP_N_RELATION = 2

# When predicting the label of a bbox, run nms on each cls
TEST.RELATION.LATER_NMS_PREDICTION_THRES = 0.3

# Synchronize_gather after each batch at inference rather than for the whole dataset
# Note: used to be used for sgdet, otherwise test on multi-gpu will cause out of memory
TEST.RELATION.SYNC_GATHER = False

# If an object detector is trained for relation detection, then it can be important to know the best possible
# performance that can be achieved, given a "perfect" relation detector
# Indeed, if a box is not detected, no related relations can be detected
# Note: This code requires the groundtruth annotation to contain the RELATIONS field.
# Note: reflexive relations are not considered.
TEST.RELATION.COMPUTE_RELATION_UPPER_BOUND = False
# When computing the upper bound, we can simulate that the relation detector can reclassify objects
# by doing object class agnostic matching. However, this is not perfect and can cause issues with large objects.
# E.g. a relation is matched by using a large object twice and reclassifying it two different ways...
TEST.RELATION.UPPER_BOUND_ALLOW_RECLASSIFICATION = True

# -------------------------------------------------------------------------------------------------------------------- #
# Settings for Scene Graph Relation testing, where we replace parts of the prediction with groundtruth
# -------------------------------------------------------------------------------------------------------------------- #

# Remove objects with no groundtruth match
TEST.RELATION.REMOVE_FALSE_POSITIVES = False
# Coordinates of predicted objects having a match with a GT object are replaced with GT coordinates
TEST.RELATION.REPLACE_MATCHED_BOXES = False
# Replace the semantic segmentation / binary masks
TEST.RELATION.REPLACE_SEGMENTATION = False
