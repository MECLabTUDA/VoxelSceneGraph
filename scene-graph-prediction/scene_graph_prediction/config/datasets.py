from scene_graph_prediction.utils.config import AccessTrackingCfgNode

# -------------------------------------------------------------------------------------------------------------------- #
# Config for selecting train/test datasets
# -------------------------------------------------------------------------------------------------------------------- #

DATASETS = AccessTrackingCfgNode()
# List of the dataset names for training, as present in paths_catalog.py
DATASETS.TRAIN = ()
# List of the dataset names for val, as present in paths_catalog.py
# Note that except dataset names, all remaining val configs reuse those of test
DATASETS.VAL = ()
# List of the dataset names for testing, as present in paths_catalog.py
DATASETS.TEST = ()

# Fold for the KFoldSpliter (if used)
DATASETS.FOLD = 0

# COCO
# Evaluation parameters for COCO, see EVALUATION_PARAMETERS registry in data.evaluation.utils
DATASETS.EVALUATION_PARAMETERS = "default"

# SGG
# If not "", specifies the name of the statistics data that will be used (instead of the stats of the training data)
# Note: it has to be computed beforehand
# Note: can be useful for Federated Learning, where the stats have been computed across clients beforehand
DATASETS.STATISTICS_OVERRIDE = ""
