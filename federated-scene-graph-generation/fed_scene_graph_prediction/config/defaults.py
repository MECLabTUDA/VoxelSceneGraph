from scene_graph_prediction.config import cfg as _c
from scene_graph_prediction.utils.config import AccessTrackingCfgNode

# -------------------------------------------------------------------------------------------------------------------- #
# Config definition
# -------------------------------------------------------------------------------------------------------------------- #
# Checkout the other files in this module to learn more about keys for each specific application
_c = _c

# -------------------------------------------------------------------------------------------------------------------- #
# Federated-learning training options
# -------------------------------------------------------------------------------------------------------------------- #


# -------------------------------------------------------------------------------------------------------------------- #
# Federated-learning training options
# -------------------------------------------------------------------------------------------------------------------- #

_c.FEDERATED_LEARNING = AccessTrackingCfgNode()
# Overall, we train over SOLVER.STEPS number of steps, which are divided over rounds
# This option determines the number of steps per round.
# The number of rounds is SOLVER.STEPS // FEDERATED_LEARNING.STEPS_PER_ROUND
_c.FEDERATED_LEARNING.STEPS_PER_ROUND = 10

# Aggregator to use for Federated Learning
_c.FEDERATED_LEARNING.AGGREGATOR = "FedAvgAggregator"

# Learning rate for server optimizers (other than FedAvg)
_c.FEDERATED_LEARNING.SERVER_OPT_BASE_LR = .1

# Path to the Theoden global context file (if required)
_c.FEDERATED_LEARNING.GLOBAL_CONTEXT = ""

# Communication address for Theoden (if required)
_c.FEDERATED_LEARNING.COMMUNICATION_ADDRESS = ""

# -------------------------------------------------------------------------------------------------------------------- #
# Federated-learning client options
# -------------------------------------------------------------------------------------------------------------------- #
# The only options that should change from one client to the other are the datasets used,
# and optionally the batch size, a local path to save some results.
# Each client should be configured using its name. For the expected structure, see th comment further below.
_c.CLIENTS = AccessTrackingCfgNode(new_allowed=True)

# Here is an example of client being configured in YAML:
# CLIENTS:
#     my_client_name:
#         DATASETS:
#             TRAIN:
#               - train_data
#             VAL:
#               - val_data
#             TEST:
#               - test_data
#         SOLVER:
#             IMS_PER_BATCH: 1
#         TEST:
#             IMS_PER_BATCH: 1
#         MODEL:
#             DEVICE: "cuda:1"
#         OUTPUT_DIR: "some local path"
# FIXME There is no auto-cast for CLIENTS.name.DATASETS. ... dataset lists
