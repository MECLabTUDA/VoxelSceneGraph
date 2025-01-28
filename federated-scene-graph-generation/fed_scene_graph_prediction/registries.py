from typing import Callable
from scene_graph_prediction.utils.registry import Registry
from theoden.operations import Aggregator, MedianAggregator, FedAvgAggregator, FedSGDServerOptimizer, \
    FedOptAggregator, FedAdamServerOptimizer, FedYogiServerOptimizer
from fed_scene_graph_prediction.config import cfg

# Federated-Learning Aggregators
AGGREGATORS: Registry[str, Callable[[type | None], Aggregator]] = Registry()
# These are manually maintained
AGGREGATORS.register(MedianAggregator.__name__, MedianAggregator)
AGGREGATORS.register(FedAvgAggregator.__name__, FedAvgAggregator)
AGGREGATORS.register("FedAdam", lambda: FedOptAggregator(FedAdamServerOptimizer(lr=cfg.FEDERATED_LEARNING.SERVER_OPT_BASE_LR)))
AGGREGATORS.register("FedSGD", lambda: FedOptAggregator(FedSGDServerOptimizer(lr=cfg.FEDERATED_LEARNING.SERVER_OPT_BASE_LR)))
AGGREGATORS.register("FedYogi", lambda: FedOptAggregator(FedYogiServerOptimizer(lr=cfg.FEDERATED_LEARNING.SERVER_OPT_BASE_LR)))
