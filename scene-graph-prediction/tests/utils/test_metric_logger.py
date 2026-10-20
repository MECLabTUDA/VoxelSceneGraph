# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import unittest

from scene_graph_prediction.config import cfg
from scene_graph_prediction.utils.metric_logger import MetricLogger


class TestMetricLogger(unittest.TestCase):
    def test_update(self):
        meter = MetricLogger(cfg)
        for i in range(10):
            meter.update(i, metric=float(i))

        m = meter.metrics["metric"]
        self.assertEqual(m.count, 10)
        self.assertEqual(m.total, 45)
        self.assertEqual(m.median, 4.5)
        self.assertEqual(m.avg, 4.5)

    def test_no_attr(self):
        meter = MetricLogger(cfg)
        _ = meter.metrics
        _ = meter.separator

        self.assertRaises(AttributeError, lambda: meter.not_existent)
