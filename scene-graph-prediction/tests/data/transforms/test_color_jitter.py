import unittest

from yacs.config import CfgNode

from scene_graph_prediction.data.transforms import ColorJitter


class TestResizeImage2D(unittest.TestCase):
    def setUp(self):
        # Build the config
        cfg = CfgNode()
        cfg.INPUT = CfgNode()
        cfg.INPUT.BRIGHTNESS = 5
        cfg.INPUT.CONTRAST = 25
        cfg.INPUT.SATURATION = 35
        cfg.INPUT.HUE = 45
        self.cfg = cfg

    def test_build_train(self):
        transform = ColorJitter.build(self.cfg, is_train=True)
        self.assertEqual([0, self.cfg.INPUT.BRIGHTNESS + 1], transform.brightness)
        self.assertEqual([0, self.cfg.INPUT.CONTRAST + 1], transform.contrast)
        self.assertEqual([0, self.cfg.INPUT.SATURATION + 1], transform.saturation)
        self.assertEqual([-self.cfg.INPUT.HUE, self.cfg.INPUT.HUE], transform.hue)

    def test_build_test(self):
        transform = ColorJitter.build(self.cfg, is_train=False)
        self.assertEqual(None, transform.brightness)
        self.assertEqual(None, transform.contrast)
        self.assertEqual(None, transform.saturation)
        self.assertEqual(None, transform.hue)
