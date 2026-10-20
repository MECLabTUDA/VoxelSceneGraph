import unittest

import torch

from scene_graph_prediction.modeling.utils.sampling import BalancedSampler


# noinspection DuplicatedCode
class TestBalancedSampler(unittest.TestCase):
    def _check_sampled_labels(self, labels: torch.Tensor, pos_mask: torch.Tensor, neg_mask: torch.Tensor):
        if pos_mask.sum() > 0:
            self.assertTrue(torch.all(labels[pos_mask]) == 1)
        if neg_mask.sum() > 0:
            self.assertTrue(torch.all(labels[neg_mask]) == 0)

    def test_ratio_0_no_pred(self):
        sampler = BalancedSampler(1024, 0.)
        labels = torch.zeros(0)
        [pos_mask], [neg_mask] = sampler([labels])
        self.assertEqual(0, pos_mask.sum())
        self.assertEqual(0, neg_mask.sum())
        self._check_sampled_labels(labels, pos_mask, neg_mask)

    def test_ratio_0_enough_negatives(self):
        sampler = BalancedSampler(1024, 0.)
        labels = torch.zeros(1024)
        [pos_mask], [neg_mask] = sampler([labels])
        self.assertEqual(0, pos_mask.sum())
        self.assertEqual(1024, neg_mask.sum())
        self._check_sampled_labels(labels, pos_mask, neg_mask)

    def test_ratio_0_too_many_negatives(self):
        sampler = BalancedSampler(1024, 0.)
        labels = torch.zeros(2048)
        [pos_mask], [neg_mask] = sampler([labels])
        self.assertEqual(0, pos_mask.sum())
        self.assertEqual(1024, neg_mask.sum())
        self._check_sampled_labels(labels, pos_mask, neg_mask)

    def test_ratio_0_positive_ignored(self):
        sampler = BalancedSampler(1024, 0.)
        labels = torch.zeros(1024)
        labels[0] = 1
        [pos_mask], [neg_mask] = sampler([labels])
        self.assertEqual(0, pos_mask.sum())
        self.assertEqual(1023, neg_mask.sum())
        self._check_sampled_labels(labels, pos_mask, neg_mask)

    def test_ratio_0_no_negatives(self):
        sampler = BalancedSampler(1024, 0.)
        labels = torch.ones(1024)
        [pos_mask], [neg_mask] = sampler([labels])
        self.assertEqual(0, pos_mask.sum())
        self.assertEqual(0, neg_mask.sum())
        self._check_sampled_labels(labels, pos_mask, neg_mask)

    def test_ratio_0_no_positives(self):
        sampler = BalancedSampler(1024, 0.)
        labels = torch.zeros(1024)
        [pos_mask], [neg_mask] = sampler([labels])
        self.assertEqual(0, pos_mask.sum())
        self.assertEqual(1024, neg_mask.sum())
        self._check_sampled_labels(labels, pos_mask, neg_mask)

    def test_ratio_1_enough_positives(self):
        sampler = BalancedSampler(1024, 1.)
        labels = torch.ones(1024)
        [pos_mask], [neg_mask] = sampler([labels])
        self.assertEqual(1024, pos_mask.sum())
        self.assertEqual(0, neg_mask.sum())
        self._check_sampled_labels(labels, pos_mask, neg_mask)

    def test_ratio_1_too_many_positives(self):
        sampler = BalancedSampler(1024, 1.)
        labels = torch.ones(2048)
        [pos_mask], [neg_mask] = sampler([labels])
        self.assertEqual(1024, pos_mask.sum())
        self.assertEqual(0, neg_mask.sum())
        self._check_sampled_labels(labels, pos_mask, neg_mask)

    def test_ratio_1_negative_ignored(self):
        sampler = BalancedSampler(1024, 1.)
        labels = torch.ones(1024)
        labels[1:3] = 0
        [pos_mask], [neg_mask] = sampler([labels])
        self.assertEqual(1022, pos_mask.sum())
        self.assertEqual(sampler.min_neg, neg_mask.sum())  # At least min_neg
        self._check_sampled_labels(labels, pos_mask, neg_mask)

    def test_ratio_1_no_positives(self):
        sampler = BalancedSampler(1024, 1.)
        labels = torch.zeros(1024)
        [pos_mask], [neg_mask] = sampler([labels])
        self.assertEqual(0, pos_mask.sum())
        self.assertEqual(sampler.min_neg, neg_mask.sum())  # At least min_neg
        self._check_sampled_labels(labels, pos_mask, neg_mask)

    def test_ratio_1_no_negatives(self):
        sampler = BalancedSampler(1024, 1.)
        labels = torch.ones(1024)
        [pos_mask], [neg_mask] = sampler([labels])
        self.assertEqual(1024, pos_mask.sum())
        self.assertEqual(0, neg_mask.sum())
        self._check_sampled_labels(labels, pos_mask, neg_mask)

    def test_ratio_05_enough(self):
        sampler = BalancedSampler(1024, .5)
        labels = torch.cat([torch.zeros(512), torch.ones(512)])
        [pos_mask], [neg_mask] = sampler([labels])
        self.assertEqual(512, pos_mask.sum())
        self.assertEqual(512, neg_mask.sum())
        self._check_sampled_labels(labels, pos_mask, neg_mask)

    def test_ratio_05_too_many(self):
        sampler = BalancedSampler(1024, .5)
        labels = torch.cat([torch.zeros(1024), torch.ones(1024)])
        [pos_mask], [neg_mask] = sampler([labels])
        self.assertEqual(512, pos_mask.sum())
        self.assertEqual(512, neg_mask.sum())
        self._check_sampled_labels(labels, pos_mask, neg_mask)

    def test_ratio_05_no_pos_at_least_min_neg(self):
        sampler = BalancedSampler(1024, .5)
        labels = torch.zeros(1024)
        [pos_mask], [neg_mask] = sampler([labels])
        self.assertEqual(0, pos_mask.sum())
        self.assertEqual(1, neg_mask.sum())
        self._check_sampled_labels(labels, pos_mask, neg_mask)

    def test_ratio_05_no_neg(self):
        sampler = BalancedSampler(1024, .5)
        labels = torch.ones(1024)
        [pos_mask], [neg_mask] = sampler([labels])
        self.assertEqual(512, pos_mask.sum())
        self.assertEqual(0, neg_mask.sum())
        self._check_sampled_labels(labels, pos_mask, neg_mask)

    def test_ratio_05_not_enough_pos(self):
        sampler = BalancedSampler(1024, .5)
        labels = torch.cat([torch.zeros(1024), torch.ones(256)])
        [pos_mask], [neg_mask] = sampler([labels])
        self.assertEqual(256, pos_mask.sum())
        self.assertEqual(256, neg_mask.sum())
        self._check_sampled_labels(labels, pos_mask, neg_mask)

    def test_ratio_05_not_enough_neg(self):
        sampler = BalancedSampler(1024, .5)
        labels = torch.cat([torch.zeros(256), torch.ones(1024)])
        [pos_mask], [neg_mask] = sampler([labels])
        self.assertEqual(512, pos_mask.sum())
        self.assertEqual(256, neg_mask.sum())
        self._check_sampled_labels(labels, pos_mask, neg_mask)

    def test_ratio_1_negative(self):
        """Non-zero result can be squeezed too much with only one result."""
        sampler = BalancedSampler(1024, 0.5)
        labels = torch.ones(1024)
        labels[0] = 0
        [pos_mask], [neg_mask] = sampler([labels])
        self.assertEqual(512, pos_mask.sum())
        self.assertEqual(1, neg_mask.sum())
        self._check_sampled_labels(labels, pos_mask, neg_mask)

    def test_ratio_1_positive(self):
        """Non-zero result can be squeezed too much with only one result."""
        sampler = BalancedSampler(1024, 0.5)
        labels = torch.zeros(1024)
        labels[0] = 1
        [pos_mask], [neg_mask] = sampler([labels])
        self.assertEqual(1, pos_mask.sum())
        self.assertEqual(1, neg_mask.sum())
        self._check_sampled_labels(labels, pos_mask, neg_mask)
