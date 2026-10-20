# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import unittest
from typing import Callable

import numpy as np
import torch

import scene_graph_prediction.modeling.backbone.fbnet.fbnet_builder as fbnet_builder


class TestFBNetBuilder(unittest.TestCase):
    def _test_primitive(
            self,
            device: torch.device,
            op_name: str,
            op_func: Callable,
            n: int,
            c_in: int,
            c_out: int,
            expand_ratio: float,
            stride: int
    ):
        op = op_func(c_in, c_out, expand_ratio, stride).to(device)
        input_tensor = torch.rand([n, c_in, 7, 7], dtype=torch.float32).to(device)
        output_tensor = op(input_tensor)
        self.assertEqual(output_tensor.shape[:2], torch.Size([n, c_out]),
                         f'Primitive {op_name} failed for shape {input_tensor.shape}.')

    def test_identity(self):
        id_op = fbnet_builder._IdentityOrConv(20, 20, 1)
        input_tensor = torch.rand([10, 20, 7, 7], dtype=torch.float32)
        output_tensor = id_op(input_tensor)
        np.testing.assert_array_equal(np.array(input_tensor), np.array(output_tensor))

        id_op = fbnet_builder._IdentityOrConv(20, 40, 2)
        input_tensor = torch.rand([10, 20, 7, 7], dtype=torch.float32)
        output_tensor = id_op(input_tensor)
        np.testing.assert_array_equal(output_tensor.shape, [10, 40, 4, 4])

    def test_primitives(self):
        """ Make sures the primitives runs """
        for op_name, op_func in fbnet_builder._PRIMITIVES.items():
            # noinspection PyTypeChecker
            self._test_primitive("cpu", op_name, op_func, n=20, c_in=16, c_out=32, expand_ratio=4, stride=1)

    @unittest.skipIf(not torch.cuda.is_available(), "no CUDA detected")
    def test_primitives_cuda(self):
        """ Make sures the primitives runs on cuda """
        for op_name, op_func in fbnet_builder._PRIMITIVES.items():
            # noinspection PyTypeChecker
            self._test_primitive("cuda", op_name, op_func, n=20, c_in=16, c_out=32, expand_ratio=4, stride=1)

    def test_primitives_empty_batch(self):
        """ Make sures the primitives runs """
        for op_name, op_func in fbnet_builder._PRIMITIVES.items():
            # test empty batch size
            # noinspection PyTypeChecker
            self._test_primitive("cpu", op_name, op_func, n=0, c_in=16, c_out=32, expand_ratio=4, stride=1)

    @unittest.skipIf(not torch.cuda.is_available(), "no CUDA detected")
    def test_primitives_cuda_empty_batch(self):
        """ Make sures the primitives runs """
        for op_name, op_func in fbnet_builder._PRIMITIVES.items():
            # test empty batch size
            # noinspection PyTypeChecker
            self._test_primitive("cuda", op_name, op_func, n=0, c_in=16, c_out=32, expand_ratio=4, stride=1)
