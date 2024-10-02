"""
# copy from https://github.com/facebookresearch/FFCV-SSL/blob/main/ffcv/transforms/grayscale.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""


from typing import Callable, Optional, Tuple
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.state import State
from ffcv.pipeline.compiler import Compiler
from dataclasses import replace
import numpy as np
import random


class RandomGrayscale(Operation):
    """Add Gaussian Blur with probability blur_prob.
    Operates on raw arrays (not tensors).

    Parameters
    ----------
    blur_prob : float
        The probability with which to flip each image in the batch
        horizontally.
    """

    def __init__(self, p: float = 0.2, seed: int = None):
        super().__init__()
        self.gray_prob = p
        self.seed = seed

    def generate_code(self) -> Callable:
        my_range = Compiler.get_iterator()
        gray_prob = self.gray_prob
        seed = self.seed

        if seed is None:

            def grayscale(images, _):
                for i in my_range(images.shape[0]):
                    if np.random.rand() > gray_prob:
                        continue
                    images[i] = (
                        0.2989 * images[i, ..., 0:1]
                        + 0.5870 * images[i, ..., 1:2]
                        + 0.1140 * images[i, ..., 2:3]
                    )
                return images

            grayscale.is_parallel = True
            return grayscale

        def grayscale(images, _, counter):
            random.seed(seed + counter)
            values = np.zeros(images.shape[0])
            for i in range(images.shape[0]):
                values[i] = random.uniform(0, 1)
            for i in my_range(images.shape[0]):
                if values[i] > gray_prob:
                    continue
                images[i] = (
                    0.2989 * images[i, ..., 0:1]
                    + 0.5870 * images[i, ..., 1:2]
                    + 0.1140 * images[i, ..., 2:3]
                )
            return images

        grayscale.with_counter = True
        grayscale.is_parallel = True
        return grayscale

    def declare_state_and_memory(
        self, previous_state: State
    ) -> Tuple[State, Optional[AllocationQuery]]:
        assert previous_state.jit_mode
        return (previous_state, None)


class LabelGrayscale(Operation):
    """ColorJitter info added to the labels. Should be initialized in exactly the same way as
    :cla:`ffcv.transforms.ColorJitter`.
    """

    def __init__(self, gray_prob: float = 0.2, seed: int = None):
        super().__init__()
        self.gray_prob = gray_prob
        self.seed = np.random.RandomState(seed).randint(0, 2**32 - 1)

    def generate_code(self) -> Callable:
        my_range = Compiler.get_iterator()
        gray_prob = self.gray_prob
        seed = self.seed

        def grayscale(labels, temp_array, indices):
            rep = ""
            for i in indices:
                rep += str(i)
            local_seed = (hash(rep) + seed) % 2**31
            temp_array[:, :-1] = labels
            for i in my_range(temp_array.shape[0]):
                np.random.seed(local_seed + i)
                if np.random.rand() < gray_prob:
                    temp_array[i, -1] = 0.0
                else:
                    temp_array[i, -1] = 1.0
            return temp_array

        grayscale.is_parallel = True
        grayscale.with_indices = True

        return grayscale

    def declare_state_and_memory(
        self, previous_state: State
    ) -> Tuple[State, Optional[AllocationQuery]]:
        previous_shape = previous_state.shape
        new_shape = (previous_shape[0] + 1,)
        return (
            replace(previous_state, shape=new_shape, dtype=np.float32),
            AllocationQuery(new_shape, dtype=np.float32),
        )