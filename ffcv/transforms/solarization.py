# copy from https://github.com/facebookresearch/FFCV-SSL/blob/main/ffcv/transforms/solarization.py
"""
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
import numpy as np
from dataclasses import replace
import random


class RandomSolarization(Operation):
    """Solarize the image randomly with a given probability by inverting all pixel
    values above a threshold. If img is a Tensor, it is expected to be in [..., 1 or 3, H, W] format,
    where ... means it can have an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".
    Parameters
    ----------
        solarization_prob (float): probability of the image being solarized. Default value is 0.5
        threshold (float): all pixels equal or above this value are inverted.
    """

    def __init__(
        self,  threshold: float = 128, p: float = 0.5, seed: int = None
    ):
        super().__init__()
        self.sol_prob = p
        self.threshold = threshold
        self.seed = seed

    def generate_code(self) -> Callable:
        my_range = Compiler.get_iterator()
        sol_prob = self.sol_prob
        threshold = self.threshold
        seed = self.seed

        if seed is None:

            def solarize(images, _):
                for i in my_range(images.shape[0]):
                    if np.random.rand() < sol_prob:
                        mask = images[i] >= threshold
                        images[i] = np.where(mask, 255 - images[i], images[i])
                return images

            solarize.is_parallel = True
            return solarize

        def solarize(images, _, counter):
            random.seed(seed + counter)
            values = np.zeros(len(images))
            for i in range(len(images)):
                values[i] = random.uniform(0, 1)
            for i in my_range(images.shape[0]):
                if values[i] < sol_prob:
                    mask = images[i] >= threshold
                    images[i] = np.where(mask, 255 - images[i], images[i])
            return images

        solarize.with_counter = True
        solarize.is_parallel = True
        return solarize

    def declare_state_and_memory(
        self, previous_state: State
    ) -> Tuple[State, Optional[AllocationQuery]]:
        return (replace(previous_state, jit_mode=True), None)


class LabelSolarization(Operation):
    """ColorJitter info added to the labels. Should be initialized in exactly the same way as
    :cla:`ffcv.transforms.ColorJitter`.
    """

    def __init__(
        self, solarization_prob: float = 0.5, threshold: float = 128, seed: int = None
    ):
        super().__init__()
        self.solarization_prob = solarization_prob
        self.threshold = threshold
        self.seed = seed

    def generate_code(self) -> Callable:
        my_range = Compiler.get_iterator()
        solarization_prob = self.solarization_prob
        seed = self.seed

        def solarize(labels, temp_array, indices):
            temp_array[:, :-1] = labels
            random.seed(seed + indices)
            for i in my_range(labels.shape[0]):
                if random.uniform(0, 1) < solarization_prob:
                    temp_array[i, -1] = 1
                else:
                    temp_array[i, -1] = 0
            return temp_array

        solarize.is_parallel = True
        solarize.with_indices = True

        return solarize

    def declare_state_and_memory(
        self, previous_state: State
    ) -> Tuple[State, Optional[AllocationQuery]]:
        previous_shape = previous_state.shape
        new_shape = (previous_shape[0] + 1,)
        return (
            replace(previous_state, shape=new_shape, dtype=np.float32),
            AllocationQuery(new_shape, dtype=np.float32),
        )