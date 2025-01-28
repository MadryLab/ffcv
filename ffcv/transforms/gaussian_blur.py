# copy from https://github.com/facebookresearch/FFCV-SSL/blob/main/ffcv/transforms/gaussian_blur.py
import numpy as np
from typing import Callable, Optional, Tuple
from dataclasses import replace
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.state import State
from ffcv.pipeline.compiler import Compiler
from scipy.signal import convolve2d


def apply_blur(img, kernel_size, w):
    pad = (kernel_size - 1) // 2
    H, W, _ = img.shape
    tmp = np.zeros(img.shape, dtype=np.float32)
    for k in range(kernel_size):
        start = max(0, pad - k)
        stop = min(W, pad - k + W)
        window = (img[:, start:stop] / 255) * w[k]
        tmp[:, np.abs(stop - W) : W - start] += window
    tmp2 = tmp + 0.0
    for k in range(kernel_size):
        start = max(0, pad - k)
        stop = min(H, pad - k + H)
        window = (tmp[start:stop] * w[k]).astype(np.uint8)
        tmp2[np.abs(stop - H) : H - start] += window
    return np.clip(tmp2 * 255.0, 0, 255).astype(np.uint8)


class GaussianBlur(Operation):
    """Blurs image with randomly chosen Gaussian blur.
    If the image is torch Tensor, it is expected
    to have [..., C, H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        blur_prob (float): probability to apply blurring to each input
        kernel_size (int or sequence): Size of the Gaussian kernel.
        sigma (float or tuple of float (min, max)): Standard deviation to be used for
            creating kernel to perform blurring. If float, sigma is fixed. If it is tuple
            of float (min, max), sigma is chosen uniformly at random to lie in the
            given range.
    """

    def __init__(self, kernel_size=5, sigma=(0.1, 2.0), p=0.5):
        super().__init__()
        self.blur_prob = p
        self.kernel_size = kernel_size
        assert sigma[1] > sigma[0]
        self.sigmas = np.linspace(sigma[0], sigma[1], 10)
        from scipy import signal

        self.weights = np.stack(
            [
                signal.gaussian(kernel_size, s)
                for s in np.linspace(sigma[0], sigma[1], 10)
            ]
        )
        self.weights /= self.weights.sum(1, keepdims=True)

    def generate_code(self) -> Callable:
        my_range = Compiler.get_iterator()
        blur_prob = self.blur_prob
        kernel_size = self.kernel_size
        weights = self.weights
        apply_blur_c = Compiler.compile(apply_blur)

        def blur(images, indices):

            for i in my_range(images.shape[0]):
                if np.random.rand() < blur_prob:
                    k = np.random.randint(low=0, high=10)
                    for ch in range(images.shape[-1]):
                        images[i, ..., ch] = convolve2d(
                            images[i, ..., ch],
                            np.outer(weights[k], weights[k]),
                            mode="same",
                        )
                        # images[i] = apply_blur_c(images[i], kernel_size, weights[k])
            return images

        blur.is_parallel = True
        blur.with_indices = True
        return blur

    def declare_state_and_memory(
        self, previous_state: State
    ) -> Tuple[State, Optional[AllocationQuery]]:
        return (
            replace(previous_state, jit_mode=False),
            None,
        )
