from abc import ABCMeta, abstractmethod
from dataclasses import replace
from typing import Optional, Callable, TYPE_CHECKING, Tuple, Type

import cv2
import numpy as np
from numba.typed import Dict
from PIL.Image import Image

from .base import Field, ARG_TYPE
from ..pipeline.operation import Operation
from ..pipeline.state import State
from ..pipeline.compiler import Compiler
from ..pipeline.allocation_query import AllocationQuery
from ..libffcv import imdecode, memcpy, resize_crop

if TYPE_CHECKING:
    from ..memory_managers.base import MemoryManager
    from ..reader import Reader

IMAGE_MODES = Dict()
IMAGE_MODES['jpg']= 0
IMAGE_MODES['raw']= 1

def encode_jpeg(numpy_image, quality):
    success, result = cv2.imencode('.jpg', numpy_image,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), quality])

    if not success:
        raise ValueError("Impossible to encode image in jpeg")

    return result.reshape(-1)

def resizer(image, target_resolution):
    if target_resolution is None:
        return image
    original_size = np.array([image.shape[1], image.shape[0]])
    ratio = target_resolution / original_size.max()
    if ratio < 1:
        new_size = (ratio * original_size).astype(int)
        image = cv2.resize(image, tuple(new_size))
    return image

def get_random_crop(height, width, scale, ratio):
    area = height * width
    log_ratio = np.log(ratio)
    for _ in range(10):
        target_area = area * np.random.uniform(scale[0], scale[1])
        aspect_ratio = np.exp(np.random.uniform(log_ratio[0], log_ratio[1]))
        w = int(round(np.sqrt(target_area * aspect_ratio)))
        h = int(round(np.sqrt(target_area / aspect_ratio)))
        if 0 < w <= width and 0 < h <= height:
            i = int(np.random.uniform(0, height - h + 1))
            j = int(np.random.uniform(0, width - w + 1))
            return i, j, h, w
    in_ratio = float(width) / float(height)
    if in_ratio < min(ratio):
        w = width
        h = int(round(w / min(ratio)))
    elif in_ratio > max(ratio):
        h = height
        w = int(round(h * max(ratio)))
    else:
        w = width
        h = height
    i = (height - h) // 2
    j = (width - w) // 2
    return i, j, h, w

def get_center_crop(height, width, _, ratio):
    s = min(height, width)
    c = int(ratio * s)
    delta_h = (height - c) // 2
    delta_w = (width - c) // 2

    return delta_h, delta_w, c, c


class SimpleRGBImageDecoder(Operation):
    def __init__(self):
        super().__init__()

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, AllocationQuery]:
        widths = self.metadata['width']
        heights = self.metadata['height']
        max_width = widths.max()
        max_height = heights.max()
        min_height = heights.min()
        min_width = widths.min()
        if min_width != max_width or max_height != min_height:
            msg = """SimpleRGBImageDecoder ony supports constant image,
consider RandomResizedCropRGBImageDecoder or CenterCropRGBImageDecoder
instead."""
            raise TypeError(msg)

        biggest_shape = (max_height, max_width, 3)
        my_dtype = np.dtype('<u1')

        return (
            replace(previous_state, jit_mode=True,
                    shape=biggest_shape, dtype=my_dtype),
            AllocationQuery(biggest_shape, my_dtype)
        )

    def generate_code(self) -> Callable:
        mem_read = self.memory_read
        imdecode_c = Compiler.compile(imdecode)

        jpg = IMAGE_MODES['jpg']
        raw = IMAGE_MODES['raw']
        my_range = Compiler.get_iterator()
        my_memcpy = Compiler.compile(memcpy)

        def decode(batch_indices, destination, metadata, storage_state):
            for dst_ix in my_range(len(batch_indices)):
                source_ix = batch_indices[dst_ix]
                field = metadata[source_ix]
                image_data = mem_read(field['data_ptr'], storage_state)
                height, width = field['height'], field['width']

                if field['mode'] == jpg:
                    imdecode_c(image_data, destination[dst_ix],
                               height, width, height, width, 0, 0, 1, 1, False, False)
                else:
                    my_memcpy(image_data, destination[dst_ix])

            return destination
        return decode

class ResizedCropRGBImageDecoder(SimpleRGBImageDecoder, metaclass=ABCMeta):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size
        
    

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, AllocationQuery]:
        widths = self.metadata['width']
        heights = self.metadata['height']
        self.max_width = widths.max()
        self.max_height = heights.max()
        output_shape = (self.output_size[0], self.output_size[1], 3)
        my_dtype = np.dtype('<u1')

        return (
            replace(previous_state, jit_mode=True,
                    shape=output_shape, dtype=my_dtype),
            AllocationQuery(output_shape, my_dtype)
        )

    def generate_code(self) -> Callable:

        jpg = IMAGE_MODES['jpg']

        mem_read = self.memory_read
        my_range = Compiler.get_iterator()
        imdecode_c = Compiler.compile(imdecode)
        resize_crop_c = Compiler.compile(resize_crop)
        get_crop_c = Compiler.compile(self.get_crop_generator)

        temp_buffer_shape = (self.max_height, self.max_width, 3)

        scale = self.scale
        ratio = self.ratio
        if isinstance(scale, tuple):
            scale = np.array(scale)
        if isinstance(ratio, tuple):
            ratio = np.array(ratio)

        def decode(batch_indices, destination, metadata, storage_state):
            for dst_ix in my_range(len(batch_indices)):
                source_ix = batch_indices[dst_ix]
                field = metadata[source_ix]
                image_data = mem_read(field['data_ptr'], storage_state)
                height = np.uint32(field['height'])
                width = np.uint32(field['width'])

                if field['mode'] == jpg:
                    temp_buffer = np.zeros(temp_buffer_shape, dtype=('<u1'))
                    imdecode_c(image_data, temp_buffer,
                               height, width, height, width, 0, 0, 1, 1, False, False)
                    selected_size = 3 * height * width
                    temp_buffer = temp_buffer.reshape(-1)[:selected_size]
                    temp_buffer = temp_buffer.reshape(height, width, 3)
                    
                else:
                    temp_buffer = image_data.reshape(height, width, 3)
                    

                i, j, h, w = get_crop_c(height, width, scale, ratio)


                resize_crop_c(temp_buffer, i, i + h, j, j + w, destination[dst_ix] )
                    
                    
            return destination
        return decode
        
    @property
    @abstractmethod
    def get_crop_generator():
        raise NotImplemented()


class RandomResizedCropRGBImageDecoder(ResizedCropRGBImageDecoder):
    def __init__(self, output_size, scale=(0.08, 1.0), ratio=(0.75, 4/3)):
        super().__init__(output_size)
        self.scale = scale
        self.ratio = ratio
        self.output_size = output_size

    @property
    def get_crop_generator(self):
        return get_random_crop

class CenterCropRGBImageDecoder(ResizedCropRGBImageDecoder):
    def __init__(self, output_size, ratio=(243/256)):
        super().__init__(output_size)
        self.scale = None
        self.ratio = ratio

    @property
    def get_crop_generator(self):
        return get_center_crop


class RGBImageField(Field):

    def __init__(self, write_mode='raw', smart_factor:float = None,
                 max_resolution: int=None, smart_threshold: int=None,
                 jpeg_quality: int = 90) -> None:
        self.write_mode = write_mode
        self.smart_factor = smart_factor
        self.smart_threshold = smart_threshold
        self.max_resolution = max_resolution
        self.jpeg_quality = jpeg_quality

    @property
    def metadata_type(self) -> np.dtype:
        return np.dtype([
            ('mode', '<u1'),
            ('width', '<u2'),
            ('height', '<u2'),
            ('data_ptr', '<u8'),
        ])
        
    def get_decoder_class(self) -> Type[Operation]:
        return SimpleRGBImageDecoder

    @staticmethod
    def from_binary(binary: ARG_TYPE) -> Field:
        return RGBImageField()

    def to_binary(self) -> ARG_TYPE:
        return np.zeros(1, dtype=ARG_TYPE)[0]

    def encode(self, destination, image, malloc):
        if isinstance(image, Image):
            image = np.array(image)

        if not isinstance(image, np.ndarray):
            raise TypeError(f"Unsupported image type {type(image)}")

        if image.dtype != np.uint8:
            raise ValueError("Image type has to be uint8")

        if image.shape[2] != 3:
            raise ValueError(f"Invalid shape for rgb image: {image.shape}")

        assert image.dtype == np.uint8

        image = resizer(image, self.max_resolution)

        write_mode = self.write_mode
        as_jpg = None

        if write_mode == 'smart':
            as_jpg = encode_jpeg(image, self.jpeg_quality)
            write_mode = 'raw'
            if self.smart_factor is not None:
                if as_jpg.nbytes * self.smart_factor <= image.nbytes:
                    write_mode = 'jpg'
            if self.smart_threshold is not None:
                if image.nbytes > self.smart_threshold:
                    write_mode = 'jpg'

        destination['mode'] = IMAGE_MODES[write_mode]
        destination['height'], destination['width'] = image.shape[:2]

        if write_mode == 'jpg':
            if as_jpg is None:
                as_jpg = encode_jpeg(image, self.jpeg_quality)
            destination['data_ptr'], storage = malloc(as_jpg.nbytes)
            storage[:] = as_jpg
        elif write_mode == 'raw':
            image_bytes = np.ascontiguousarray(image).view('<u1').reshape(-1)
            destination['data_ptr'], storage = malloc(image.nbytes)
            storage[:] = image_bytes
        else:
            raise ValueError(f"Unsupported write mode {self.write_mode}")
