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
from ..libffcv import *

if TYPE_CHECKING:
    from ..memory_managers.base import MemoryManager
    from ..reader import Reader

IMAGE_MODES = Dict()
IMAGE_MODES['jpg'] = 0
IMAGE_MODES['raw'] = 1
IMAGE_MODES['png'] = 2

from turbojpeg import TurboJPEG, TJCS_RGB, TJSAMP_444
turbo_jpeg = TurboJPEG()
def encode_jpeg(numpy_image, quality,jpeg_subsample=TJSAMP_444):
    result = turbo_jpeg.encode(numpy_image, quality=quality, pixel_format=TJCS_RGB,jpeg_subsample=jpeg_subsample)
    result = np.frombuffer(result, np.uint8)
    return result.reshape(-1)

def encode_png(numpy_image):
    # x=cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    result = cv2.imencode('.png', numpy_image)[1]
    result = np.frombuffer(result, np.uint8)
    return result.reshape(-1)

def resizer(image, target_resolution):
    if target_resolution is None:
        return image
    original_size = np.array([image.shape[1], image.shape[0]])
    ratio = target_resolution / original_size.max()
    if ratio < 1:
        new_size = (ratio * original_size).astype(int)
        image = cv2.resize(image, tuple(new_size), interpolation=cv2.INTER_CUBIC)
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
    """Most basic decoder for the :class:`~ffcv.fields.RGBImageField`.

    It only supports dataset with constant image resolution and will simply read (potentially decompress) and pass the images as is.
    """
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
            msg = """SimpleRGBImageDecoder only supports constant image,
consider RandomResizedCropRGBImageDecoder or CenterCropRGBImageDecoder
instead."""
            raise TypeError(msg)
        
        max_shape = ((np.uint64(widths)*np.uint64(heights)*3).max(),)
        my_dtype = np.dtype('<u1')

        return (
            replace(previous_state, jit_mode=True,
                    shape=max_shape, dtype=my_dtype),
            AllocationQuery(max_shape, my_dtype)
        )

    def generate_code(self) -> Callable:
        mem_read = self.memory_read
        imdecode_c = Compiler.compile(imdecode)
        cv_imdecode_c = Compiler.compile(cv_imdecode)

        jpg = IMAGE_MODES['jpg']
        raw = IMAGE_MODES['raw']
        png = IMAGE_MODES['png']
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
                elif field['mode'] == raw:
                    my_memcpy(image_data, destination[dst_ix])
                elif field['mode'] == png:
                    cv_imdecode_c(image_data, destination[dst_ix])
                else:
                    pass

            return destination[:len(batch_indices)]

        decode.is_parallel = True
        return decode


class ResizedCropRGBImageDecoder(SimpleRGBImageDecoder, metaclass=ABCMeta):
    """Abstract decoder for :class:`~ffcv.fields.RGBImageField` that performs a crop and and a resize operation.

    It supports both variable and constant resolution datasets.
    """
    def __init__(self, output_size,interpolation):
        super().__init__()
        self.output_size = output_size
        self.interpolation = interpolation
        self.use_crop_decode = True
    
    def use_crop_decode_(self, value):
        self.use_crop_decode = value

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, AllocationQuery]:
        widths = self.metadata['width']
        heights = self.metadata['height']
        # We convert to uint64 to avoid overflows
        self.max_width = np.uint64(widths.max())
        self.max_height = np.uint64(heights.max())
        output_shape = (self.output_size[0], self.output_size[1], 3)
        my_dtype = np.dtype('<u1')
        if self.use_crop_decode:
            max_shape = ((np.uint64(widths)*np.uint64(heights)*3).max(),)
        else:
            max_shape = (1,) # we don't need the buffer memory
        return (
            replace(previous_state, jit_mode=True,
                    shape=output_shape, dtype=my_dtype),
            (AllocationQuery(output_shape, my_dtype),
            AllocationQuery(max_shape, my_dtype),
            )
        )

    def generate_code(self) -> Callable:

        jpg = IMAGE_MODES['jpg']
        raw = IMAGE_MODES['raw']
        png = IMAGE_MODES['png']

        mem_read = self.memory_read
        my_range = Compiler.get_iterator()
        imdecode_c = Compiler.compile(imdecode)
        cv_imdecode_c = Compiler.compile(cv_imdecode)
        resize_crop_c = Compiler.compile(resize_crop)
        imcropresizedecode_c = Compiler.compile(imcropresizedecode)
        get_crop_c = Compiler.compile(self.get_crop_generator)

        scale = self.scale
        ratio = self.ratio
        use_crop_decode = self.use_crop_decode
        interpolation = self.interpolation
        if isinstance(scale, tuple):
            scale = np.array(scale)
        if isinstance(ratio, tuple):
            ratio = np.array(ratio)

        def decode(batch_indices, my_storage, metadata, storage_state):
            destination, temp_storage = my_storage
            for dst_ix in my_range(len(batch_indices)):
                source_ix = batch_indices[dst_ix]
                field = metadata[source_ix]
                image_data = mem_read(field['data_ptr'], storage_state)
                height = np.uint32(field['height'])
                width = np.uint32(field['width'])

                i, j, h, w = get_crop_c(height, width, scale, ratio)
                
                if field['mode'] == jpg:
                    temp_buffer = temp_storage[dst_ix]
                    if use_crop_decode:
                        imcropresizedecode_c(image_data,  destination[dst_ix],                               
                                    h,w, 
                                    i, j, interpolation)
                    else:
                        ## decode the whole image
                        imdecode_c(image_data, temp_buffer,
                                    height, width, height, width, 0, 0, 1, 1, False, False)
                        ## crop and resize the image
                        selected_size = 3 * height * width
                        temp_buffer = temp_buffer.reshape(-1)[:selected_size]
                        temp_buffer = temp_buffer.reshape(height, width, 3)
                        resize_crop_c(temp_buffer, i, i + h, j, j + w,
                              destination[dst_ix])
                elif field['mode'] == raw:
                    temp_buffer = image_data.reshape(height, width, 3)
                    resize_crop_c(temp_buffer, i, i + h, j, j + w,
                                destination[dst_ix])
                elif field['mode'] == png:
                    temp_buffer = temp_storage[dst_ix]
                    cv_imdecode_c(image_data, temp_buffer)
                    buffer = temp_buffer[:height*width*3].reshape(height,width,3)                    
                    resize_crop_c(buffer, i, i + h, j, j + w,
                                destination[dst_ix])
                else:
                    pass
                
            return destination[:len(batch_indices)]
        decode.is_parallel = True
        return decode

    @property
    @abstractmethod
    def get_crop_generator():
        raise NotImplementedError


class RandomResizedCropRGBImageDecoder(ResizedCropRGBImageDecoder):
    """Decoder for :class:`~ffcv.fields.RGBImageField` that performs a Random crop and and a resize operation.

    It supports both variable and constant resolution datasets.

    Parameters
    ----------
    output_size : Tuple[int]
        The desired resized resolution of the images
    scale : Tuple[float]
        The range of possible ratios (in area) than can randomly sampled
    ratio : Tuple[float]
        The range of potential aspect ratios that can be randomly sampled
    """
    def __init__(self, output_size, scale=(0.08, 1.0), ratio=(0.75, 4/3), interpolation=cv2.INTER_CUBIC):
        super().__init__(output_size, interpolation=interpolation)
        self.scale = scale
        self.ratio = ratio
        self.output_size = output_size

    @property
    def get_crop_generator(self):
        return get_random_crop


class CenterCropRGBImageDecoder(ResizedCropRGBImageDecoder):
    """Decoder for :class:`~ffcv.fields.RGBImageField` that performs a center crop followed by a resize operation.

    It supports both variable and constant resolution datasets.

    Parameters
    ----------
    output_size : Tuple[int]
        The desired resized resolution of the images
    ratio: float
        ratio of (crop size) / (min side length)
    """
    # output size: resize crop size -> output size
    def __init__(self, output_size, ratio, interpolation=cv2.INTER_AREA):
        super().__init__(output_size,interpolation=interpolation)
        self.scale = None
        self.ratio = ratio

    @property
    def get_crop_generator(self):
        return get_center_crop


class RGBImageField(Field):
    """
    A subclass of :class:`~ffcv.fields.Field` supporting RGB image data.

    Parameters
    ----------
    write_mode : str, optional
        How to write the image data to the dataset file. Should be either 'raw'
        (``uint8`` pixel values), 'jpg' (compress to JPEG format), 'smart'
        (decide between saving pixel values and JPEG compressing based on image
        size), and 'proportion' (JPEG compress a random subset of the data with
        size specified by the ``compress_probability`` argument). By default: 'raw'.
    max_resolution : int, optional
        If specified, will resize images to have maximum side length equal to
        this value before saving, by default None
    smart_threshold : int, optional
        When `write_mode='smart`, will compress an image if it would take more than `smart_threshold` times to use RAW instead of jpeg.
    jpeg_quality : int, optional
        The quality parameter for JPEG encoding (ignored for
        ``write_mode='raw'``), by default 90
    compress_probability : float, optional
        Ignored unless ``write_mode='proportion'``; in the latter case it is the
        probability with which image is JPEG-compressed, by default 0.5.
    """
    def __init__(self, write_mode='raw', max_resolution: int = None,
                 smart_threshold: int = None, jpeg_quality: int = 90,
                 compress_probability: float = 0.5) -> None:
        self.write_mode = write_mode
        self.smart_threshold = smart_threshold
        self.max_resolution = max_resolution
        self.jpeg_quality = int(jpeg_quality)
        self.proportion = compress_probability

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
    def from_binary(binary: ARG_TYPE) -> Field: # type: ignore
        return RGBImageField()

    def to_binary(self) -> ARG_TYPE: # type: ignore
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
        ccode = None # compressed code

        if write_mode == 'smart':
            write_mode = 'raw'
            if self.smart_threshold is not None:
                if image.nbytes > self.smart_threshold:
                    write_mode = 'jpg'
        elif write_mode == 'proportion':
            if np.random.rand() < self.proportion:
                write_mode = 'jpg'
            else:
                write_mode = 'raw'

        destination['mode'] = IMAGE_MODES[write_mode]
        destination['height'], destination['width'] = image.shape[:2]

        if write_mode == 'jpg':
            ccode = encode_jpeg(image, self.jpeg_quality)
            destination['data_ptr'], storage = malloc(ccode.nbytes)
            storage[:] = ccode
        elif write_mode == 'raw':
            image_bytes = np.ascontiguousarray(image).view('<u1').reshape(-1)
            destination['data_ptr'], storage = malloc(image.nbytes)
            storage[:] = image_bytes
        elif write_mode == 'png':
            ccode = encode_png(image)
            destination['data_ptr'], storage = malloc(ccode.nbytes)
            storage[:] = ccode
        else:
            raise ValueError(f"Unsupported write mode {self.write_mode}")
