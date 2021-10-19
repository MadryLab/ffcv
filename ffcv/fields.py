from __future__ import annotations
from io import UnsupportedOperation
import numpy as np
from collections import namedtuple
from abc import ABC, abstractmethod

ARG_TYPE = np.dtype('<u1', 1024)

class Field(ABC):

    @property
    @abstractmethod
    def metadata_type(self) -> np.dtype:
        raise NotImplemented

    @staticmethod
    @abstractmethod
    def from_binary(binary: ARG_TYPE) -> Field:
        raise NotImplemented()

    @abstractmethod
    def to_binary(self) -> ARG_TYPE:
        raise NotImplemented()

    @abstractmethod
    def encode(field, metadata_destination, malloc):
        raise NotImplemented()


class FloatField(Field):
    def __init__(self):
        pass

    @property
    def metadata_type(self) -> np.dtype:
        return np.dtype('<f8')

    @staticmethod
    def from_binary(binary: ARG_TYPE) -> Field:
        return FloatField()

    def to_binary(self) -> ARG_TYPE:
        return np.zeros(1, dtype=ARG_TYPE)[0]

    def encode(self, destination, field, malloc):
        destination[0] = field

class IntField(Field):
    @property
    def metadata_type(self) -> np.dtype:
        return np.dtype('<i8')

    @staticmethod
    def from_binary(binary: ARG_TYPE) -> Field:
        return IntField()

    def to_binary(self) -> ARG_TYPE:
        return np.zeros(1, dtype=ARG_TYPE)[0]

    def encode(self, destination, field, malloc):
        # We just allocate 1024bytes for fun
        destination[0] = field

from PIL.Image import Image
import io

IMAGE_MODES = {
    'jpg': 0,
    'raw': 1
}

class RGBImageField(Field):

    def __init__(self, write_mode='raw') -> None:
        self.write_mode = write_mode

    @property
    def metadata_type(self) -> np.dtype:
        return np.dtype([
            ('mode', '<u1'),
            ('width', '<u2'),
            ('height', '<u2'),
            ('data_ptr', '<u8'),
        ])

    @staticmethod
    def from_binary(binary: ARG_TYPE) -> Field:
        return RGBImageField()

    def to_binary(self) -> ARG_TYPE:
        return np.zeros(1, dtype=ARG_TYPE)[0]

    def encode(self, destination, image, malloc):
        if isinstance(image, Image):
            destination['height'], destination['width'] = image.size
            destination['mode'] = IMAGE_MODES[self.write_mode]

            if self.write_mode == 'jpg':
                with io.BytesIO() as output:
                    image.save(output, format="GIF")
                    contents = np.frombuffer(output.getvalue(), dtype='<u1')
                    destination['data_ptr'], storage = malloc(len(contents))
                    storage[:] = contents
            elif self.write_mode == 'raw':
                image_np = np.array(image).transpose(2, 0, 1)
                assert image_np.dtype == np.uint8
                image_np = np.ascontiguousarray(image_np).view('<u1').reshape(-1)
                destination['data_ptr'], storage = malloc(len(image_np))
                storage[:] = image_np
            else:
                raise ValueError(f"Unsupported write mode {self.write_mode}")

        else:
            raise TypeError(f"Unsupported image type {type(image)}")

