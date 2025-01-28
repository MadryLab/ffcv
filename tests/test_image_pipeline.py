import numpy as np
import torch as ch
from torch.utils.data import Dataset
from assertpy import assert_that
from tempfile import NamedTemporaryFile
from torchvision.datasets import CIFAR10
from torch.utils.data import Subset

from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, RGBImageField
from ffcv.loader import Loader
from ffcv.pipeline.compiler import Compiler

class DummyDataset(Dataset):

    def __init__(self, length, height, width, reversed=False):
        self.length = length
        self.height = height
        self.width = width
        self.reversed = reversed
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if index > self.length:
            raise IndexError
        dims = (self.height, self.width, 3)
        image_data = ((np.ones(dims) * index) % 255).astype('uint8')
        result = index,image_data
        if self.reversed:
            result = tuple(reversed(result))
        return result

def create_and_validate(length, mode='raw', reversed=False):

    dataset = DummyDataset(length, 500, 300, reversed=reversed)

    with NamedTemporaryFile() as handle:
        name = handle.name
        
        fields = {
            'index': IntField(),
            'value': RGBImageField(write_mode=mode, jpeg_quality=95)
        }
        
        if reversed:
            fields = {
                'value': RGBImageField(write_mode=mode, jpeg_quality=95),
                'index': IntField()
            }

        writer = DatasetWriter(name, fields, num_workers=2)

        writer.from_indexed_dataset(dataset, chunksize=5)
            
        Compiler.set_enabled(False)
        
        loader = Loader(name, batch_size=5, num_workers=2)
        
        for res in loader:
            if not reversed:
                index, images  = res
            else:
                images , index = res

            for i, image in zip(index, images):
                if mode == 'raw':
                    assert_that(ch.all((image == (i % 255)).reshape(-1))).is_true()
                else:
                    assert_that(ch.all((image == (i % 255)).reshape(-1))).is_true()
                
def make_and_read_cifar_subset(length):
    my_dataset = Subset(CIFAR10(root='/tmp', train=True, download=True), range(length))

    with NamedTemporaryFile() as handle:
        name = handle.name
        writer = DatasetWriter(name, {
            'image': RGBImageField(write_mode='smart', 
                                max_resolution=32),
            'label': IntField(),
        }, num_workers=2)

        writer.from_indexed_dataset(my_dataset, chunksize=10)

        Compiler.set_enabled(False)
        
        loader = Loader(name, batch_size=5, num_workers=2)
        
        for index, images in loader:
            pass

def test_cifar_subset():
    make_and_read_cifar_subset(200)

def test_simple_raw_image_pipeline():
    create_and_validate(500, 'raw', False)

def test_simple_raw_image_pipeline_rev():
    create_and_validate(500, 'raw', True)

def test_simple_jpg_image_pipeline():
    create_and_validate(500, 'jpg', False)

def test_simple_jpg_image_pipeline_rev():
    create_and_validate(500, 'jpg', True)

def test_simple_png_image_pipeline():
    create_and_validate(500, 'png', False)
