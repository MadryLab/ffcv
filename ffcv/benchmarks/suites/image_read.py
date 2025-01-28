import logging
import os
from tempfile import NamedTemporaryFile
from time import sleep, time

import numpy as np
from assertpy import assert_that
from ffcv.fields import BytesField, IntField, RGBImageField
from ffcv.memory_managers import OSCacheManager
from ffcv.pipeline.compiler import Compiler
from ffcv.reader import Reader
from ffcv.writer import DatasetWriter
from torch.utils.data import Dataset
from tqdm import tqdm

from ..benchmark import Benchmark
from ..decorator import benchmark



class DummyDataset(Dataset):

    def __init__(self, length, size):
        self.length = length
        self.size = size
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if index > self.length:
            raise IndexError

        dims = tuple([*self.size, 3])
        image_data = np.random.randint(low=0, high=255, size=dims, dtype='uint8')
        return index, image_data



@benchmark({
    'n': [3000],
    'length': [3000],
    'mode': [
        'raw',
        'jpg',
        'png',
        ],
    'num_workers': [
        1,
        8,
        16,
        32,
    ],
    'batch_size': [
        500
    ],
    'size': [
        (32, 32),  # CIFAR
        (224,224),
        (300, 500),  # ImageNet
    ],
    'compile': [
        True,
        # False
    ],
    'random_reads': [
        True,
        # False
    ]
})
class ImageReadBench(Benchmark):
    
    def __init__(self, n, length, mode, size, random_reads, compile, num_workers, batch_size):
        self.n = n
        self.mode = mode
        self.length = length
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.size = size
        self.compile = compile
        self.random_reads = random_reads
        self.dataset = DummyDataset(length, size)
        
    def __enter__(self):
        self.handle = NamedTemporaryFile()
        self.handle.__enter__()
        name = self.handle.name

        writer = DatasetWriter(name, {
            'index': IntField(),
            'value': RGBImageField(write_mode=self.mode)
        })

        writer.from_indexed_dataset(self.dataset, chunksize=100)

        reader = Reader(name)
        manager = OSCacheManager(reader)

        Compiler.set_enabled(self.compile)
        Compiler.set_num_threads(self.num_workers)

        memreader = manager.compile_reader()
        Decoder = RGBImageField().get_decoder_class()
        decoder = Decoder()
        decoder.accept_globals(reader.metadata['f1'], memreader)

        context = manager.schedule_epoch(np.arange(self.n))
        context.__enter__()
        self.context = context

        decode = decoder.generate_code()
        decode = Compiler.compile(decode)
        
        self.buff = np.zeros((self.batch_size, *self.size, 3), dtype='uint8')
        
        if self.random_reads:
            self.indices = np.random.choice(self.n, size=self.n, replace=False)
        else:
            self.indices = np.arange(self.n)
            
        def code(indices, buff, state):
            result = 0
            for i in range(0, len(indices), self.batch_size):
                result += decode(indices[i:i + self.batch_size], buff, reader.metadata['f1'], state)[0, 5, 5]
            return result
                
        self.code = code

    def run(self):
        self.code(self.indices, self.buff, self.context.state)

    def __exit__(self, *args):
        self.handle.__exit__(*args)
        self.context.__exit__(*args)
        pass
