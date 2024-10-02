import os
from tempfile import NamedTemporaryFile
from time import sleep, time

import numpy as np
from tqdm import tqdm
from assertpy import assert_that
from torch.utils.data import Dataset

from ffcv.writer import DatasetWriter
from ffcv.reader import Reader
from ffcv.fields import BytesField, IntField
from ffcv.pipeline.compiler import Compiler
from ffcv.memory_managers import OSCacheManager
from ffcv.libffcv import memcpy

from ..decorator import benchmark
from ..benchmark import Benchmark

class DummyDataset(Dataset):

    def __init__(self, l, size):
        self.l = l
        self.size = size

    def __len__(self):
        return self.l

    def __getitem__(self, index):
        if index > self.l:
            raise IndexError
        np.random.seed(index)
        return index, np.random.randint(0, 255, size=self.size, dtype='u1')

@benchmark({
    'num_samples': [3000],
    'size_bytes': [
        32 * 32 * 3, # CIFAR RAW image size,
        500 * 300 * 3, # IMAGENET raw image size,
        128 * 1024, # IMAGENET jpg image size,
    ],
    'compiled': [
        True
    ],
    'random_reads': [True, False],
    'n': [3000]
})
class MemoryReadBytesBench(Benchmark):

    def __init__(self, num_samples, size_bytes, random_reads, n, compiled):
        self.num_samples = num_samples
        self.size_bytes = size_bytes
        self.random_reads = random_reads
        self.n = n
        self.compiled = compiled
        
    def __enter__(self):
        self.handle = NamedTemporaryFile()
        handle = self.handle.__enter__()
        name = handle.name
        dataset = DummyDataset(self.num_samples, self.size_bytes)
        writer = DatasetWriter(name, {
            'index': IntField(),
            'value': BytesField()
        }, num_workers=-1)

        writer.from_indexed_dataset(dataset,  chunksize=100)

        reader = Reader(name)
        manager = OSCacheManager(reader)
        context = manager.schedule_epoch(np.arange(self.num_samples))
        context.__enter__()
        self.context = context

        Compiler.set_enabled(self.compiled)
        memcpy_c = Compiler.compile(memcpy)

        read_fn = manager.compile_reader()

        if self.random_reads:
            indices = np.random.choice(self.num_samples, self.n, replace=False)
        else:
            indices = np.arange(self.num_samples)[:self.n]

        addresses = reader.alloc_table['ptr'][indices]
        
        self.buffer = np.zeros(self.size_bytes, dtype='<u1')
        
        def code(buff, state):
            for i in range(addresses.shape[0]):
                memcpy_c(read_fn(addresses[i], state), buff)
        
        self.code = Compiler.compile(code)

    
    def run(self):
        self.code(self.buffer, self.context.state)
        
    def __exit__(self, *args):
        self.handle.__exit__(*args)