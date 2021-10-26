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
from ffcv.memory_managers.ram import RAMMemoryManager

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
    'num_samples': [30000],
    'size_bytes': [
        32 * 32 * 3, # CIFAR RAW image size,
        500 * 300 * 3, # IMAGENET raw image size,
        128 * 1024, # IMAGENET jpg image size,
    ],
    'compiled': [True, False],
    'random_reads': [True, False],
    'n': [30000]
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
        writer = DatasetWriter(self.num_samples, name, {
            'index': IntField(),
            'value': BytesField()
        })

        with writer:
            writer.write_pytorch_dataset(dataset, num_workers=-1, chunksize=100)

        reader = Reader(name)
        manager = RAMMemoryManager(reader)

        Compiler.set_enabled(self.compiled)

        with manager:
            read_fn = manager.compile_reader()

        if self.random_reads:
            indices = np.random.choice(self.num_samples, self.n, replace=False)
        else:
            indices = np.arange(self.num_samples)[:self.n]

        addresses = reader.alloc_table['ptr'][indices]
        
        def code():
            result = 0
            for i in range(addresses.shape[0]):
                result += read_fn(addresses[i]).min()
            return result
            
        
        self.code = Compiler.compile(code)

    
    def run(self):
        self.code()
        
    def __exit__(self, *args):
        self.handle.__exit__(*args)