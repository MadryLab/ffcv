from tempfile import NamedTemporaryFile
import torch as ch
from tqdm import tqdm
import time
from multiprocessing import cpu_count

from assertpy import assert_that
import numpy as np
from torch.utils.data import Dataset
from ffcv import DatasetWriter
from ffcv.fields import IntField, NDArrayField
from ffcv import Loader
from ffcv.fields.basics import IntDecoder
from ffcv.fields.ndarray import NDArrayDecoder
from ffcv.loader.loader import OrderOption
from ffcv.transforms import ToDevice, ToTensor, Squeeze
import time

BATCH = 256
SIZE = 25_000
WORKERS = min(10, cpu_count())

class DummyArrayDataset(Dataset):
    def __init__(self, n_samples, shape):
        self.n_samples = n_samples
        self.shape = shape

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        if index >= self.n_samples:
            raise IndexError()
        np.random.seed(index)
        return (np.random.rand(SIZE) > 0.5).astype('bool'), np.random.rand(SIZE).astype('float32'), index

def run_experiment_cuda(weight, loader, sync=False):
    total = 0.
    X = ch.empty(BATCH, SIZE, device=weight.device)
    for X_bool, _, __ in tqdm(loader):
        if sync: ch.cuda.synchronize()
        X.copy_(X_bool)
        total += X @ weight
        total += X @ weight
        total += X @ weight

    return total.sum(0)

def run_cuda(weight, sync):
    n_samples, shape = (BATCH * WORKERS, (SIZE,))
    with NamedTemporaryFile() as handle:
        name = handle.name
        dataset = DummyArrayDataset(n_samples, shape)
        writer = DatasetWriter(name, {
            'mask': NDArrayField(dtype=np.dtype('bool'), shape=(SIZE,)),
            'targets': NDArrayField(dtype=np.dtype('float32'), shape=(SIZE,)),
            'idx': IntField()
        })

        writer.from_indexed_dataset(dataset)

        loader = Loader(
                name,
                batch_size=BATCH,
                num_workers=WORKERS,
                order=OrderOption.QUASI_RANDOM,
                indices=np.arange(n_samples),
                drop_last=False,
                cache_type=1,
                pipelines={
                    'mask': [NDArrayDecoder(), ToTensor(), ToDevice(ch.device('cuda:0'), non_blocking=False)],
                    'targets': [NDArrayDecoder(), ToTensor(), ToDevice(ch.device('cuda:0'), non_blocking=False)],
                    'idx': [IntDecoder(), ToTensor(), Squeeze(), ToDevice(ch.device('cuda:0'), non_blocking=False)]
                })

        return run_experiment_cuda(weight, loader, sync)

def test_cuda():
    weight = ch.randn(SIZE, SIZE).cuda()
    async_1 = run_cuda(weight, False)
    sync_1 = run_cuda(weight, True)
    sync_2 = run_cuda(weight, True)
    print(async_1)
    print(sync_1)
    print(sync_2)
    print(ch.abs(sync_1 - sync_2).max())
    print(ch.abs(sync_1 - async_1).max())
    assert ch.abs(sync_1 - sync_2).max().cpu().item() < float(WORKERS), 'Sync-sync mismatch'
    assert ch.abs(async_1 - sync_1).max().cpu().item() < float(WORKERS), 'Async-sync mismatch'

# test_cuda()