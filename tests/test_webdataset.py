from os import path
from glob import glob
import tempfile

import numpy as np
from tempfile import TemporaryDirectory, NamedTemporaryFile
import torch as ch
from torch.utils.data import Dataset

from ffcv import DatasetWriter
from ffcv.reader import Reader
from ffcv.fields import IntField, FloatField
from test_writer import validate_simple_dataset

field_names = [
    'index',
    'value.pyd'
]

class DummyDataset(Dataset):

    def __init__(self, l):
        self.l = l

    def __len__(self):
        return self.l

    def __getitem__(self, index):
        if index >= self.l:
            raise IndexError()
        return (index, np.sin(index))

def write_webdataset(folder, dataset, field_names):
    import webdataset as wds
    pattern = path.join(folder, "dataset-%06d.tar")
    writer = wds.ShardWriter(pattern, maxcount=20)
    with writer as sink:
        for i, sample in enumerate(dataset):
            data = {
                '__key__': f'sample_{i}'
            }

            for field_name, value in zip(field_names, sample):
                data[field_name] = value
            sink.write(data)


def pipeline(dataset):
    return (dataset
        .decode()
        .to_tuple(*field_names)
    )

if __name__ == '__main__':
    N = 1007
    dataset = DummyDataset(N)
    with TemporaryDirectory() as temp_directory:
        with NamedTemporaryFile() as handle:
            fname = handle.name
            write_webdataset(temp_directory, dataset, field_names)
            files = glob(path.join(temp_directory, '*'))
            files = list(sorted(files))

            print(fname)
            writer = DatasetWriter(fname, {
                'index': IntField(),
                'value': FloatField()
            })

            writer.from_webdataset(files, pipeline)

            validate_simple_dataset(fname, N, shuffled=False)