from functools import partial
from typing import Callable, List, Mapping
from os import SEEK_END, path, sched_getaffinity
import numpy as np
from time import sleep
import ctypes
from multiprocessing import (shared_memory, cpu_count, Queue, Process, Value)

from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

from .utils import chunks, is_power_of_2
from .fields.base import Field
from .memory_allocator import MemoryAllocator
from .types import (TYPE_ID_HANDLER, get_metadata_type, HeaderType,
                    FieldDescType, CURRENT_VERSION, ALLOC_TABLE_TYPE)


MIN_PAGE_SIZE = 1 << 21  # 2MiB, which is the most common HugePage size
MAX_PAGE_SIZE = 1 << 32  # Biggest page size that will not overflow uint32

def from_shard(shard, pipeline):
    # We import webdataset here so that it desn't crash if it's not required
    # (Webdataset is an optional depdency)
    from webdataset import WebDataset

    dataset = WebDataset(shard)
    dataset = pipeline(dataset)
    return dataset

def count_samples_in_shard(shard, pipeline):
    #
    # We count the length of the dataset
    # We are not using __len__ since it might not be implemented
    count = 0
    for _ in from_shard(shard, pipeline):
        count += 1

    return count


def handle_sample(sample, dest_ix, field_names, metadata, allocator, fields):
    # We should only have to retry at least one
    for i in range(2):
        try:
            allocator.set_current_sample(dest_ix)
            # We extract the sample in question from the dataset
            # We write each field individually to the metadata region
            for field_name, field, field_value in zip(field_names, fields.values(), sample):
                destination = metadata[field_name][dest_ix: dest_ix + 1]
                field.encode(destination, field_value, allocator.malloc)
            # We managed to write all the data without reaching
            # the end of the page so we stop retrying
            break
        # If we couldn't fit this sample in the previous page we retry once from a fresh page
        except MemoryError:
            # We raise the error if it happens on the second try
            if i == 1:
                raise

def worker_job_webdataset(input_queue, metadata_sm, metadata_type, fields,
               allocator, done_number, allocations_queue, pipeline):

    metadata = np.frombuffer(metadata_sm.buf, dtype=metadata_type)
    field_names = metadata_type.names

    # This `with` block ensures that all the pages allocated have been written
    # onto the file
    with allocator:
        while True:
            todo = input_queue.get()

            if todo is None:
                # No more work left to do
                break

            shard, offset = todo

            # For each sample in the chunk
            done = 0
            for i, sample in enumerate(from_shard(shard, pipeline)):
                done += 1
                dest_ix = offset + i
                handle_sample(sample, dest_ix, field_names, metadata, allocator, fields)

            # We warn the main thread of our progress
            with done_number.get_lock():
                done_number.value += done

    allocations_queue.put(allocator.allocations)



def worker_job_indexed_dataset(input_queue, metadata_sm, metadata_type, fields,
               allocator, done_number, allocations_queue, dataset):

    metadata = np.frombuffer(metadata_sm.buf, dtype=metadata_type)
    field_names = metadata_type.names

    # This `with` block ensures that all the pages allocated have been written
    # onto the file
    with allocator:
        while True:
            chunk = input_queue.get()

            if chunk is None:
                # No more work left to do
                break

            # For each sample in the chunk
            for dest_ix, source_ix in chunk:
                sample = dataset[source_ix]
                handle_sample(sample, dest_ix, field_names, metadata, allocator, fields)

            # We warn the main thread of our progress
            with done_number.get_lock():
                done_number.value += len(chunk)

    allocations_queue.put(allocator.allocations)

def worker_foder_dataset(input_queue, metadata_sm, metadata_type, fields,
               allocator, done_number, allocations_queue, dataset):

    metadata = np.frombuffer(metadata_sm.buf, dtype=metadata_type)
    field_names = metadata_type.names

    # This `with` block ensures that all the pages allocated have been written
    # onto the file
    with allocator:
        while True:
            chunk = input_queue.get()

            if chunk is None:
                # No more work left to do
                break

            # For each sample in the chunk
            for dest_ix, source_ix in chunk:
                sample = dataset[source_ix]
                handle_sample(sample, dest_ix, field_names, metadata, allocator, fields)

            # We warn the main thread of our progress
            with done_number.get_lock():
                done_number.value += len(chunk)

    allocations_queue.put(allocator.allocations)



class DatasetWriter():
    """Writes given dataset into FFCV format (.beton).
    Supports indexable objects (e.g., PyTorch Datasets) and webdataset.

    Parameters
    ----------
    fname: str
        File name to store dataset in FFCV format (.beton)
    fields : Mapping[str, Field]
        Map from keys to Field's (order matters!)
    page_size : int
        Page size used internally
    num_workers : int
        Number of processes to use
    """
    def __init__(self, fname: str, fields: Mapping[str, Field],
                 page_size: int = 4 * MIN_PAGE_SIZE, num_workers: int = -1):
        self.fields = fields
        self.fname = fname
        self.metadata_type = get_metadata_type(list(self.fields.values()))

        self.num_workers = num_workers
        # We use all cores by default
        if self.num_workers < 1:
            self.num_workers = len(sched_getaffinity(0))

        if not is_power_of_2(page_size):
            raise ValueError(f'page_size isnt a power of 2')
        if page_size < MIN_PAGE_SIZE:
            raise ValueError(f"page_size can't be lower than{MIN_PAGE_SIZE}")
        if page_size >= MAX_PAGE_SIZE:
            raise ValueError(f"page_size can't be bigger(or =) than{MAX_PAGE_SIZE}")

        self.page_size = page_size

    def prepare(self):

        with open(self.fname, 'wb') as fp:
            # Prepare the header data
            header = np.zeros(1, dtype=HeaderType)[0]
            header['version'] = CURRENT_VERSION
            header['num_samples'] = self.num_samples
            header['num_fields'] = len(self.fields)
            header['page_size'] = self.page_size
            self.header = header

            # We will write the header at the end because we need to know where
            # The memory allocation table is in the file
            # We still write it here to make space for it later
            fp.write(self.header.tobytes())


            # Writes the information about the fields
            fields_descriptor = np.zeros(len(self.fields),
                                              dtype=FieldDescType)
            field_type_to_type_id = {v: k for (k, v) in TYPE_ID_HANDLER.items()}

            fieldname_max_len = fields_descriptor[0]['name'].shape[0]

            for i, (name, field) in enumerate(self.fields.items()):
                type_id = field_type_to_type_id.get(type(field), 255)
                encoded_name = name.encode('ascii')
                encoded_name = np.frombuffer(encoded_name, dtype='<u1')
                actual_length = min(fieldname_max_len, len(encoded_name))
                fields_descriptor[i]['type_id'] = type_id
                fields_descriptor[i]['name'][:actual_length] = (
                    encoded_name[:actual_length])
                fields_descriptor[i]['arguments'][:] = field.to_binary()[0]

            fp.write(fields_descriptor.tobytes())

        total_metadata_size = self.num_samples * self.metadata_type.itemsize

        # Shared memory for all the writers to fill the information
        self.metadata_sm = 3
        self.metadata_start = (HeaderType.itemsize + fields_descriptor.nbytes)

        self.metadata_sm = shared_memory.SharedMemory(create=True,
                                                      size=total_metadata_size)

        self.data_region_start = self.metadata_start + total_metadata_size


    def _write_common(self, num_samples, queue_content, work_fn, extra_worker_args):
        self.num_samples = num_samples

        self.prepare()
        allocation_list = []

        # Makes a memmap to the metadata for the samples

        # We publish all the work that has to be done into a queue
        workqueue: Queue = Queue()
        for todo in queue_content:
            workqueue.put(todo)

        # This will contain all the memory allocations each worker
        # produced. This will go at the end of the file
        allocations_queue: Queue = Queue()

        # We add a token for each worker to warn them that there
        # is no more work to be done
        for _ in range(self.num_workers):
            workqueue.put(None)

        # Define counters we need to orchestrate the workers
        done_number = Value(ctypes.c_uint64, 0)
        allocator = MemoryAllocator(self.fname,
                                    self.data_region_start,
                                    self.page_size)

        # Arguments that have to be passed to the workers
        worker_args = (workqueue, self.metadata_sm,
                       self.metadata_type, self.fields,
                       allocator, done_number,
                       allocations_queue, *extra_worker_args)

        # Create the workers
        processes = [Process(target=work_fn, args=worker_args)
                     for _ in range(self.num_workers)]
        # start the workers
        for p in processes: p.start()
        # Wait for all the workers to be done

        # Display progress
        progress = tqdm(total=self.num_samples)
        previous = 0
        while previous != self.num_samples:
            val = done_number.value
            diff = val - previous
            if diff > 0:
                progress.update(diff)
            previous = val
            sleep(0.1)
        progress.close()

        # Wait for all the workers to be done and get their allocations
        for p in processes:
            content = allocations_queue.get()
            allocation_list.extend(content)

        self.finalize(allocation_list)
        self.metadata_sm.close()
        self.metadata_sm.unlink()


    def from_indexed_dataset(self, dataset,
                              indices: List[int]=None, chunksize=100,
                              shuffle_indices: bool = False):
        """Read dataset from an indexable dataset.
        See https://docs.ffcv.io/writing_datasets.html#indexable-dataset for sample usage.

        Parameters
        ----------
        dataset: Indexable
            An indexable object that implements `__getitem__` and `__len__`.
        indices : List[int]
            Use a subset of the dataset specified by indices.
        chunksize : int
            Size of chunks processed by each worker during conversion.
        shuffle_indices : bool
            Shuffle order of the dataset.
        """
        # If the user didn't specify an order we just add samples
        # sequentially
        if indices is None:
            indices = np.arange(len(dataset))

        if shuffle_indices:
            np.random.shuffle(indices)

        # We add indices to the indices so that workers know where
        # to write in the metadata array
        indices: List[int] = list(enumerate(indices))

        self._write_common(len(indices), chunks(indices, chunksize),
                           worker_job_indexed_dataset, (dataset, ))


    def from_webdataset(self, shards: List[str], pipeline: Callable):
        """Read from webdataset-like format.
        See https://docs.ffcv.io/writing_datasets.html#webdataset for sample usage.

        Parameters
        ----------
        shards: List[str]
            List of shards that comprise the dataset folder.
        pipeline: Callable
            Called by each worker to decode. Similar to pipelines used to load webdataset.
        """
        counter = partial(count_samples_in_shard, pipeline=pipeline)
        lengths = thread_map(counter, shards, max_workers=self.num_workers)
        total_len = sum(lengths)

        offsets = np.cumsum([0] + lengths)[:-1]

        todos = zip(shards, offsets)
        self._write_common(total_len, todos, worker_job_webdataset, (pipeline, ))

    def from_image_folder(self, root: str):
        """Read from image folder.
        The images with the same class should be in the same folder.
        
        Parameters
        ----------
        root: str
            Path to the folder containing the images.
        """
        import glob
        from torchvision.datasets.folder import DatasetFolder
        classes, class_to_idx = DatasetFolder.find_classes(root)
        samples = DatasetFolder.make_dataset(root, class_to_idx)   
        total_len = len(samples)     
        self._write_common(total_len, samples, worker_foder_dataset, (class_to_idx, ))

        

    def finalize(self, allocations) :
        # Writing metadata
        with open(self.fname, 'r+b') as fp:
            fp.seek(self.metadata_start)
            fp.write(self.metadata_sm.buf)

            # We go at the end of the file
            fp.seek(0, SEEK_END)

            # Look at the current address
            allocation_table_location = fp.tell()
            # Retrieve all the allocations from the workers
            # Turn them into a numpy array
            try:
                allocation_table = np.array([
                    np.array(x, dtype=ALLOC_TABLE_TYPE) for x in allocations if len(x)
                ])
            except:
                allocation_table = np.array([]).view(ALLOC_TABLE_TYPE)
            fp.write(allocation_table.tobytes())

            self.header['alloc_table_ptr'] = allocation_table_location
            # We go at the start of the file
            fp.seek(0)
            # And write the header
            fp.write(self.header.tobytes())
