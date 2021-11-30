from typing import List, Mapping
from os import SEEK_END, path
import numpy as np
from time import sleep
import ctypes
from multiprocessing import (shared_memory, cpu_count, Queue, Process, Value)

from tqdm import tqdm

from .utils import chunks, is_power_of_2
from .fields.base import Field
from .memory_allocator import MemoryAllocator
from .types import (TYPE_ID_HANDLER, get_metadata_type, HeaderType,
                    FieldDescType, CURRENT_VERSION, ALLOC_TABLE_TYPE)
                    

MIN_PAGE_SIZE = 1 << 21  # 2MiB, which is the most common HugePage size

def handle_sample(sample, dest_ix, field_name, metadata, allocator):
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


def worker_job_indexed_dataset(input_queue, metadata_sm, metadata_type, fields,
               allocator, done_number, dataset, allocations_queue):

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
                # We should only have to retry at least one
                sample = dataset[source_ix]
                handle_sample(sample, dest_ix, field_name, metadata, allocator)

            # We warn the main thread of our progress
            with done_number.get_lock():
                done_number.value += len(chunk)

    allocations_queue.put(allocator.allocations)


class DatasetWriter():
    def __init__(self, num_samples: int, fname: str, fields: Mapping[str, Field],
                 page_size: int = 4 * MIN_PAGE_SIZE):
        self.num_samples = num_samples
        self.fields = fields
        self.fname = fname
        self.metadata_type = get_metadata_type(list(self.fields.values()))

        if not is_power_of_2(page_size):
            raise ValueError(f'page_size isnt a power of 2')
        if page_size < MIN_PAGE_SIZE:
            raise ValueError(f"page_size can't be lower than{MIN_PAGE_SIZE}")

        self.page_size = page_size


    def __enter__(self):
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
                type_id = field_type_to_type_id[type(field)]
                encoded_name = name.encode('ascii')
                encoded_name = np.frombuffer(encoded_name, dtype='<u1')
                actual_length = min(fieldname_max_len, len(encoded_name))
                fields_descriptor[i]['type_id'] = type_id
                fields_descriptor[i]['name'][:actual_length] = (
                    encoded_name[:actual_length])
                fields_descriptor[i]['arguments'][:] = field.to_binary()[0]

            fp.write(fields_descriptor.tobytes())


        # Makes a memmap to the metadata for the samples

        total_metadata_size = self.num_samples * self.metadata_type.itemsize

        # Shared memory for all the writers to fill the information
        self.metadata_sm = 3
        self.metadata_start = (HeaderType.itemsize + fields_descriptor.nbytes)

        self.metadata_sm = shared_memory.SharedMemory(create=True,
                                                      size=total_metadata_size)

        self.data_region_start = self.metadata_start + total_metadata_size
        self.allocation_list = []



    def write_pytorch_dataset(self, dataset, num_workers=-1,
                              indices: List[int]=None, chunksize=100,
                              shuffle_indices: bool = False):

        # We use all cores by default
        if num_workers == -1:
            num_workers = cpu_count()

        # If the user didn't specify an order we just add samples
        # sequentially
        if indices is None:
            indices = np.arange(self.num_samples)
            
        if shuffle_indices:
            np.random.shuffle(indices)
            
        # We add indices to the indices so that workers know where
        # to write in the metadata array
        indices: List[int] = list(enumerate(indices))

        # We publish all the work that has to be done into a queue
        workqueue: Queue = Queue()
        for chunk in chunks(indices, chunksize):
            workqueue.put(chunk)

        # This will contain all the memory allocations each worker
        # produced. This will go at the end of the file
        allocations_queue: Queue = Queue()

        # We add a token for each worker to warn them that there
        # is no more work to be done
        for _ in range(num_workers):
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
                       dataset, allocations_queue)

        # Create the workers
        processes = [Process(target=worker_job_indexed_dataset, args=worker_args)
                     for _ in range(num_workers)]
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
        self.allocation_list.extend([allocations_queue.get() for p in processes])

    def finalize(self) :
        allocations = self.allocation_list
        # Writing metadata
        with open(self.fname, 'r+b') as fp:
            fp.seek(self.metadata_start)
            fp.write(self.metadata_sm.buf)

            # We go at the end of the file
            # Look at the current address
            allocation_table_location = fp.tell()
            # Retrieve all the allocations from the workers
            # Turn them into a numpy array
            try:
                allocation_table = np.concatenate([
                    np.array(x).view(ALLOC_TABLE_TYPE) for x in allocations if len(x)
                ])
            except:
                allocation_table = np.array([]).view(ALLOC_TABLE_TYPE)
            # print(allocation_table)
            fp.write(allocation_table.tobytes())
            self.header['alloc_table_ptr'] = allocation_table_location
            # We go at the start of the file
            fp.seek(0)
            # And write the header
            fp.write(self.header.tobytes())



    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finalize()
        self.metadata_sm.close()
        self.metadata_sm.unlink()