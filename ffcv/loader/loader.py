"""
FFCV loader
"""
import enum
from os import environ, sched_getaffinity
import ast
from multiprocessing import cpu_count
from re import sub
from typing import Any, Callable, Mapping, Sequence, Type, Union, Literal
from collections import defaultdict
from collections.abc import Collection
from enum import Enum, unique, auto

from ffcv.fields.base import Field

import torch as ch
import numpy as np

from .epoch_iterator import EpochIterator
from ..reader import Reader
from ..traversal_order.base import TraversalOrder
from ..traversal_order import Random, Sequential, QuasiRandom
from ..pipeline import Pipeline, PipelineSpec, Compiler
from ..pipeline.operation import Operation
from ..pipeline.graph import Graph
from ..memory_managers import (
    ProcessCacheManager, OSCacheManager, MemoryManager
)

@unique
class OrderOption(Enum):
    SEQUENTIAL = auto()
    RANDOM = auto()
    QUASI_RANDOM = auto()

ORDER_TYPE = Union[
    TraversalOrder,
    Literal[OrderOption.SEQUENTIAL,
            OrderOption.RANDOM]

]

ORDER_MAP: Mapping[ORDER_TYPE, TraversalOrder] = {
    OrderOption.RANDOM: Random,
    OrderOption.SEQUENTIAL: Sequential,
    OrderOption.QUASI_RANDOM: QuasiRandom
}

DEFAULT_PROCESS_CACHE = int(environ.get('FFCV_DEFAULT_CACHE_PROCESS', "0"))


class Loader:
    """FFCV loader class that can be used as a drop-in replacement
    for standard (e.g. PyTorch) data loaders.

    Parameters
    ----------
    fname: str
        Full path to the location of the dataset (.beton file format).
    batch_size : int
        Batch size.
    num_workers : int
        Number of workers used for data loading. Consider using the actual number of cores instead of the number of threads if you only use JITed augmentations as they usually don't benefit from hyper-threading.
    os_cache : bool
        Leverages the operating for caching purposes. This is beneficial when there is enough memory to cache the dataset and/or when multiple processes on the same machine training using the same dataset. See https://docs.ffcv.io/performance_guide.html for more information.
    order : Union[OrderOption, TraversalOrder]
        Traversal order, one of: SEQUENTIAL, RANDOM, QUASI_RANDOM, or a custom TraversalOrder

        QUASI_RANDOM is a random order that tries to be as uniform as possible while minimizing the amount of data read from the disk. Note that it is mostly useful when `os_cache=False`. Currently unavailable in distributed mode.
    distributed : bool
        For distributed training (multiple GPUs). Emulates the behavior of DistributedSampler from PyTorch.
    seed : int
        Random seed for batch ordering.
    indices : Sequence[int]
        Subset of dataset by filtering only some indices.
    pipelines : Mapping[str, Sequence[Union[Operation, torch.nn.Module]]
        Dictionary defining for each field the sequence of Decoders and transforms to apply.
        Fileds with missing entries will use the default pipeline, which consists of the default decoder and `ToTensor()`,
        but a field can also be disabled by explicitly by passing `None` as its pipeline.
    custom_fields : Mapping[str, Field]
        Dictonary informing the loader of the types associated to fields that are using a custom type.
    drop_last : bool
        Drop non-full batch in each iteration.
    batches_ahead : int
        Number of batches prepared in advance; balances latency and memory.
    recompile : bool
        Recompile every iteration. This is necessary if the implementation of some augmentations are expected to change during training.
    """
    def __init__(self,
                 fname: str,
                 batch_size: int,
                 num_workers: int = -1,
                 cache_type: int = DEFAULT_PROCESS_CACHE,
                 order: Union[ORDER_TYPE, TraversalOrder] = OrderOption.SEQUENTIAL,
                 distributed: bool = False,
                 seed: int = None,  # For ordering of samples
                 indices: Sequence[int] = None,  # For subset selection
                 pipelines: Mapping[str,
                                    Sequence[Union[Operation, ch.nn.Module]]] = {},
                 custom_fields: Mapping[str, Type[Field]] = {},
                 drop_last: bool = True,
                 batches_ahead: int = 3,
                 recompile: bool = False,  # Recompile at every epoch
                 order_kwargs: dict = dict(),
                 ):

        if distributed and order == OrderOption.RANDOM and (seed is None):
            print('Warning: no ordering seed was specified with distributed=True. '
                  'Setting seed to 0 to match PyTorch distributed sampler.')
            seed = 0
        elif seed is None:
            tinfo = np.iinfo('int32')
            seed = np.random.randint(0, tinfo.max)

        # We store the original user arguments to be able to pass it to the
        # filtered version of the datasets
        self._args = {
            'fname': fname,
            'batch_size': batch_size,
            'num_workers': num_workers,
            'cache_type': cache_type,
            'order': order,
            'distributed': distributed,
            'seed': seed,
            'indices': indices,
            'pipelines': pipelines,
            'drop_last': drop_last,
            'batches_ahead': batches_ahead,
            'recompile': recompile
        }
        self.fname: str = fname
        self.batch_size: int = batch_size
        self.batches_ahead = batches_ahead
        self.seed: int = seed
        self.reader: Reader = Reader(self.fname, custom_fields)
        self.num_workers: int = num_workers
        self.drop_last: bool = drop_last
        self.distributed: bool = distributed
        self.code = None
        self.recompile = recompile

        if self.num_workers < 1:
            self.num_workers = len(sched_getaffinity(0))

        Compiler.set_num_threads(self.num_workers)

        if indices is None:
            self.indices = np.arange(self.reader.num_samples, dtype='uint64')
        else:
            self.indices = np.array(indices)

        if cache_type == 0:
            self.memory_manager: MemoryManager = OSCacheManager(self.reader)
        elif cache_type == 1:
            self.memory_manager: MemoryManager = ProcessCacheManager(
                self.reader)
        elif cache_type == 2:
            from ffcv.memory_managers.shared_cache import SharedMemoryManager
            self.memory_manager: MemoryManager = SharedMemoryManager(self.reader)
        else:
            raise ValueError("Unknown cache type. Use 0 for process cache, 1 for os cache, or 2 for no cache.")
        
        if order in ORDER_MAP:
            self.traversal_order: TraversalOrder = ORDER_MAP[order](self)
        elif isinstance(order, TraversalOrder):
            self.traversal_order: TraversalOrder = order(self)
        elif issubclass(order, TraversalOrder):
            self.traversal_order: TraversalOrder = order(self)
        else:
            raise ValueError(f"Order {order} is not a supported order type or a subclass of TraversalOrder")

        self.compile_pipeline(pipelines)
        
        
        self.next_epoch: int = 0

        self.first_traversal_order = self.next_traversal_order()

    def compile_pipeline(self,pipelines):
        
        self.pipeline_specs = {}
        self.field_name_to_f_ix = {}
        
        custom_pipeline_specs = {}

        # Creating PipelineSpec objects from the pipeline dict passed
        # by the user
        for output_name, spec in pipelines.items():
            if isinstance(spec, PipelineSpec):
                pass
            elif isinstance(spec, Sequence):
                spec = PipelineSpec(output_name, decoder=None, transforms=spec)
            elif spec is None:
                continue  # This is a disabled field
            else:
                msg = f"The pipeline for {output_name} has to be "
                msg += f"either a PipelineSpec or a sequence of operations"
                raise ValueError(msg)
            custom_pipeline_specs[output_name] = spec

        # Adding the default pipelines, WARN: disable default pipelines
        for f_ix, (field_name, field) in enumerate(self.reader.handlers.items()):
            self.field_name_to_f_ix[field_name] = f_ix

            if field_name not in custom_pipeline_specs:
                # We add the default pipeline
                if field_name not in pipelines:
                    self.pipeline_specs[field_name] = PipelineSpec(field_name)
            else:
                self.pipeline_specs[field_name] = custom_pipeline_specs[field_name]

        # We add the custom fields after the default ones
        # This is to preserve backwards compatibility and make sure the order
        # is intuitive
        for field_name, spec in custom_pipeline_specs.items():
            if field_name not in self.pipeline_specs:
                self.pipeline_specs[field_name] = spec

        memory_read = self.memory_manager.compile_reader()
        self.graph = Graph(self.pipeline_specs, self.reader.handlers,
                           self.field_name_to_f_ix, self.reader.metadata,
                           memory_read)
        
        self.generate_code()

    def next_traversal_order(self):
        return self.traversal_order.sample_order(self.next_epoch)

    def __iter__(self):
        Compiler.set_num_threads(self.num_workers)
        order = self.next_traversal_order()
        selected_order = order[:len(self) * self.batch_size]
        self.next_epoch += 1

        # Compile at the first epoch
        if self.code is None or self.recompile:
            self.generate_code()

        return EpochIterator(self, selected_order)

    def filter(self, field_name:str, condition: Callable[[Any], bool]) -> 'Loader':
        new_args = {**self._args}
        pipelines = {}

        # Disabling all the other fields
        for other_field_name in self.reader.handlers.keys():
            pipelines[other_field_name] = None

        # We reuse the original pipeline for the field we care about
        try:
            pipelines[field_name] = new_args['pipelines'][field_name]
        except KeyError:
            # We keep the default one if the user didn't setup a custom one
            del pipelines[field_name]
            pass

        new_args['pipelines'] = pipelines

        # We use sequential order for speed and to know which index we are
        # filtering
        new_args['order'] = OrderOption.SEQUENTIAL
        new_args['drop_last'] = False
        sub_loader = Loader(**new_args)
        selected_indices = []

        # Iterate through the loader and test the user defined condition
        for i, (batch,) in enumerate(sub_loader):
            for j, sample in enumerate(batch):
                sample_id = i * self.batch_size + j
                if condition(sample):
                    selected_indices.append(sample_id)

        final_args = {**self._args}
        final_args['indices'] = np.array(selected_indices)
        return Loader(**final_args)


    def __len__(self):
        next_order = self.first_traversal_order
        if self.drop_last:
            return len(next_order) // self.batch_size
        else:
            return int(np.ceil(len(next_order) / self.batch_size))



    def generate_code(self):
        queries, code = self.graph.collect_requirements()
        self.code = self.graph.codegen_all(code)
        

