import argparse
import builtins
import datetime
import json
import math
import os
import sys
import time
from pathlib import Path

from ffcv.loader import Loader, OrderOption
import gin
import numpy as np
import timm
import torch.backends.cudnn as cudnn
from PIL import Image  # a trick to solve loading lib problem
from tqdm import tqdm

assert timm.__version__ >= "0.6.12"  # version check
from torchvision import datasets
import ffcv

from psutil import Process, net_io_counters
import socket
import json
from os import getpid

from ffcv.transforms import ToTensor, ToDevice, ToTorchImage, NormalizeImage,RandomHorizontalFlip, View, Convert
from ffcv.fields.decoders import IntDecoder, RandomResizedCropRGBImageDecoder, SimpleRGBImageDecoder, CenterCropRGBImageDecoder

import torch

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255

@gin.configurable
def SimplePipeline(img_size=224,scale=(0.2,1), ratio=(3.0/4.0, 4.0/3.0),device='cuda'):
    device = torch.device(device)
    image_pipeline = [
            RandomResizedCropRGBImageDecoder((img_size, img_size), scale=scale,ratio=ratio,),
            RandomHorizontalFlip(),          
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float32),  
            ToTensor(),  ToTorchImage(),
            ]
    label_pipeline = [IntDecoder(), ToTensor(),ToDevice(device), View(-1)]
    # Pipeline for each data field
    pipelines = {
        'image': image_pipeline,
        'label': label_pipeline
    } 
    return pipelines    


def get_args_parser():
    parser = argparse.ArgumentParser('Data loading benchmark', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus)')
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--img_size', default=224,type=int)

    # Dataset parameters
    parser.add_argument('--data_set', default='ffcv')
    parser.add_argument("--cache_type",type=int, default=0,)
    parser.add_argument('--data_path', default=os.getenv("IMAGENET_DIR"), type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default=None, type=str,
                        help='path where to save, empty for no saving')

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local-rank','--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


class ramqdm(tqdm):
    """tqdm progress bar that reports RAM usage with each update"""
    _empty_desc = "using ? GB RAM; ?  CPU ? IO"
    _desc = "{:.2f} GB RAM; {:.2f} % CPU {:.2f} MB IO"
    _GB = 10**9
    """"""
    def __init__(self, *args, **kwargs):
        """Override desc and get reference to current process"""
        if "desc" in kwargs:
            # prepend desc to the reporter mask:
            self._empty_desc = kwargs["desc"] + " " + self._empty_desc
            self._desc = kwargs["desc"] + " " + self._desc
            del kwargs["desc"]
        else:
            # nothing to prepend, reporter mask is at start of sentence:
            self._empty_desc = self._empty_desc.capitalize()
            self._desc = self._desc.capitalize()
        super().__init__(*args, desc=self._empty_desc, **kwargs)
        self._process = Process(getpid())
        self.metrics = []
    """"""
    def update(self, n=1):
        """Calculate RAM usage and update progress bar"""
        rss = self._process.memory_info().rss
        ps = self._process.cpu_percent()
        io_counters = self._process.io_counters().read_bytes
        # net_io = net_io_counters().bytes_recv
        # io_counters += net_io
        
        current_desc = self._desc.format(rss/self._GB, ps, io_counters/1e6)
        self.set_description(current_desc)
        self.metrics.append({'mem':rss/self._GB, 'cpu':ps, 'io':io_counters/1e6})
        super().update(n)
    
    def summary(self):
        res = {}
        for key in self.metrics[0].keys():
            res[key] = np.mean([i[key] for i in self.metrics])
        return res

@gin.configurable(denylist=["args"])
def build_dataset(args, transform_fn=SimplePipeline):
    transform_train = transform_fn(img_size=args.img_size)
    if args.data_set == 'IF':
        # simple augmentation
        dataset_train = datasets.ImageFolder(args.data_path, transform=transform_train)
    elif args.data_set == 'cifar10':
        dataset_train = datasets.CIFAR10(args.data_path, transform=transform_train)    
    elif args.data_set == 'ffcv':
        order = OrderOption.RANDOM if args.distributed else OrderOption.QUASI_RANDOM
        dataset_train =  Loader(args.data_path, pipelines=transform_train,
                            batch_size=args.batch_size, num_workers=args.num_workers,
                            batches_ahead=4, #cache_type=args.cache_type,
                            order=order, distributed=args.distributed,seed=args.seed,drop_last=True)
    else:
        raise ValueError("Wrong dataset: ", args.data_set)
    return dataset_train

def load_one_epoch(args,loader):
    start = time.time()
    l=ramqdm(loader,disable=args.rank>0)
    
    for x1,y in l:
        x1.mean()
        torch.cuda.synchronize()
        
    end = time.time()

    if args.rank ==0:
        res = l.summary()
        throughput=loader.reader.num_samples/(end-start)
        res['throughput'] = throughput
        return res

import torch

def main(args):
    init_distributed_mode(args)
    
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    cudnn.benchmark = True

    # build dataset
    dataset_train = build_dataset(args)
    
    num_tasks = args.world_size
    global_rank = args.rank
    if args.data_set != "ffcv":
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )
    else:
        data_loader_train = dataset_train
    
    for epoch in range(args.epochs):
        res = load_one_epoch(args,data_loader_train)    
        if res:
            throughput = res['throughput']
            print(f"Throughput: {throughput:.2f} samples/s for {args.data_path}.")
            res.update(args.__dict__)
            res['version'] = ffcv.__version__
            res['hostname'] = socket.gethostname()
            res['epoch'] = epoch
            if args.output_dir:
                with open(os.path.join(args.output_dir,"data_loading.txt"),"a") as file:
                    file.write(json.dumps(res)+"\n")


def init_distributed_mode(args):
    if hasattr(args,'dist_on_itp') and args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), *args, **kwargs)  # print with time stamp

    builtins.print = print

    

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
