#%%

import time
from PIL import Image# a trick to solve loading lib problem
from ffcv.fields.rgb_image import * 
from ffcv.transforms import  RandomHorizontalFlip, NormalizeImage,  ToTensor, ToTorchImage, ToDevice
import numpy as np
import torchvision

from ffcv import Loader
import ffcv
import argparse
from tqdm.auto import tqdm,trange
import torch.nn as nn
import torch
from psutil import Process, net_io_counters
import json
from os import getpid

from ffcv.transforms.ops import Convert

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255

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


def load_one_epoch(args,loader):
    start = time.time()
    l=ramqdm(loader)
    
    for x1,y in l:
        x_std = x1.float().flatten(1).std(1)
        if x_std.min() <= 0:
            torchvision.utils.save_image(x1.float(), "invalid_sample.png", normalize=True)
            # assert x_std.min() > 0, "invalid sample"
    end = time.time()
    res = l.summary()
    throughput=loader.reader.num_samples/(end-start)
    res['throughput'] = throughput
    x1 = x1.float()
    print("Mean: ", x1.mean().item(), "Std: ", x1.std().item())
    return res

def main(args):
    # pipe = ThreeAugmentPipeline()
    pipe = {
        'image': [RandomResizedCropRGBImageDecoder((args.img_size,args.img_size)),
            RandomHorizontalFlip(),
            ToTensor(), 
            # ToDevice(torch.device('cuda')),
            ToTorchImage(),
            # NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16),            
            # Convert(torch.float16),
        ]
    }
    loader = Loader(args.data_path, batch_size=args.batch_size, num_workers=args.num_workers, 
         pipelines=pipe, 
        batches_ahead=2, distributed=False,seed=0,drop_last=True)
    
    decoder = loader.pipeline_specs['image'].decoder    
    decoder.use_crop_decode = (args.use_ffcv)
        
    # warmup
    load_one_epoch(args,loader)
    
    for _ in range(args.repeat):
        res = load_one_epoch(args,loader)    
        yield res

#%%    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="FFCV Profiler")
    parser.add_argument("-r", "--repeat", type=int, default=5, help="number of samples to record one step for profile.")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("-p", "--data_path", type=str, help="data path", required=True)
    parser.add_argument("--use_ffcv",default=False,action="store_true")
    parser.add_argument("--num_workers", type=int, default=10, help="number of workers")
    parser.add_argument("--exp", default=False, action="store_true", help="run experiments")
    parser.add_argument("--img_size", type=int, default=224, help="image size")
    parser.add_argument("--write_path", type=str, help='path to write result',default=None)
    args = parser.parse_args()
    if args.exp == False:
        for res  in main(args):
            throughput = res['throughput']
            print(f"Throughput: {throughput:.2f} samples/s for {args.data_path}.")
            res.update(args.__dict__)
            if args.write_path:
                with open(args.write_path,"a") as file:
                    file.write(json.dumps(res)+"\n")
    else:
        data = []
        with open(args.write_path,"a") as file:
            for num_workers in [10,20,40]:
                for use_ffcv in [False,True]:
                    for bs in [128,256,512]:
                        args.num_workers=num_workers
                        args.batch_size = bs
                        args.use_ffcv=use_ffcv
                        row = args.__dict__
                        for res  in main(args):
                            row.update(res)
                            file.write(json.dumps(row)+"\n")
                            file.flush()
                            print(row)
                        data.append(row)
        import pandas as pd
        df = pd.DataFrame(data)
        print(df)
    exit(0)