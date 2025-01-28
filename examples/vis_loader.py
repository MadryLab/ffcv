import argparse
import time
from PIL import Image # a trick to solve loading lib problem
from ffcv import Loader
from ffcv.transforms import *
from ffcv.fields.decoders import CenterCropRGBImageDecoder, RandomResizedCropRGBImageDecoder


import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FFCV Profiler')
    parser.add_argument('data_path', type=str, default='data/imagenet', help='Path to the dataset')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--write_path', type=str, default='viz.png', help='Path to write result')
    args = parser.parse_args()
    
    loader = Loader(args.data_path, batch_size=args.batch_size, num_workers=10, cache_type=0, pipelines={
        'image':[CenterCropRGBImageDecoder((224, 224),224/256), 
                 ToTensor(), 
                 ToTorchImage()]
    }, batches_ahead=0,)
    
    print("num samples: ", loader.reader.num_samples, "fields: ", loader.reader.field_names)
    for x,_ in loader:
        x1 = x.float()
        print("Mean: ", x1.mean().item(), "Std: ", x1.std().item())
        break
    
    print('Done')
    num = int(np.sqrt(args.batch_size))
    import cv2
    
    image = np.zeros((224*num, 224*num, 3), dtype=np.uint8)
    for i in range(num):
        for j in range(num):
            if i*num+j >= args.batch_size:
                break
            img = x[i*num+j].numpy().transpose(1,2,0)
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            image[i*224:(i+1)*224, j*224:(j+1)*224] = (img).astype(np.uint8)
    
    if args.write_path:        
        Image.fromarray(image).save(args.write_path)
    

