from os import path

import numpy as np
import cv2
from numpy.core.numeric import full

from ..decorator import benchmark
from ..benchmark import Benchmark

from ...pipeline.compiler import Compiler

from ...libffcv import imdecode

@benchmark({
    'n': [500],
    'source_image': ['../../../test_data/pig.png'],
    'image_width': [224, 500, 1024],
    'quality': [50, 80, 90, 95],
    'compile': [True],
})
class JPEGDecodeBenchmark(Benchmark):

    def __init__(self, n, source_image, image_width, quality, compile):
        self.n = n
        self.compile = compile
        self.source_image = source_image
        self.image_width = image_width
        self.quality = quality

    def __enter__(self):
        full_path = path.join(path.dirname(__file__), self.source_image)
        loaded_image = cv2.imread(full_path, cv2.IMREAD_COLOR)   
        previous_width = loaded_image.shape[1]
        new_width = self.image_width
        factor = new_width / previous_width
        new_height = int(loaded_image.shape[0] * factor)
        resized_image = cv2.resize(loaded_image, (new_width, new_height),
                                   interpolation=cv2.INTER_AREA)
        _, self.encoded_image = cv2.imencode('.jpg', resized_image,
                                          [int(cv2.IMWRITE_JPEG_QUALITY),
                                              self.quality])
                                          
        self.destination = np.zeros((new_height, new_width, 3), dtype='uint8')
        
        Compiler.set_enabled(self.compile)
        
        n = self.n
        decode = Compiler.compile(imdecode)
        def code(source, dest):
            for _ in range(n):
                decode(source, dest,
                           new_height,
                           new_width,
                           new_height,
                           new_width,
                           0, 0, 1, 1, False)
                
        self.code = Compiler.compile(code)
        
    def run(self):
        self.code(self.encoded_image, self.destination)
        
    def __exit__(self, *args):
        pass
