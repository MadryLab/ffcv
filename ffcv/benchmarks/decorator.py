from itertools import product
from time import time
from collections import defaultdict
import pathlib

import numpy as np
from tqdm import tqdm

from .benchmark import Benchmark

ALL_SUITES = {}


def benchmark(arg_values={}):
    args_list = product(*arg_values.values())
    runs = [dict(zip(arg_values.keys(), x)) for x in args_list]
    def wrapper(cls):
        ALL_SUITES[cls.__name__] = (cls, runs)
    
    return wrapper

def run_all(runs=3, warm_up=1, pattern='*'):
    results = defaultdict(list)

    selected_suites = {}
    for sname in  ALL_SUITES.keys():
        if pathlib.PurePath(sname).match(pattern):
            selected_suites[sname] = ALL_SUITES[sname]

    it_suite = tqdm(selected_suites.items(), desc='Suite', leave=False)

    for suite_name, (cls, args_list) in it_suite:
        it_suite.set_postfix({'name': suite_name})
        it_args = tqdm(args_list, desc='configuration', leave=False)

        for args in it_args:
            benchmark: Benchmark = cls(**args)
            with benchmark:
                for _ in range(warm_up):
                    benchmark.run()
                    
                timings = []
                for _ in range(runs):
                    start = time()
                    benchmark.run()
                    timings.append(time() - start)
                
            median_time = np.median(timings)
            
            throughput = None
            
            if 'n' in args:
                throughput = args['n'] / median_time
             
            results[suite_name].append({
                **args,
                'time': median_time,
                'throughput': throughput
            })
        it_args.close()
    it_suite.close()
    return results