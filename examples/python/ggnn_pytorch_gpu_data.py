#! /usr/bin/python3

import pyggnn as ggnn
import torch

# get detailed logs
ggnn.set_log_level(4)


# create data
base = torch.rand((10_000, 128), dtype=torch.float32, device='cuda')
query = torch.rand((10_000, 128), dtype=torch.float32, device='cuda')


# initialize GGNN
my_ggnn = ggnn.GGNN()
my_ggnn.set_base(base)
my_ggnn.set_return_results_on_gpu(True)

# choose a distance measure
measure=ggnn.DistanceMeasure.Euclidean

# build the graph
my_ggnn.build(k_build=24, tau_build=0.5, measure=measure)

# run query
k_query: int = 10
tau_query: float = 0.64
max_iterations: int = 400

indices, dists = my_ggnn.query(query, k_query, tau_query, max_iterations, measure)

print('indices:', indices[:5], '\n squared dists:',  dists[:5], '\n')
