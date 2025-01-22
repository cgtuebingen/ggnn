#! /usr/bin/python3

import ggnn
import torch

# create data
base = torch.rand((100_000, 128), dtype=torch.float32, device='cpu')
query = torch.rand((10_000, 128), dtype=torch.float32, device='cpu')

# initialize GGNN and prepare multi-GPU
my_ggnn = ggnn.GGNN()
my_ggnn.set_base(base)
my_ggnn.set_shard_size(n_shard=25_000)
my_ggnn.set_gpus(gpu_ids=[0,1])

# build the graph
my_ggnn.build(k_build=24, tau_build=0.5)

# run query
k_query: int = 10

indices, dists = my_ggnn.query(query, k_query=k_query, tau_query=0.64, max_iterations=400)

print('indices:', indices[:5], '\n squared dists:',  dists[:5], '\n')
