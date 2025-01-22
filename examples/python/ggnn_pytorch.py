#! /usr/bin/python3

import ggnn
import torch

# get detailed logs
ggnn.set_log_level(4)


# create data
base = torch.rand((10_000, 128), dtype=torch.float32, device='cpu')
query = torch.rand((10_000, 128), dtype=torch.float32, device='cpu')


# initialize ggnn
my_ggnn = ggnn.GGNN()
my_ggnn.set_base(base)

# choose a distance measure
measure=ggnn.DistanceMeasure.Euclidean

# build the graph
my_ggnn.build(k_build=24, tau_build=0.5, refinement_iterations=2, measure=measure)


# run query
k_query: int = 10
tau_query: float = 0.64
max_iterations: int = 400

indices, dists = my_ggnn.query(query, k_query, tau_query, max_iterations, measure)


# run brute-force query to get a ground truth and evaluate the results of the query
gt_indices, gt_dists = my_ggnn.bf_query(query, k_gt=k_query, measure=measure)
evaluator = ggnn.Evaluator(base, query, gt_indices, k_query=k_query)
print(evaluator.evaluate_results(indices))

# print the indices of the 10 NN of the first five queries and their squared euclidean distances 
print('indices:', indices[:5], '\n squared dists:',  dists[:5], '\n')
