#! /usr/bin/python3

import pyggnn as ggnn

import h5py
import numpy as np

path_to_dataset = '/graphics/scratch/ruppert/sift-128-euclidean.hdf5'

# load ANN-benchmark-style HDF5 dataset
with h5py.File(path_to_dataset, 'r') as f:
  base = np.array(f['train'])
  query = np.array(f['test'])
  gt = np.array(f['neighbors'])
  # gt_dist = np.array(f['distances'])

k_query: int = 10

evaluator = ggnn.Evaluator(base, query, gt=gt, k_query=k_query)

my_ggnn = ggnn.GGNN()
my_ggnn.set_base(base)
my_ggnn.build(k_build=24, tau_build=0.5)

# 90% C@1 / R@1
indices, dists = my_ggnn.query(query, k_query, tau_query=0.34, max_iterations=200)
print(evaluator.evaluate_results(indices))
# 95% C@1 / R@1
indices, dists = my_ggnn.query(query, k_query, tau_query=0.41, max_iterations=200)
print(evaluator.evaluate_results(indices))
# 99% C@1 / R@1
indices, dists = my_ggnn.query(query, k_query, tau_query=0.51, max_iterations=200)
print(evaluator.evaluate_results(indices))
# 99% C@10
indices, dists = my_ggnn.query(query, k_query, tau_query=0.64, max_iterations=400)
print(evaluator.evaluate_results(indices))
