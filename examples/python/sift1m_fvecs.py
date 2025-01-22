#! /usr/bin/python3

import ggnn

path_to_dataset = '/graphics/scratch/datasets/ANN_datasets/SIFT1M/sift/'

base = ggnn.FloatDataset.load(path_to_dataset + 'sift_base.fvecs')
query = ggnn.FloatDataset.load(path_to_dataset + 'sift_query.fvecs')
gt = ggnn.IntDataset.load(path_to_dataset + 'sift_groundtruth.ivecs')

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
