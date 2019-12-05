# GGNN: Graph-based GPU Nearest Neighbor Search
Fabian Groh, Lukas Ruppert, Patrick Wieschollek, Hendrik P.A. Lensch

Abstract
-------------------

Approximate nearest neighbor (ANN) search in high dimensions is an integral part of several computer vision systems and gains importance in deep learning with explicit memory representations. Since PQT and FAISS started to leverage the massive parallelism offered by GPUs GPU-based implementations are a crucial resource for today’s state-of-the-art ANN methods. While most of these methods allow for faster queries, less emphasis is devoted to accelerate the construction of the underlying index structures. In this paper, we propose a novel search structure based on nearest neighbor graphs and information propagation on graphs. Our method is designed to take advantage of GPU architectures to accelerate the hierarchical building of the index structure and for performing the query. Empirical evaluation shows that GGNN significantly surpasses the state-of-the-art GPU- and CPU based systems in terms of build-time, accuracy and search speed.

Code
-------------------

Code will be available upon publication.
If you have any questions please feel free to contact us.


More Resources
-------------------

- [Arxiv Pre-Print](https://arxiv.org/abs/1912.01059)