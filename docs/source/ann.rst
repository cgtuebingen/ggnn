Approximate Nearest Neighbor Search
===================================

Approximate nearest neighbor (ANN) search is of great importance in various fields including databases, data mining, and machine learning. ANN is derived from *k*-nearest-neighbor search.
In contrast to kNN methods, ANN methods deliver approximate results,
but typically allow for much faster queries.

.. _search graph parameters:

Search Graph Parameters
-----------------------

In order to build a search graph for ANN search,
GGNN requires two parameters:

``k_build: int``
  The number of outgoing edges per point in the dataset, typically :math:`k_{build} \in [20,96]`.
  The maximum number of edges per point is 512 and the minimum is 2.
  Higher values increase construction time and memory consumption
  but typically improve query performance.
``tau_build: float``
  A cost factor, typically :math:`\tau_{build} \in [0.3,1.0]`
  Higher values increase construction time
  but typically improve query performance.


.. _query parameters:

Query Parameters
----------------

``k_query: int``
  The number of nearest neighbors to search for. Typically, :math:`10-100`.
  Technically, up to :math:`6000` neighbors can be searched for before reaching the shared memory limit.

.. note::
  Due to the design of the stopping criterion,
  it is advisable to always search for at least 10 nearest neighbors,
  even when fewer results are required.

``tau_query: float``
  A cost factor, typically :math:`\tau_{query} \in [0.7,2.0]`
  Higher values increase query time but produce more accurate results.
  There are diminishing returns in query accuracy when increasing this value.

``max_iterations: int``
  A hard limit of search iterations to perform per query.
  Each iteration visits one point in the search graph.
  Typically, :math:`200-2000` iterations are approximate.

.. note::
  If increasing ``tau_query`` and ``max_iterations`` does not yield sufficient accuracy,
  try increasing ``k_build`` and ``tau_build`` during search graph construction.

.. caution::
   Increasing ``k_query`` and ``max_iterations`` increases the shared memory consumption
   and may limit on-GPU parallelism (SM occupancy).
   Increasing the query parameters too much can slow down the query significantly.


.. _distance measures:

Distance Measures
-----------------

GGNN supports ``Euclidean`` (L2) and ``Cosine`` distance measures.
The ``measure`` can be specified during search graph construction and query and will default to ``Euclidean``.
