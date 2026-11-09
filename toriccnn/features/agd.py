"""Average geodesic distance (graph-geodesic approximation), as in ``calcAgd.m``."""
from __future__ import annotations

import logging

import numpy as np
from scipy.sparse.csgraph import dijkstra

from ..mesh.ops import edge_length_graph, one_ring_areas, smooth_combinatorial

log = logging.getLogger(__name__)


def pairwise_graph_distances(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """All-pairs shortest path lengths along mesh edges, (n, n)."""
    n = len(V)
    if n > 20000:
        log.warning("all-pairs distances for %d vertices need %.1f GB", n, n * n * 8 / 1e9)
    return dijkstra(edge_length_graph(V, F), directed=False)


def compute_agd(V: np.ndarray, F: np.ndarray, dist: np.ndarray | None = None,
                n_smooth: int = 10) -> np.ndarray:
    if dist is None:
        dist = pairwise_graph_distances(V, F)
    agd_raw = dist @ one_ring_areas(V, F)
    return smooth_combinatorial(F, agd_raw, n_smooth)
