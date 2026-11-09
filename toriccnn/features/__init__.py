"""Per-vertex input functions (``getFunctionsOnMesh.m``): 21 WKS + 4 curvature + AGD."""
from __future__ import annotations

import numpy as np

from ..mesh.io import normalize_area
from .agd import compute_agd, pairwise_graph_distances
from .curvature import curvature_features, principal_curvatures
from .wks import WKS_COLUMNS, compute_wks

NUM_FUNCTIONS = len(WKS_COLUMNS) + 4 + 1  # 26


def functions_on_mesh(V: np.ndarray, F: np.ndarray, dist: np.ndarray | None = None,
                      agd: np.ndarray | None = None) -> np.ndarray:
    """(n_vertices, 26) features, each min-max normalised to [0, 255].

    ``dist``/``agd`` may be passed in to avoid recomputing all-pairs distances
    (they must have been computed on the area-normalised mesh).
    """
    V = normalize_area(V, F)
    wks = compute_wks(V, F)[:, WKS_COLUMNS]
    curv = curvature_features(V, F)
    if agd is None:
        agd = compute_agd(V, F, dist)
    feats = np.concatenate([wks, curv, agd[:, None]], axis=1)
    lo, hi = feats.min(0), feats.max(0)
    return 255.0 * (feats - lo) / np.where(hi > lo, hi - lo, 1.0)


__all__ = [
    "functions_on_mesh",
    "compute_wks",
    "compute_agd",
    "pairwise_graph_distances",
    "curvature_features",
    "principal_curvatures",
    "NUM_FUNCTIONS",
]
