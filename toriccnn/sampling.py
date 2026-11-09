"""Cone-triplet selection (``getPointTripletsByFarthestPointSampling.m``)."""
from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np

from .mesh.ops import vertex_rings


def select_triplets(F: np.ndarray, dist: np.ndarray, agd: np.ndarray, n: int = 5,
                    n_farthest: int = 20, num_triplets: int = 300, T: float = 0.4,
                    rng: np.random.Generator | int | None = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(triplets (k,3), sampled_points)``.

    ``n`` extremities are AGD local maxima (over the 2-ring) separated by at
    least ``T * diameter``; farthest point sampling then extends them to
    ``n_farthest`` points. Triplets are all C(n,3) extremity triplets plus random
    triplets of the FPS points, ``num_triplets`` in total.
    """
    rng = np.random.default_rng(rng)
    nv = len(agd)
    diameter = dist[np.isfinite(dist)].max()

    if n > 3:
        rings = vertex_rings(F, nv)
        is_max = np.zeros(nv, dtype=bool)
        for i in range(nv):
            nb = np.unique(np.concatenate([rings[i]] + [rings[j] for j in rings[i]]))
            nb = nb[nb != i]
            is_max[i] = np.all(agd[i] >= agd[nb])
        max_idx = np.flatnonzero(is_max)
        max_idx = max_idx[np.argsort(-agd[max_idx], kind="stable")]
        init: list[int] = []
        for c in max_idx:
            if init and dist[init, c].min() < T * diameter:
                continue
            init.append(int(c))
            if len(init) == n:
                break
        if len(init) < n:
            raise RuntimeError(f"only {len(init)} separated AGD maxima found (need {n}); lower T or n")
    else:
        init = [int(np.argmax(dist[0]))]

    pts = list(init)
    while len(pts) < n_farthest:
        pts.append(int(np.argmax(dist[pts].min(0))))
    pts_arr = np.asarray(pts)

    fps_triplets = np.array(list(combinations(pts_arr, 3)))
    if n > 3:
        base = np.array(list(combinations(init, 3)))
        n_extra = num_triplets - len(base)
    else:
        base = np.zeros((0, 3), dtype=np.int64)
        n_extra = num_triplets
    n_extra = max(0, min(n_extra, comb(len(pts_arr), 3)))
    extra = fps_triplets[rng.choice(len(fps_triplets), n_extra, replace=False)]
    return np.concatenate([base, extra]).astype(np.int64), pts_arr
