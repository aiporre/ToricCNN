"""Basic discrete-geometry helpers on triangle meshes (all 0-based)."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def face_areas(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    e1 = V[F[:, 1]] - V[F[:, 0]]
    e2 = V[F[:, 2]] - V[F[:, 0]]
    if V.shape[1] == 2:
        return 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    return 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)


def one_ring_areas(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """1/3 of the area of all faces incident to each vertex."""
    a = face_areas(V, F)
    out = np.zeros(len(V))
    np.add.at(out, F.ravel(), np.repeat(a / 3.0, 3))
    return out


def unique_edges(F: np.ndarray) -> np.ndarray:
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    E = np.sort(E, axis=1)
    return np.unique(E, axis=0)


def edge_length_graph(V: np.ndarray, F: np.ndarray, n: int | None = None) -> sp.csr_matrix:
    """Symmetric sparse matrix with Euclidean edge lengths as weights."""
    n = len(V) if n is None else n
    E = unique_edges(F)
    d = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
    A = sp.coo_matrix(
        (np.concatenate([d, d]), (np.concatenate([E[:, 0], E[:, 1]]), np.concatenate([E[:, 1], E[:, 0]]))),
        shape=(n, n),
    )
    return A.tocsr()


def adjacency(F: np.ndarray, n: int | None = None) -> sp.csr_matrix:
    """Binary vertex adjacency matrix."""
    n = int(F.max()) + 1 if n is None else n
    E = unique_edges(F)
    ones = np.ones(2 * len(E))
    A = sp.coo_matrix((ones, (np.concatenate([E[:, 0], E[:, 1]]), np.concatenate([E[:, 1], E[:, 0]]))), shape=(n, n))
    return A.tocsr()


def smooth_combinatorial(F: np.ndarray, f: np.ndarray, n_iter: int = 10) -> np.ndarray:
    """Average a vertex function over its closed 1-ring ``n_iter`` times."""
    n = len(f)
    W = adjacency(F, n) + sp.identity(n, format="csr")
    W = sp.diags(1.0 / np.asarray(W.sum(axis=1)).ravel()) @ W
    for _ in range(n_iter):
        f = W @ f
    return f


def vertex_rings(F: np.ndarray, n: int | None = None) -> list[np.ndarray]:
    A = adjacency(F, n)
    return [A.indices[A.indptr[i]:A.indptr[i + 1]] for i in range(A.shape[0])]


def vertex_face_adjacency(F: np.ndarray, n: int) -> sp.csr_matrix:
    """(n_vertices, n_faces) incidence matrix."""
    rows = F.ravel()
    cols = np.repeat(np.arange(len(F)), 3)
    return sp.csr_matrix((np.ones(len(rows), dtype=bool), (rows, cols)), shape=(n, len(F)))


def boundary_loop(F: np.ndarray) -> np.ndarray:
    """Ordered vertices of the single boundary loop of a disk mesh.

    Follows directed boundary edges (consistent with face orientation).
    Raises if the boundary is not a single simple closed loop.
    """
    he = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    key = he[:, 0].astype(np.int64) * (F.max() + 1) + he[:, 1]
    rev = he[:, 1].astype(np.int64) * (F.max() + 1) + he[:, 0]
    bnd = he[~np.isin(key, rev)]
    if len(bnd) == 0:
        return np.zeros(0, dtype=np.int64)
    nxt = {}
    for a, b in bnd:
        if a in nxt:
            raise ValueError("boundary is not a simple closed loop (non-manifold boundary vertex)")
        nxt[int(a)] = int(b)
    start = int(bnd[0, 0])
    loop = [start]
    cur = nxt[start]
    while cur != start:
        loop.append(cur)
        cur = nxt[cur]
        if len(loop) > len(bnd):
            raise ValueError("boundary traversal did not close")
    if len(loop) != len(bnd):
        raise ValueError("the boundary is not a single closed loop")
    return np.asarray(loop, dtype=np.int64)
