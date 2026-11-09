"""Wave Kernel Signature (Aubry, Schlickewei, Cremers 2011).

Port of ``compute_WKS.m``. One deliberate fix: the MATLAB code computed the
squared edge lengths used for the Voronoi areas with linear indexing
(``vertices(faces(:,i2))``), i.e. only from the x coordinate. Here the true
squared edge lengths are used.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _angles_and_sqlen(V: np.ndarray, F: np.ndarray):
    angles = np.zeros(F.shape)
    sqlen = np.zeros(F.shape)
    for i in range(3):
        i1, i2, i3 = i, (i + 1) % 3, (i + 2) % 3
        pp = V[F[:, i2]] - V[F[:, i1]]
        qq = V[F[:, i3]] - V[F[:, i1]]
        pp = pp / np.maximum(np.linalg.norm(pp, axis=1, keepdims=True), np.finfo(float).eps)
        qq = qq / np.maximum(np.linalg.norm(qq, axis=1, keepdims=True), np.finfo(float).eps)
        angles[:, i1] = np.arccos(np.clip((pp * qq).sum(1), -1.0, 1.0))
        sqlen[:, i1] = ((V[F[:, i2]] - V[F[:, i3]]) ** 2).sum(1)
    return angles, sqlen


def cotan_laplacian_and_voronoi(V: np.ndarray, F: np.ndarray):
    """Positive semi-definite cotan Laplacian and normalised Voronoi areas."""
    n = len(V)
    angles, sqlen = _angles_and_sqlen(V, F)
    rows, cols, vals = [], [], []
    for i in range(3):
        i1, i2, i3 = i, (i + 1) % 3, (i + 2) % 3
        rows.append(F[:, i1])
        cols.append(F[:, i2])
        vals.append(-1.0 / np.tan(angles[:, i3]))
    L = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()
    L = 0.5 * (L + L.T)
    L = L - sp.diags(np.asarray(L.sum(axis=1)).ravel())

    # Mixed Voronoi areas (Meyer et al. 2002)
    cot = 1.0 / np.tan(angles)
    faces_area = 0.25 * (sqlen * cot).sum(1)
    obtuse = angles.max(1) > np.pi / 2
    A = np.zeros(n)
    for i in range(3):
        i1, i2, i3 = i, (i + 1) % 3, (i + 2) % 3
        voronoi = 0.125 * (cot[:, i2] * sqlen[:, i2] + cot[:, i3] * sqlen[:, i3])
        contrib = np.where(~obtuse, voronoi,
                           np.where(angles[:, i1] > np.pi / 2, faces_area / 2, faces_area / 4))
        np.add.at(A, F[:, i1], contrib)
    A = np.maximum(A, 1e-8)
    A = A / A.sum()
    return L, A


def compute_wks(V: np.ndarray, F: np.ndarray, n_eigen: int = 300, n_energies: int = 100,
                wks_variance: float = 6.0) -> np.ndarray:
    """Return the (n_vertices, n_energies) WKS matrix."""
    n = len(V)
    L, A = cotan_laplacian_and_voronoi(V, F)
    k = min(n_eigen, n - 2)
    E, PHI = spla.eigsh(L.tocsc(), k=k, M=sp.diags(A).tocsc(), sigma=-1e-5, which="LM")
    E = np.abs(np.real(E))
    order = np.argsort(E)
    E, PHI = E[order], np.real(PHI[:, order])

    log_E = np.log(np.maximum(E, 1e-6))
    e = np.linspace(log_E[1], log_E.max() / 1.02, n_energies)
    sigma = (e[1] - e[0]) * wks_variance
    G = np.exp(-((e[:, None] - log_E[None, :]) ** 2) / (2 * sigma ** 2))  # (N, k)
    WKS = (PHI ** 2) @ G.T
    WKS /= G.sum(1)[None, :]
    return WKS


# MATLAB columns [20:3:70, 90:3:100] (1-based)
WKS_COLUMNS = list(range(19, 70, 3)) + list(range(89, 100, 3))
