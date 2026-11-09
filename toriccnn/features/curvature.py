"""Per-vertex principal curvatures (Rusinkiewicz 2004), vectorised numpy port of
``dataGeneration/code/curvature/*.m`` (Itzik Ben Shabat's implementation).
"""
from __future__ import annotations

import numpy as np


def _normr(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.where(n == 0, 1.0, n)


def _rdot(a, b):
    return (a * b).sum(-1)


def _rotate_coord_system(up, vp, nf):
    """Rotate frame (up, vp) so that its normal aligns with nf (row-wise)."""
    np_ = _normr(np.cross(up, vp))
    ndot = _rdot(nf, np_)
    perp = nf - ndot[:, None] * np_
    dperp = (np_ + nf) / (1.0 + ndot)[:, None]
    new_u = up - dperp * _rdot(perp, up)[:, None]
    new_v = vp - dperp * _rdot(perp, vp)[:, None]
    flip = ndot <= -1
    new_u[flip] = -up[flip]
    new_v[flip] = -vp[flip]
    return new_u, new_v


def principal_curvatures(V: np.ndarray, F: np.ndarray):
    """Return (k1, k2) with |k1| >= |k2| per vertex."""
    nv = len(V)
    e0 = V[F[:, 2]] - V[F[:, 1]]
    e1 = V[F[:, 0]] - V[F[:, 2]]
    e2 = V[F[:, 1]] - V[F[:, 0]]
    FN = _normr(np.cross(e0, e1))
    e0n, e1n, e2n = _normr(e0), _normr(e1), _normr(e2)
    de0, de1, de2 = (np.linalg.norm(e, axis=1) for e in (e0, e1, e2))
    l2 = np.stack([de0 ** 2, de1 ** 2, de2 ** 2], 1)
    ew = np.stack([l2[:, 0] * (l2[:, 1] + l2[:, 2] - l2[:, 0]),
                   l2[:, 1] * (l2[:, 2] + l2[:, 0] - l2[:, 1]),
                   l2[:, 2] * (l2[:, 0] + l2[:, 1] - l2[:, 2])], 1)
    s = (de0 + de1 + de2) / 2
    Af = np.sqrt(np.maximum(s * (s - de0) * (s - de1) * (s - de2), 0))

    # vertex normals weighted by area / (product of squared adjacent edge lengths)
    w = np.stack([Af / (l2[:, 1] * l2[:, 2]), Af / (l2[:, 0] * l2[:, 2]), Af / (l2[:, 1] * l2[:, 0])], 1)
    VN = np.zeros((nv, 3))
    for j in range(3):
        np.add.at(VN, F[:, j], w[:, j:j + 1] * FN)
    VN = _normr(VN)

    # Voronoi corner areas
    dot = lambda a, b: _rdot(a, b)  # noqa: E731
    with np.errstate(divide="ignore", invalid="ignore"):
        c0 = np.stack([Af * 0, -0.25 * l2[:, 2] * Af / dot(e0, e2), -0.25 * l2[:, 1] * Af / dot(e0, e1)], 1)
        c0[:, 0] = Af - c0[:, 1] - c0[:, 2]
        c1 = np.stack([-0.25 * l2[:, 2] * Af / dot(e1, e2), Af * 0, -0.25 * l2[:, 0] * Af / dot(e1, e0)], 1)
        c1[:, 1] = Af - c1[:, 0] - c1[:, 2]
        c2 = np.stack([-0.25 * l2[:, 1] * Af / dot(e2, e1), -0.25 * l2[:, 0] * Af / dot(e2, e0), Af * 0], 1)
        c2[:, 2] = Af - c2[:, 0] - c2[:, 1]
        ews = 0.5 * Af / ew.sum(1)
        c3 = np.stack([ews * (ew[:, 1] + ew[:, 2]), ews * (ew[:, 0] + ew[:, 2]), ews * (ew[:, 1] + ew[:, 0])], 1)
    Acorner = np.where((ew[:, 0] <= 0)[:, None], c0,
                       np.where((ew[:, 1] <= 0)[:, None], c1,
                                np.where((ew[:, 2] <= 0)[:, None], c2, c3)))
    Avertex = np.zeros(nv)
    for j in range(3):
        np.add.at(Avertex, F[:, j], Acorner[:, j])

    # per-vertex tangent frame (last face in loop order wins, as in MATLAB)
    up = np.zeros((nv, 3))
    up[F.ravel()] = np.stack([e2n, e0n, e1n], 1).reshape(-1, 3)
    up = _normr(np.cross(up, VN))
    vp = np.cross(VN, up)

    # per-face second fundamental form by least squares
    t = e0n
    B = _normr(np.cross(FN, t))
    n0, n1, n2 = VN[F[:, 0]], VN[F[:, 1]], VN[F[:, 2]]
    m = len(F)
    A = np.zeros((m, 6, 3))
    b = np.zeros((m, 6))
    for r, (e, dn) in enumerate(((e0, n2 - n1), (e1, n0 - n2), (e2, n1 - n0))):
        et, eb = dot(e, t), dot(e, B)
        A[:, 2 * r, 0], A[:, 2 * r, 1] = et, eb
        A[:, 2 * r + 1, 1], A[:, 2 * r + 1, 2] = et, eb
        b[:, 2 * r], b[:, 2 * r + 1] = dot(dn, t), dot(dn, B)
    AtA = np.einsum("mij,mik->mjk", A, A)
    Atb = np.einsum("mij,mi->mj", A, b)
    x = np.linalg.solve(AtA, Atb[..., None])[..., 0]

    # project face tensors onto vertex frames and accumulate with corner weights
    ku = np.zeros(nv)
    kuv = np.zeros(nv)
    kv = np.zeros(nv)
    for j in range(3):
        vi = F[:, j]
        wfp = Acorner[:, j] / Avertex[vi]
        ru, rv = _rotate_coord_system(up[vi], vp[vi], FN)
        u1, v1 = dot(ru, t), dot(ru, B)
        u2, v2 = dot(rv, t), dot(rv, B)
        a, bb, c = x[:, 0], x[:, 1], x[:, 2]
        np.add.at(ku, vi, wfp * (a * u1 * u1 + 2 * bb * u1 * v1 + c * v1 * v1))
        np.add.at(kuv, vi, wfp * (a * u1 * u2 + bb * (u1 * v2 + u2 * v1) + c * v1 * v2))
        np.add.at(kv, vi, wfp * (a * u2 * u2 + 2 * bb * u2 * v2 + c * v2 * v2))

    # Jacobi rotation diagonalisation
    tt = np.zeros(nv)
    nz = kuv != 0
    h = 0.5 * (kv[nz] - ku[nz]) / kuv[nz]
    # == 1/(h - sqrt(1+h^2)) if h < 0 else 1/(h + sqrt(1+h^2)), without evaluating the unstable branch
    tt[nz] = 1.0 / (h + np.where(h < 0, -1.0, 1.0) * np.sqrt(1 + h * h))
    k1 = ku - tt * kuv
    k2 = kv + tt * kuv
    swap = np.abs(k1) < np.abs(k2)
    k1, k2 = np.where(swap, k2, k1), np.where(swap, k1, k2)
    return k1, k2


def curvature_features(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """(n, 4) features in the original channel order.

    MATLAB ``getFunctionsOnMesh`` appends ``[Cmin Cmax Cmean Cgauss]`` where,
    because of the output order of ``calcCurvature``/``getCurvatureFeatures``,
    Cmin = k2 (smaller |k|), Cmax = k1 (larger |k|), and the "Cgauss" slot
    actually holds sign(K)*sqrt(|K|).
    """
    k1, k2 = principal_curvatures(V, F)
    K = k1 * k2
    return np.stack([k2, k1, 0.5 * (k1 + k2), np.sign(K) * np.sqrt(np.abs(K))], 1)
