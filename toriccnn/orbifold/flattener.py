"""Seamless flattening of a sphere-type mesh to the square (4,4,2) Euclidean orbifold.

Port of ``flatten_sphere.m`` / ``Flattener.m`` (Aigerman & Lipman 2015) for
orbifold type I. Cones 1 and 2 are fixed at (-1,-1) and (1,1); each seam pair
is related by a 90 degree rotation about its cone. The resulting flat mesh
tiles the plane; the tiling has translational period 4 in x and y, so a
4x4 window (e.g. [-3,1]^2) is one flat torus (see ``raster.py``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..mesh.ops import boundary_loop, face_areas
from .cutter import CutMesh, cut_mesh

CONE_POSITIONS = np.array([[-1.0, -1.0], [1.0, 1.0]])
CONE_ANGLE = 4  # cone of angle 2*pi/4 at cones 1 and 2


def cotmatrix(V: np.ndarray, F: np.ndarray) -> sp.csr_matrix:
    """gptoolbox ``cotmatrix``: off-diagonals 0.5*cot (positive), rows sum to 0."""
    n = len(V)
    v1 = V[F[:, 2]] - V[F[:, 1]]
    v2 = V[F[:, 0]] - V[F[:, 2]]
    v3 = V[F[:, 1]] - V[F[:, 0]]
    dblA = np.linalg.norm(np.cross(v1, v2), axis=1) if V.shape[1] == 3 else np.abs(np.cross(v1, v2))
    cot12 = -(v1 * v2).sum(1) / dblA / 2
    cot23 = -(v2 * v3).sum(1) / dblA / 2
    cot31 = -(v3 * v1).sum(1) / dblA / 2
    i1, i2, i3 = F[:, 0], F[:, 1], F[:, 2]
    I = np.concatenate([i1, i2, i2, i3, i3, i1, i1, i2, i3])
    J = np.concatenate([i2, i1, i3, i2, i1, i3, i1, i2, i3])
    Vals = np.concatenate([cot12, cot12, cot23, cot23, cot31, cot31,
                           -cot12 - cot31, -cot12 - cot23, -cot31 - cot23])
    return sp.coo_matrix((Vals, (I, J)), shape=(n, n)).tocsr()


class _Constraints:
    """Sparse linear equality constraints A x = b on interleaved (x0,y0,x1,y1,...)."""

    def __init__(self, n_vars: int):
        self.n = 2 * n_vars
        self.rows, self.cols, self.vals, self.b = [], [], [], []

    def _row(self, entries, rhs):
        r = len(self.b)
        for c, v in entries:
            self.rows.append(r)
            self.cols.append(c)
            self.vals.append(v)
        self.b.append(rhs)

    def fix(self, i: int, pos):
        self._row([(2 * i, 1.0)], pos[0])
        self._row([(2 * i + 1, 1.0)], pos[1])

    def rotation(self, s: np.ndarray, t: np.ndarray, R: np.ndarray):
        """R (x_s[k] - x_s[0]) - (x_t[k] - x_t[0]) = 0 for k >= 1."""
        for k in range(1, len(s)):
            for y in range(2):
                self._row([(2 * s[k], R[y, 0]), (2 * s[k] + 1, R[y, 1]),
                           (2 * s[0], -R[y, 0]), (2 * s[0] + 1, -R[y, 1]),
                           (2 * t[k] + y, -1.0), (2 * t[0] + y, 1.0)], 0.0)

    def matrices(self):
        A = sp.coo_matrix((self.vals, (self.rows, self.cols)), shape=(len(self.b), self.n)).tocsr()
        return A, np.asarray(self.b)


@dataclass
class Flattening:
    cut: CutMesh
    V_flat: np.ndarray        # (n_cut, 2)
    cones: np.ndarray         # (3,) uncut vertex indices

    @property
    def T(self) -> np.ndarray:
        return self.cut.T

    def face_dets(self) -> np.ndarray:
        """Area scale (det of the differential) per face, signed."""
        Vf, T = self.V_flat, self.cut.T
        e1, e2 = Vf[T[:, 1]] - Vf[T[:, 0]], Vf[T[:, 2]] - Vf[T[:, 0]]
        flat = 0.5 * (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
        return flat / face_areas(self.cut.V, T)

    def vertex_scale(self) -> np.ndarray:
        """Area-weighted average of face dets around each *uncut* vertex."""
        T = self.cut.T
        areas = face_areas(self.cut.V, T)
        c = areas * self.face_dets()
        n_unc = len(self.cut.uncut_to_cut)
        num, den = np.zeros(n_unc), np.zeros(n_unc)
        Tu = self.cut.cut_to_uncut[T]
        for j in range(3):
            np.add.at(num, Tu[:, j], c)
            np.add.at(den, Tu[:, j], areas)
        return num / den

    def to_arrays(self) -> dict:
        lens = np.array([len(p) for p in self.cut.path_pairs], dtype=np.int64)
        return dict(V_flat=self.V_flat, T_cut=self.cut.T, V_cut=self.cut.V,
                    cut_to_uncut=self.cut.cut_to_uncut, cones=self.cones,
                    path_pairs=np.concatenate(self.cut.path_pairs), path_lens=lens)

    @classmethod
    def from_arrays(cls, d) -> "Flattening":
        cut_to_uncut = np.asarray(d["cut_to_uncut"])
        n_unc = int(cut_to_uncut.max()) + 1
        uncut_to_cut: list[list[int]] = [[] for _ in range(n_unc)]
        for c, u in enumerate(cut_to_uncut):
            uncut_to_cut[u].append(c)
        pp = np.split(np.asarray(d["path_pairs"]), np.cumsum(d["path_lens"])[:-1])
        cut = CutMesh(np.asarray(d["V_cut"]), np.asarray(d["T_cut"]), pp, cut_to_uncut, uncut_to_cut)
        return cls(cut, np.asarray(d["V_flat"]), np.asarray(d["cones"]))


def flatten_sphere(V: np.ndarray, F: np.ndarray, cones, verbose: bool = False) -> Flattening:
    """Square-orbifold flattening with cones ``(c1, c2, c3)`` (uncut vertex indices)."""
    cones = np.asarray(cones, dtype=np.int64)
    assert len(cones) == 3 and len(set(cones.tolist())) == 3
    cut = cut_mesh(V, F, cones)
    n = len(cut.V)

    loop = boundary_loop(cut.T)
    ends = np.unique(np.concatenate([p[[0, -1]].ravel() for p in cut.path_pairs]))
    cons = _Constraints(n)
    angs: dict[int, int | None] = {}
    for v in loop[np.isin(loop, ends)]:
        which = np.flatnonzero(cones == cut.cut_to_uncut[v])[0]
        if which <= 1:
            cons.fix(int(v), CONE_POSITIONS[which])
            angs[int(v)] = CONE_ANGLE
        else:
            angs[int(v)] = None

    for pair in cut.path_pairs:
        s, t = pair[:, 0], pair[:, 1]
        sign = -1
        if s[-1] == t[-1]:
            s, t, sign = s[::-1], t[::-1], 1
        ang = angs.get(int(s[0])) or 1
        ang = sign * ang
        th = 2 * np.pi / ang
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        cons.rotation(s, t, R)
    A, b = cons.matrices()

    L = cotmatrix(cut.V, cut.T).tolil()
    off = sp.triu(L, 1)
    if off.min() < 0:
        if verbose:
            print("mesh is not Delaunay; clamping negative cotan weights")
        L = L.tocsr()
        L.data[L.data < 0] = 1e-2
        L.setdiag(0)
        L.eliminate_zeros()
        L.setdiag(-np.asarray(L.sum(axis=0)).ravel())
    L = sp.csr_matrix(L)
    L2 = sp.kron(L, sp.identity(2), format="csr")  # interleaved x/y

    K = sp.bmat([[L2, A.T], [A, None]], format="csc")
    rhs = np.concatenate([np.zeros(2 * n), b])
    sol = spla.spsolve(K, rhs)
    res = np.abs(K @ sol - rhs).max()
    if not np.isfinite(res) or res > 1e-6:
        raise RuntimeError(f"orbifold linear system not solved (residual {res:.2e})")
    V_flat = sol[:2 * n].reshape(n, 2)
    return Flattening(cut, V_flat, cones)
