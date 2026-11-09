"""Cut a sphere-type mesh into a disk along a tree of shortest paths between cones.

Port of ``TreeCutter.m`` specialised to the 3-cone tree used by the square
orbifold: root = cone 3, children = cones 1 and 2. Each cut duplicates the
interior path vertices; afterwards the root (cone 3) is duplicated once per
wedge of its 1-ring.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

from ..mesh.ops import boundary_loop, unique_edges


@dataclass
class CutMesh:
    V: np.ndarray                    # (n_cut, 3) vertex positions
    T: np.ndarray                    # (m, 3) faces in cut indices
    path_pairs: list[np.ndarray]     # each (k, 2): corresponding vertices on the two sides of a seam
    cut_to_uncut: np.ndarray         # (n_cut,)
    uncut_to_cut: list[list[int]] = field(default_factory=list)

    def first_copy(self) -> np.ndarray:
        """For every uncut vertex, one of its cut copies."""
        return np.array([c[0] for c in self.uncut_to_cut], dtype=np.int64)


class TreeCutter:
    def __init__(self, V: np.ndarray, F: np.ndarray):
        self.V = np.asarray(V, dtype=np.float64).copy()
        self.T = np.asarray(F, dtype=np.int64).copy()
        self.n_orig = len(V)
        self.cut_to_uncut = list(range(len(V)))
        self.uncut_to_cut = [[i] for i in range(len(V))]
        self.path_pairs: list[np.ndarray] = []

    # ------------------------------------------------------------------ utils
    def _add_vertex_copy(self, v: int) -> int:
        new = len(self.V)
        self.V = np.vstack([self.V, self.V[v]])
        orig = self.cut_to_uncut[v]
        self.cut_to_uncut.append(orig)
        self.uncut_to_cut[orig].append(new)
        return new

    def _shortest_path(self, source: int, target: int) -> np.ndarray:
        n = len(self.V)
        E = unique_edges(self.T)
        d = np.linalg.norm(self.V[E[:, 0]] - self.V[E[:, 1]], axis=1)
        loop = boundary_loop(self.T) if len(self.V) > self.n_orig else np.zeros(0, dtype=np.int64)
        banned = np.setdiff1d(loop, [source, target])
        keep = ~(np.isin(E[:, 0], banned) | np.isin(E[:, 1], banned))
        E, d = E[keep], d[keep]
        A = sp.coo_matrix((np.concatenate([d, d]), (np.concatenate([E[:, 0], E[:, 1]]),
                                                    np.concatenate([E[:, 1], E[:, 0]]))), shape=(n, n)).tocsr()
        _, pred = dijkstra(A, directed=False, indices=source, return_predecessors=True)
        if pred[target] < 0:
            raise RuntimeError(f"no path between cones {source} and {target}")
        path = [target]
        while path[-1] != source:
            path.append(int(pred[path[-1]]))
        return np.asarray(path[::-1], dtype=np.int64)

    # -------------------------------------------------------------- cutting
    def split_mesh_by_path(self, p: np.ndarray) -> np.ndarray:
        """Cut along path ``p`` (duplicating interior vertices). Returns (len(p)-1, 2) pairs."""
        if len(p) < 3:
            raise RuntimeError("cones are adjacent; choose cones further apart")
        T = self.T
        left, right = set(), set()
        for a, b in zip(p[:-1], p[1:]):
            tris = np.flatnonzero(np.isin(T, [a, b]).sum(1) == 2)
            assert len(tris) == 2, "path edge is not interior"
            for t in tris:
                row = list(T[t])
                ia, ib = row.index(a), row.index(b)
                (left if (ib - ia) % 3 == 1 else right).add(int(t))

        # triangles touching interior path vertices get assigned to a side by flooding
        cand = set(np.flatnonzero(np.isin(T, p[1:-1]).any(1)).tolist()) - left - right

        def edges(t):
            a, b, c = T[t]
            return {tuple(sorted((a, b))), tuple(sorted((b, c))), tuple(sorted((c, a)))}

        edge_cache = {t: edges(t) for t in cand | left | right}
        left_edges = set().union(*(edge_cache[t] for t in left))
        right_edges = set().union(*(edge_cache[t] for t in right))
        while cand:
            progress = False
            for t in list(cand):
                e = edge_cache[t]
                if e & left_edges:
                    left.add(t)
                    left_edges |= e
                elif e & right_edges:
                    right.add(t)
                    right_edges |= e
                else:
                    continue
                cand.discard(t)
                progress = True
            if not progress:
                raise RuntimeError("could not assign all triangles to a side of the cut")

        left_idx = np.fromiter(left, dtype=np.int64)
        pairs = []
        for v in p[1:-1]:
            new = self._add_vertex_copy(int(v))
            tl = self.T[left_idx]
            tl[tl == v] = new
            self.T[left_idx] = tl
            pairs.append((int(v), new))
        pairs.append((int(p[-1]), int(p[-1])))
        return np.asarray(pairs, dtype=np.int64)

    def split_center_node(self, center: int, star_pairs: list[np.ndarray]) -> list[np.ndarray]:
        inds = list(np.flatnonzero((self.T == center).any(1)))
        groups = []
        while inds:
            group = [inds.pop(0)]
            while True:
                vs = set(np.unique(self.T[group]).tolist()) - {center}
                new = [t for t in inds if vs & set(self.T[t].tolist())]
                if not new:
                    break
                group += new
                inds = [t for t in inds if t not in new]
            groups.append(np.asarray(sorted(group)))

        group_centers = []
        for gi, g in enumerate(groups):
            c = center if gi == 0 else self._add_vertex_copy(center)
            tg = self.T[g]
            tg[tg == center] = c
            self.T[g] = tg
            group_centers.append(c)

        out = []
        for pair in star_pairs:
            centers = [None, None]
            for k in range(2):
                for g, c in zip(groups, group_centers):
                    if np.isin(pair[:, k], self.T[g]).any():
                        if centers[k] is not None:
                            raise RuntimeError("seam touches several wedges of the root cone")
                        centers[k] = c
            if None in centers:
                raise RuntimeError("could not attach seam to a wedge of the root cone")
            out.append(np.vstack([np.asarray(centers, dtype=np.int64)[None], pair]))
        return out

    def cut(self, cones: np.ndarray) -> CutMesh:
        root, sons = int(cones[2]), [int(cones[0]), int(cones[1])]
        star = []
        for s in sons:
            path = self._shortest_path(root, s)
            star.append(self.split_mesh_by_path(path))
        self.path_pairs += self.split_center_node(root, star)
        return CutMesh(self.V, self.T, self.path_pairs, np.asarray(self.cut_to_uncut, dtype=np.int64),
                       self.uncut_to_cut)


def cut_mesh(V: np.ndarray, F: np.ndarray, cones) -> CutMesh:
    return TreeCutter(V, F).cut(np.asarray(cones))
