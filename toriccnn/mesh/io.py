"""Mesh loading and normalisation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from .ops import face_areas


def load_mesh(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a triangle mesh (.ply, .off, .obj, ...) without any reordering.

    Returns ``V`` (n, 3) float64 and ``F`` (m, 3) int64 (0-based). Vertices are
    not merged or reordered so per-face label files stay aligned.
    """
    mesh = trimesh.load(str(path), process=False, force="mesh")
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int64)
    return V, F


def load_off(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Minimal OFF reader (used for the bundled sphere asset)."""
    with open(path) as fh:
        tokens = [t for line in fh for t in line.split("#")[0].split()]
    assert tokens[0] == "OFF", "not an OFF file"
    nv, nf = int(tokens[1]), int(tokens[2])
    pos = 4
    V = np.array(tokens[pos:pos + 3 * nv], dtype=np.float64).reshape(nv, 3)
    pos += 3 * nv
    F = []
    for _ in range(nf):
        k = int(tokens[pos])
        F.append([int(t) for t in tokens[pos + 1:pos + 1 + k]])
        pos += 1 + k
    return V, np.asarray(F, dtype=np.int64)


def normalize_area(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Scale the mesh to unit total surface area (``V / sqrt(area)``)."""
    return V / np.sqrt(face_areas(V, F).sum())
