import numpy as np
import pytest
import trimesh

from toriccnn.lift import SPHERE_PATH, sphere_cones
from toriccnn.mesh.io import load_off, normalize_area
from toriccnn.orbifold import flatten_sphere


def bumpy_icosphere(subdivisions=3, amp=0.15):
    m = trimesh.creation.icosphere(subdivisions=subdivisions)
    V, F = np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces, dtype=np.int64)
    V = V * (1 + amp * np.sin(3 * V[:, [0]]) * np.cos(2 * V[:, [1]]))
    return normalize_area(V, F), F


def _flatten(V, F):
    cones = sphere_cones(V / np.linalg.norm(V, axis=1, keepdims=True))
    return flatten_sphere(V, F, cones)


@pytest.fixture(scope="session")
def sphere_mesh():
    V, F = load_off(SPHERE_PATH)
    return normalize_area(V, F), F


@pytest.fixture(scope="session")
def bumpy_mesh():
    return bumpy_icosphere(3)


@pytest.fixture(scope="session", params=["sphere", "bumpy"])
def mesh_and_flat(request, sphere_mesh, bumpy_mesh):
    V, F = sphere_mesh if request.param == "sphere" else bumpy_mesh
    return V, F, _flatten(V, F)
