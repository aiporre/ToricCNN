"""Pull toric images back to mesh vertices, and the sphere-relative scale.

``lift_image`` replaces the MATLAB lifting of network predictions; and
``scale_on_mesh`` replaces ``getScaleDividedBySphere.m`` /
``compute_map_from_sphere_embeddings.m``: instead of locating each mesh
vertex in the sphere's tiled flattening, the sphere's per-vertex scale is
rendered into a toric image and sampled there.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from .mesh.io import load_off, normalize_area
from .orbifold import Flattening, flatten_sphere
from .raster import plane_to_pixel, rasterize_vertex_functions

SPHERE_PATH = Path(__file__).parent / "orbifold" / "assets" / "sphere.off"


def _vertex_plane_positions(flat: Flattening) -> np.ndarray:
    """Flat position of one cut copy (the first) of every uncut vertex, (n_uncut, 2)."""
    return np.asarray(flat.V_flat)[flat.cut.first_copy()]


def sample_image(img: np.ndarray, xy: np.ndarray, mode: str = "bilinear") -> np.ndarray:
    """Sample a toric image at plane points ``xy`` (n, 2), wrapping periodically."""
    H, W = img.shape[:2]
    r, c = plane_to_pixel(xy, (H, W))
    if mode == "nearest":
        ri = np.floor(r + 0.5).astype(np.int64) % H
        ci = np.floor(c + 0.5).astype(np.int64) % W
        return img[ri, ci]
    if mode != "bilinear":
        raise ValueError(f"unknown mode {mode!r}")
    r0, c0 = np.floor(r), np.floor(c)
    fr, fc = r - r0, c - c0
    r0, c0 = r0.astype(np.int64), c0.astype(np.int64)
    r1, c1 = (r0 + 1) % H, (c0 + 1) % W
    r0, c0 = r0 % H, c0 % W
    if img.ndim == 3:
        fr, fc = fr[:, None], fc[:, None]
    imgf = img.astype(np.float64, copy=False)
    return ((1 - fr) * (1 - fc) * imgf[r0, c0] + (1 - fr) * fc * imgf[r0, c1]
            + fr * (1 - fc) * imgf[r1, c0] + fr * fc * imgf[r1, c1])


def lift_image(flat: Flattening, img: np.ndarray, mode: str = "nearest") -> np.ndarray:
    """Per-uncut-vertex values of a toric image, (n_uncut,) or (n_uncut, C)."""
    return sample_image(np.asarray(img), _vertex_plane_positions(flat), mode)


def sphere_cones(V: np.ndarray) -> np.ndarray:
    a = np.array([0.0, 2 * np.pi / 3, 4 * np.pi / 3])
    ideal = np.stack([np.cos(a), np.sin(a), np.zeros(3)], 1)
    d = np.linalg.norm(V[None, :, :] - ideal[:, None, :], axis=2)
    return d.argmin(1)


@lru_cache(maxsize=1)
def sphere_reference_flattening() -> Flattening:
    """Flattening of the bundled unit-area sphere (cones on the equator)."""
    V, F = load_off(SPHERE_PATH)
    # cones are chosen on the raw sphere (as knnsearch in MATLAB, before normalising)
    cones = sphere_cones(V)
    V = normalize_area(V, F)
    return flatten_sphere(V, F, cones)


@lru_cache(maxsize=4)
def _sphere_scale_image(res: int) -> np.ndarray:
    sph = sphere_reference_flattening()
    return rasterize_vertex_functions(sph, sph.vertex_scale(), size=res)


def scale_on_mesh(flat: Flattening, res: int = 1024) -> np.ndarray:
    """Mesh vertex scale divided by the reference sphere's scale at the same plane point."""
    sphere_scale = lift_image(flat, _sphere_scale_image(res), mode="bilinear")
    return flat.vertex_scale() / sphere_scale
