import time

import numpy as np
import pytest
import torch

from toriccnn.lift import lift_image, sample_image, scale_on_mesh, sphere_reference_flattening
from toriccnn.raster import (_rasterize, plane_to_pixel, rasterize_face_labels,
                             rasterize_vertex_functions, torus_copies)

from conftest import _flatten, bumpy_icosphere

SIZE = 256


def _smooth_funcs(V):
    """Smooth vertex functions on [0, 255]."""
    f = np.stack([V[:, 0], V[:, 1] * V[:, 2], np.sin(2 * V[:, 2]) + V[:, 0] ** 2], 1)
    return 255 * (f - f.min(0)) / (f.max(0) - f.min(0))


def test_plane_to_pixel():
    r, c = plane_to_pixel(np.array([[-3.0, 1.0], [1.0, -3.0], [-1.0, -1.0]]), 4)
    np.testing.assert_allclose(c, [-0.5, 3.5, 1.5])
    np.testing.assert_allclose(r, [-0.5, 3.5, 1.5])


def test_copies_cover_torus_once(mesh_and_flat):
    _, _, fl = mesh_and_flat
    assert len(torus_copies(fl.V_flat)) == 4
    _, _, pix = _rasterize(fl, (SIZE, SIZE), torch.device("cpu"))
    cnt = np.bincount(pix.numpy(), minlength=SIZE * SIZE)
    assert (cnt == 0).mean() < 1e-3                              # (almost) no cracks
    assert (cnt == 1).mean() > 0.97                             # copies do not overlap


def test_raster_lift_round_trip(mesh_and_flat):
    V, F, fl = mesh_and_flat
    f = _smooth_funcs(V)
    img = rasterize_vertex_functions(fl, f, size=SIZE)
    assert img.shape == (SIZE, SIZE, 3) and img.dtype == np.float32 and np.isfinite(img).all()
    for mode in ("nearest", "bilinear"):
        g = lift_image(fl, img, mode=mode)
        assert g.shape == f.shape
        assert np.median(np.abs(g - f)) < 2, mode
    # every cut copy of a vertex (seams, cone 3) sees the same value: toric consistency
    for verts in fl.cut.uncut_to_cut:
        if len(verts) > 1:
            vals = [sample_image(img, fl.V_flat[[c]], "bilinear")[0] for c in verts]
            assert np.abs(np.ptp(np.array(vals), axis=0)).max() < 25


def test_raster_periodic(mesh_and_flat):
    V, _, fl = mesh_and_flat
    img = rasterize_vertex_functions(fl, _smooth_funcs(V)[:, 0], size=SIZE)
    pts = np.random.default_rng(0).uniform(-3, 1, (500, 2))
    a = sample_image(img, pts)
    for shift in ([4, 0], [0, 4], [-4, 8]):
        np.testing.assert_allclose(sample_image(img, pts + shift), a, atol=1e-3)
    # the tiling is also invariant under a 90 degree rotation about cone (1, 1) (orbifold symmetry)
    c = np.array([1.0, 1.0])
    rot = (pts - c) @ np.array([[0, -1], [1, 0]]).T + c
    assert np.median(np.abs(sample_image(img, rot) - a)) < 2


def test_face_labels_round_trip(mesh_and_flat):
    V, F, fl = mesh_and_flat
    labels = (V[F].mean(1)[:, 2] > 0).astype(np.int64) + 2 * (V[F].mean(1)[:, 0] > 0.2)
    img = rasterize_face_labels(fl, labels, size=SIZE)
    assert img.dtype == np.int64 and (img >= 0).all()
    lifted = lift_image(fl, img, mode="nearest")
    # majority label of the faces around each vertex
    n = len(V)
    votes = np.zeros((n, labels.max() + 1))
    for j in range(3):
        np.add.at(votes, (F[:, j], labels), 1)
    majority = votes.argmax(1)
    clear = votes.max(1) == votes.sum(1)                         # vertex not on a label boundary
    assert (lifted[clear] == majority[clear]).mean() > 0.98


def test_scale_on_mesh(mesh_and_flat):
    _, _, fl = mesh_and_flat
    s = scale_on_mesh(fl, res=512)
    assert s.shape == (len(fl.cut.uncut_to_cut),) and np.isfinite(s).all() and (s > 0).all()


def test_scale_on_sphere_is_one():
    s = scale_on_mesh(sphere_reference_flattening(), res=512)
    assert np.abs(s - 1).max() < 0.05


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_raster_timing(device):
    V, F = bumpy_icosphere(5)                                   # 10242 vertices
    fl = _flatten(V, F)
    vals = np.random.default_rng(0).uniform(0, 255, (len(V), 26))
    rasterize_vertex_functions(fl, vals, size=512, device=device)  # warm-up
    t = time.perf_counter()
    img = rasterize_vertex_functions(fl, vals, size=512, device=device)
    dt = time.perf_counter() - t
    print(f"\nrasterize_vertex_functions [{device}] n={len(V)} C=26 512^2: {dt:.3f}s")
    assert img.shape == (512, 512, 26)
    assert dt < 2.0
