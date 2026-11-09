import numpy as np
import trimesh

from toriccnn.features import NUM_FUNCTIONS, functions_on_mesh, principal_curvatures
from toriccnn.mesh.ops import boundary_loop
from toriccnn.orbifold import Flattening

from conftest import bumpy_icosphere


def _rot(deg):
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def test_cut_and_cones(mesh_and_flat):
    V, F, fl = mesh_and_flat
    cut = fl.cut
    assert cut.T.shape == F.shape
    assert np.array_equal(cut.cut_to_uncut[cut.T], F)          # same face order / connectivity
    loop = boundary_loop(cut.T)                                 # raises unless a single loop
    assert len(loop) > 0 and np.isfinite(fl.V_flat).all()
    c1, c2, c3 = (cut.uncut_to_cut[c] for c in fl.cones)
    assert len(c1) == 1 and len(c2) == 1 and len(c3) == 2
    np.testing.assert_allclose(fl.V_flat[c1[0]], [-1, -1], atol=1e-10)
    np.testing.assert_allclose(fl.V_flat[c2[0]], [1, 1], atol=1e-10)
    np.testing.assert_allclose(np.sort(fl.V_flat[c3], axis=0), [[-1, -1], [1, 1]], atol=1e-8)


def test_seam_rotations(mesh_and_flat):
    _, _, fl = mesh_and_flat
    X = fl.V_flat
    assert len(fl.cut.path_pairs) == 2
    for pair in fl.cut.path_pairs:
        s, t = pair[:, 0], pair[:, 1]
        assert s[-1] == t[-1]                                   # seams meet at their son cone
        ds, dt = X[s] - X[s[-1]], X[t] - X[t[-1]]
        errs = [np.abs(dt - ds @ _rot(a).T).max() for a in (90, -90)]
        assert min(errs) < 1e-8, errs


def test_no_flips_and_area(mesh_and_flat):
    _, _, fl = mesh_and_flat
    d = fl.face_dets()
    assert (d > 0).all() or (d < 0).all()
    T, X = fl.cut.T, fl.V_flat
    e1, e2 = X[T[:, 1]] - X[T[:, 0]], X[T[:, 2]] - X[T[:, 0]]
    area = np.abs(0.5 * (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])).sum()
    assert abs(area - 4.0) < 1e-6                               # 4 copies tile the 16-area torus
    assert (fl.vertex_scale() > 0).all()


def test_arrays_round_trip(mesh_and_flat):
    _, _, fl = mesh_and_flat
    fl2 = Flattening.from_arrays(fl.to_arrays())
    np.testing.assert_array_equal(fl2.V_flat, fl.V_flat)
    np.testing.assert_array_equal(fl2.cut.T, fl.cut.T)
    assert [list(c) for c in fl2.cut.uncut_to_cut] == [sorted(c) for c in fl.cut.uncut_to_cut]
    np.testing.assert_allclose(fl2.vertex_scale(), fl.vertex_scale())


def test_functions_on_mesh_smoke():
    V, F = bumpy_icosphere(3)
    X = functions_on_mesh(V, F)
    assert X.shape == (len(V), NUM_FUNCTIONS) == (len(V), 26)
    assert np.isfinite(X).all()
    assert X.min() >= 0 and X.max() <= 255 + 1e-9


def test_principal_curvatures_sphere():
    r = 2.5
    m = trimesh.creation.icosphere(subdivisions=4, radius=r)
    k1, k2 = principal_curvatures(np.asarray(m.vertices), np.asarray(m.faces))
    for k in (k1, k2):
        assert np.abs(np.median(np.abs(k)) - 1 / r) < 0.02 / r
        assert np.abs(np.abs(k) - 1 / r).max() < 0.1 / r
