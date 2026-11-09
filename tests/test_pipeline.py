"""End-to-end smoke test: make-data -> train -> predict -> aggregate on tiny meshes."""
import importlib.util

import numpy as np
import pytest

HAVE_DEPS = all(importlib.util.find_spec(m) is not None
                for m in ("toriccnn.raster", "toriccnn.lift", "toriccnn.nn"))


def _make_mesh(path, seed):
    import trimesh

    m = trimesh.creation.icosphere(subdivisions=3)
    V = np.asarray(m.vertices, dtype=np.float64)
    rng = np.random.default_rng(seed)
    # smooth radial noise: a few low-frequency bumps
    dirs = rng.normal(size=(4, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    r = 1.0 + 0.15 * np.exp(4 * (V @ dirs.T - 1)).sum(1) + 0.03 * rng.normal() * V[:, 0] * V[:, 1]
    V = V * r[:, None] * np.array([1.0, 0.8, 1.2])
    F = np.asarray(m.faces)
    trimesh.Trimesh(V, F, process=False).export(path)
    c = V[F].mean(1)
    labels = 1 + (c[:, 0] > 0) + 2 * (c[:, 1] > 0) + 4 * (c[:, 2] > 0)  # 1..8
    np.savetxt(path.with_suffix(".txt"), labels, fmt="%d")


@pytest.mark.skipif(not HAVE_DEPS, reason="toriccnn.raster / lift / nn not available yet")
def test_pipeline(tmp_path):
    from toriccnn import aggregate, predict, train
    from toriccnn.data import make_data

    raw = tmp_path / "raw"
    raw.mkdir()
    for i in range(2):
        _make_mesh(raw / f"m{i}.ply", seed=i)
    proc = tmp_path / "proc"
    make_data.main([str(raw), "--out", str(proc), "--num-triplets", "2", "--size", "128",
                    "--n-extremities", "3", "--n-farthest", "6"])
    trip = sorted((proc / "triplets").glob("*.npz"))
    assert len(trip) == 4
    with np.load(trip[0], allow_pickle=False) as d:
        assert d["data"].shape == (128, 128, 26) and d["data"].dtype == np.uint8
        assert d["seg"].shape == (128, 128) and d["seg"].dtype == np.int16
        assert d["seg"].min() >= 0 and d["seg"].max() <= 7
    assert len(list((proc / "meshes").glob("*.npz"))) == 2

    exp = tmp_path / "exp"
    args = ["--data", str(proc), "--exp-dir", str(exp), "--epochs", "1", "--batch-size", "2",
            "--sub-batches", "2", "--no-pretrained", "--workers", "0", "--val-frac", "0.25"]
    train.main(args)
    assert (exp / "net-epoch-1.pt").exists() and (exp / "log.csv").exists()
    train.main(args[:5] + ["2"] + args[6:])  # resume to epoch 2
    assert (exp / "net-epoch-2.pt").exists()

    preds = tmp_path / "preds"
    predict.main(["--exp-dir", str(exp), "--data", str(proc), "--out", str(preds), "--plots"])
    pf = sorted(preds.glob("pred_*.npz"))
    assert len(pf) == 4
    with np.load(pf[0], allow_pickle=False) as d:
        n = len(d["V"])
        assert d["scores_on_mesh"].shape == (n, 8)
        assert d["scale"].shape == (n,) and d["gt_on_mesh"].shape == (n,)

    res = aggregate.aggregate(preds, tmp_path / "agg", meshes=proc, plots=True)
    assert len(res) == 2
    for r in res.values():
        assert 0.0 <= r["vertex_accuracy"] <= 1.0
        assert 0.0 <= r["face_accuracy"] <= 1.0
