"""Aggregate per-triplet predictions into one segmentation per mesh
(port of ``postprocessAggregateSegmentations.m`` / ``visualizeAgregatedSegmentationsFinal.m``).

Predictions are grouped by mesh (``pred_{mesh}_tri_{j}.npz``) and combined by
scale-weighted averaging: ``combined = sum_j scores_j * scale_j[:, None]``,
``prediction = argmax(combined)``. Writes ``{mesh}_combined.npz`` (+ PNG).

Accuracy: with ``--meshes`` (folder of ``meshes/*.npz`` from make-data) the
ground truth is the original per-face labels. We report
  * vertex accuracy: GT vertex label = majority label of the incident faces
    (ties -> smallest label);
  * face accuracy (area-weighted): face prediction = argmax of the combined
    scores summed over its 3 vertices.
Without ``--meshes`` the vertex GT lifted from the seg images (``gt_on_mesh``
of the first triplet) is used and only vertex accuracy is reported.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np


def plot_mesh(ax, V, F, values, cmap="jet", clim=None) -> None:
    """Draw a mesh on a 3D matplotlib axis coloured by per-vertex ``values``
    (face colour = mean of its vertex values)."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    values = np.asarray(values, dtype=np.float64)
    fv = values[F].mean(1)
    lo, hi = clim if clim is not None else (np.nanmin(fv), np.nanmax(fv))
    norm = (fv - lo) / (hi - lo if hi > lo else 1.0)
    colors = plt.get_cmap(cmap)(np.clip(norm, 0, 1))
    ax.add_collection3d(Poly3DCollection(V[F], facecolors=colors, edgecolors="none"))
    c, r = V.mean(0), np.abs(V - V.mean(0)).max()
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1))
    ax.axis("off")


def face_to_vertex_labels(F: np.ndarray, face_labels: np.ndarray, n_vertices: int) -> np.ndarray:
    """Majority vote of incident face labels per vertex (ties -> smallest label)."""
    K = int(face_labels.max()) + 1
    votes = np.zeros((n_vertices, K), dtype=np.int64)
    for j in range(3):
        np.add.at(votes, (F[:, j], face_labels), 1)
    lab = votes.argmax(1)
    lab[votes.sum(1) == 0] = -1
    return lab


def _face_areas(V, F):
    return 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)


def aggregate(pred_dir, out, meshes=None, plots: bool = False) -> dict:
    from .data.dataset import mesh_name

    pred_dir, out = Path(pred_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list[Path]] = defaultdict(list)
    for f in sorted(pred_dir.glob("pred_*.npz")):
        groups[mesh_name(f)].append(f)
    if not groups:
        raise FileNotFoundError(f"no pred_*.npz files in {pred_dir}")
    mesh_dir = None
    if meshes is not None:
        mesh_dir = Path(meshes)
        if (mesh_dir / "meshes").is_dir():
            mesh_dir = mesh_dir / "meshes"

    results = {}
    for name, files in groups.items():
        combined = None
        for f in files:
            with np.load(f, allow_pickle=False) as d:
                contrib = d["scores_on_mesh"].astype(np.float64) * d["scale"][:, None]
                if combined is None:
                    combined = contrib
                    V, F, gt_lifted = d["V"], d["F"], d["gt_on_mesh"]
                else:
                    combined += contrib
        pred = combined.argmax(1)
        res = dict(combined_scores=combined, prediction=pred, V=V, F=F,
                   num_triplets=np.int64(len(files)))
        gt_v = gt_lifted
        face_acc = None
        if mesh_dir is not None and (mesh_dir / f"{name}.npz").exists():
            with np.load(mesh_dir / f"{name}.npz", allow_pickle=False) as d:
                face_labels = d["labels"].astype(np.int64)
            gt_v = face_to_vertex_labels(F, face_labels, len(V))
            face_pred = combined[F].sum(1).argmax(1)
            area = _face_areas(V, F)
            face_acc = float((area * (face_pred == face_labels)).sum() / area.sum())
            res.update(face_labels=face_labels, face_prediction=face_pred, face_accuracy=face_acc)
        ok = gt_v >= 0
        vert_acc = float((pred[ok] == gt_v[ok]).mean()) if ok.any() else float("nan")
        res.update(gt_on_mesh=gt_v, vertex_accuracy=vert_acc)
        np.savez_compressed(out / f"{name}_combined.npz", **res)
        if plots:
            _plot(res, name, out)
        msg = f"{name}: {len(files)} triplets, vertex acc {vert_acc:.4f}"
        if face_acc is not None:
            msg += f", face acc (area) {face_acc:.4f}"
        print(msg)
        results[name] = dict(vertex_accuracy=vert_acc, face_accuracy=face_acc)

    va = [r["vertex_accuracy"] for r in results.values()]
    print(f"mean vertex accuracy over {len(va)} meshes: {np.nanmean(va):.4f}")
    fa = [r["face_accuracy"] for r in results.values() if r["face_accuracy"] is not None]
    if fa:
        print(f"mean face accuracy (area-weighted) over {len(fa)} meshes: {np.mean(fa):.4f}")
    return results


def _plot(res, name, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    K = res["combined_scores"].shape[1]
    fig = plt.figure(figsize=(10, 5))
    for i, (vals, t) in enumerate([(res["prediction"], "combined prediction"),
                                   (res["gt_on_mesh"], "GT")], start=1):
        a = fig.add_subplot(1, 2, i, projection="3d")
        plot_mesh(a, res["V"], res["F"], vals, cmap="jet", clim=(0, K - 1))
        a.set_title(t)
    fig.savefig(out / f"{name}_combined.png", dpi=100)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="toriccnn-aggregate", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred-dir", type=Path, required=True, help="folder with pred_*.npz")
    p.add_argument("--out", type=Path, default=None, help="output folder (default: --pred-dir)")
    p.add_argument("--meshes", type=Path, default=None,
                   help="processed folder (or its meshes/) with per-mesh face labels")
    p.add_argument("--plots", action="store_true")
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    aggregate(args.pred_dir, args.out or args.pred_dir, args.meshes, args.plots)


if __name__ == "__main__":
    main()
