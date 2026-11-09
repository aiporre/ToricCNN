"""Per-triplet prediction lifted to the mesh (port of ``generateSegmentationsPerTriplet.m``).

For every triplet npz in ``--data`` the network's raw (max-projected, no softmax)
scores are lifted to the mesh vertices; writes ``{out}/pred_{name}.npz`` with
``scores_on_mesh`` (n, K), ``prediction`` (n,) 0-based, ``gt_on_mesh`` (n,),
``scale`` (n,), ``V``, ``F``, ``cones``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="toriccnn-predict", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp-dir", type=Path, required=True, help="training experiment folder")
    p.add_argument("--epoch", type=int, default=None, help="checkpoint epoch (default: last)")
    p.add_argument("--data", type=Path, required=True,
                   help="processed test folder (containing triplets/) or a triplets folder")
    p.add_argument("--pattern", default="*.npz", help="glob for triplet files")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--lift-mode", default="nearest", help="lift_image sampling mode for scores")
    p.add_argument("--plots", action="store_true", help="save image / mesh PNGs")
    p.add_argument("--device", default=None)
    return p


def predict_file(model, mean, path, device, lift_mode: str = "nearest") -> dict:
    import torch

    from .lift import lift_image, scale_on_mesh
    from .orbifold.flattener import Flattening

    with np.load(path, allow_pickle=False) as d:
        arrays = {k: d[k] for k in d.files}
    flat = Flattening.from_arrays(arrays)
    x = arrays["data"].astype(np.float32) - np.asarray(mean, np.float32)
    x = torch.from_numpy(x).permute(2, 0, 1)[None].to(device)
    with torch.no_grad():
        scores = model(x)[0].float().permute(1, 2, 0).cpu().numpy()  # (H, W, K)
    seg = arrays["seg"].astype(np.int64)
    scores_on_mesh = np.asarray(lift_image(flat, scores, mode=lift_mode), dtype=np.float32)
    gt_on_mesh = np.rint(lift_image(flat, seg, mode="nearest")).astype(np.int64)
    return dict(scores_on_mesh=scores_on_mesh, prediction=scores_on_mesh.argmax(1),
                gt_on_mesh=gt_on_mesh, scale=np.asarray(scale_on_mesh(flat), np.float64),
                V=arrays["V"], F=arrays["F"], cones=arrays["cones"],
                image_prediction=scores.argmax(-1).astype(np.int16), seg=seg.astype(np.int16))


def plot_prediction(res: dict, name: str, out: Path, num_classes: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .aggregate import plot_mesh

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    for a, img, t in zip(ax, (res["image_prediction"], res["seg"]), ("prediction", "GT")):
        a.imshow(img, cmap="jet", vmin=0, vmax=num_classes - 1, interpolation="nearest")
        a.set_title(t)
        a.axis("off")
    fig.savefig(out / f"{name}_image.png", dpi=100)
    plt.close(fig)

    V, F, cones = res["V"], res["F"], res["cones"]
    fig = plt.figure(figsize=(15, 5))
    panels = [(res["prediction"], "prediction", "jet", (0, num_classes - 1)),
              (np.log(np.abs(res["scale"]) + 1e-12), "log(scale)", "viridis", None),
              (res["gt_on_mesh"], "GT", "jet", (0, num_classes - 1))]
    for i, (vals, title, cmap, lim) in enumerate(panels, start=1):
        a = fig.add_subplot(1, 3, i, projection="3d")
        plot_mesh(a, V, F, vals, cmap=cmap, clim=lim)
        a.scatter(*V[cones].T, c="r", s=30, depthshade=False)
        a.set_title(title)
    fig.savefig(out / f"{name}_mesh.png", dpi=100)
    plt.close(fig)


def predict(args) -> list[Path]:
    import torch

    from .data.dataset import list_triplet_files
    from .train import load_model

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, stats, cfg, epoch = load_model(args.exp_dir, args.epoch, device)
    files = list_triplet_files(args.data, args.pattern)
    if not files:
        raise FileNotFoundError(f"no triplet files matching {args.pattern} in {args.data}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for i, f in enumerate(files, start=1):
        print(f"{f.stem}: running on instance {i}/{len(files)} (epoch {epoch})")
        res = predict_file(model, stats["mean"], f, device, args.lift_mode)
        if args.plots:
            plot_prediction(res, f.stem, out, cfg["num_classes"])
        path = out / f"pred_{f.stem}.npz"
        np.savez_compressed(path, **{k: v for k, v in res.items()
                                     if k not in ("image_prediction", "seg")})
        written.append(path)
    return written


def main(argv=None) -> None:
    predict(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
