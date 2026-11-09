"""Generate toric training images from labelled meshes (port of ``make_data.m``).

For every mesh ``<stem>.<ext>`` in the input folder (with per-face labels in
``<stem>.txt``, one 1-based integer per line) this writes, in ``--out``:

* ``triplets/{prefix}_{stem}_tri_{j}.npz``: ``data`` (H, W, 26) uint8,
  ``seg`` (H, W) int16 (0-based labels), all ``Flattening.to_arrays()`` keys and
  the area-normalised ``V``, ``F``;
* ``meshes/{prefix}_{stem}.npz``: ``functions`` (n, 26), ``labels`` (m,) 0-based,
  ``V``, ``F``, ``triplets`` (k, 3);
* ``selected_points/{prefix}_{stem}.png``: preview of the sampled cones.

All heavy imports are inside functions so ``--help`` is instant.
"""
from __future__ import annotations

import argparse
import logging
import traceback
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

log = logging.getLogger("toriccnn.make_data")


def load_face_labels(path: Path, n_faces: int) -> np.ndarray:
    """Read a 1-based per-face label txt file and return 0-based int64 labels."""
    labels = np.loadtxt(path, dtype=np.int64, ndmin=1).reshape(-1)
    if len(labels) != n_faces:
        raise ValueError(f"{path}: {len(labels)} labels for {n_faces} faces")
    return labels - 1


def process_triplet(V, F, functions, labels, cones, size: int) -> dict:
    """Flatten with ``cones`` and rasterise functions/labels. Returns npz arrays."""
    from ..orbifold.flattener import flatten_sphere
    from ..raster import rasterize_face_labels, rasterize_vertex_functions

    flat = flatten_sphere(V, F, cones)
    img = rasterize_vertex_functions(flat, functions, size=size)
    seg = rasterize_face_labels(flat, labels, size=size)
    out = dict(flat.to_arrays())
    out["data"] = np.clip(np.round(img), 0, 255).astype(np.uint8)
    out["seg"] = np.asarray(seg).astype(np.int16)
    out["V"] = V
    out["F"] = F
    return out


def _triplet_job(args):
    V, F, functions, labels, cones, size, path = args
    try:
        arrays = process_triplet(V, F, functions, labels, cones, size)
        np.savez_compressed(path, **arrays)
        return path, None
    except Exception:  # noqa: BLE001 - logged and skipped, like a failed triplet
        return path, traceback.format_exc()


def plot_selected_points(V, F, agd, triplets, points, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ..aggregate import plot_mesh

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection="3d")
    plot_mesh(ax, V, F, agd, cmap="viridis")
    cones = np.unique(triplets)
    ax.scatter(*V[points].T, c="k", s=12, depthshade=False, label="FPS points")
    ax.scatter(*V[cones].T, c="r", s=30, depthshade=False, label="cones")
    ax.legend(loc="lower left", fontsize=7)
    ax.set_title(f"{path.stem}: {len(triplets)} triplets")
    fig.savefig(path, dpi=100)
    plt.close(fig)


def process_mesh(mesh_path: Path, out: Path, prefix: str = "train", num_triplets: int = 300,
                 size: int = 512, seed: int = 0, workers: int = 0, plots: bool = True,
                 n_extremities: int = 5, n_farthest: int = 20, T: float = 0.4) -> int:
    """Process one mesh; returns the number of successfully written triplets."""
    from ..features import functions_on_mesh
    from ..features.agd import compute_agd, pairwise_graph_distances
    from ..mesh.io import load_mesh, normalize_area
    from ..sampling import select_triplets

    stem = mesh_path.stem
    name = f"{prefix}_{stem}"
    V, F = load_mesh(mesh_path)
    labels = load_face_labels(mesh_path.with_suffix(".txt"), len(F))
    V = normalize_area(V, F)

    dist = pairwise_graph_distances(V, F)
    agd = compute_agd(V, F, dist)
    functions = functions_on_mesh(V, F, dist, agd)
    rng = np.random.default_rng([seed, zlib.crc32(stem.encode())])
    triplets, points = select_triplets(F, dist, agd, n=n_extremities, n_farthest=n_farthest,
                                       num_triplets=num_triplets, T=T, rng=rng)
    del dist

    np.savez_compressed(out / "meshes" / f"{name}.npz", functions=functions, labels=labels,
                        V=V, F=F, triplets=triplets)
    if plots:
        try:
            plot_selected_points(V, F, agd, triplets, points, out / "selected_points" / f"{name}.png")
        except Exception:  # noqa: BLE001
            log.warning("plotting failed for %s:\n%s", name, traceback.format_exc())

    jobs = [(V, F, functions, labels, cones, size, out / "triplets" / f"{name}_tri_{j}.npz")
            for j, cones in enumerate(triplets, start=1)]
    if workers and workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            results = list(ex.map(_triplet_job, jobs))
    else:
        results = [_triplet_job(j) for j in jobs]
    ok = 0
    for path, err in results:
        if err is None:
            ok += 1
        else:
            log.warning("triplet %s failed, skipped:\n%s", Path(path).name, err)
    return ok


def make_data(in_dir, out, ext: str = "ply", **kwargs) -> list[str]:
    """Process every ``*.{ext}`` mesh in ``in_dir``; returns the processed stems."""
    in_dir, out = Path(in_dir), Path(out)
    for sub in ("triplets", "meshes", "selected_points"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    files = sorted(in_dir.glob(f"*.{ext.lstrip('.')}"))
    if not files:
        raise FileNotFoundError(f"no *.{ext} meshes in {in_dir}")
    done = []
    for i, f in enumerate(files, start=1):
        log.info("Processing mesh %d/%d: %s", i, len(files), f.stem)
        try:
            n = process_mesh(f, out, **kwargs)
        except Exception:  # noqa: BLE001
            log.error("mesh %s failed, skipped:\n%s", f.stem, traceback.format_exc())
            continue
        log.info("  %s: %d triplets written", f.stem, n)
        done.append(f.stem)
    return done


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="toriccnn-make-data", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", type=Path, help="folder with meshes and <stem>.txt face labels")
    p.add_argument("--out", type=Path, required=True, help="output folder")
    p.add_argument("--ext", default="ply", help="mesh extension: ply, off, obj (default ply)")
    p.add_argument("--prefix", default="train", help="file prefix, e.g. train / test")
    p.add_argument("--num-triplets", type=int, default=300)
    p.add_argument("--size", type=int, default=512, help="image size in pixels")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=0, help="process pool size over triplets")
    p.add_argument("--n-extremities", type=int, default=5, help="AGD extremities (n)")
    p.add_argument("--n-farthest", type=int, default=20, help="farthest point samples")
    p.add_argument("--T", type=float, default=0.4, help="extremity separation (fraction of diameter)")
    p.add_argument("--no-plots", action="store_true", help="skip selected-points previews")
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    make_data(args.input, args.out, ext=args.ext, prefix=args.prefix,
              num_triplets=args.num_triplets, size=args.size, seed=args.seed,
              workers=args.workers, plots=not args.no_plots, n_extremities=args.n_extremities,
              n_farthest=args.n_farthest, T=args.T)
    log.info("Done processing data!")


if __name__ == "__main__":
    main()
