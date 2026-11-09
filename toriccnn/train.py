"""Train the toric FCN-32s (port of ``fcnTrain.m`` / MatConvNet ``cnn_train_dag``).

Writes to ``--exp-dir``: ``split.json``, ``stats.npz``, ``config.json``,
``net-epoch-N.pt`` and ``log.csv``. Re-running with the same ``--exp-dir``
resumes from the last checkpoint (MatConvNet ``continue``).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path

import numpy as np

DEFAULT_CLASSES = ["head", "hand", "lower arm", "upper arm", "torso", "upper leg",
                   "lower leg", "foot"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="toriccnn-train", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True,
                   help="processed data folder (containing triplets/) from toriccnn-make-data")
    p.add_argument("--exp-dir", type=Path, required=True)
    p.add_argument("--pattern", default="*.npz", help="glob for training triplet files")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--sub-batches", type=int, default=5,
                   help="split each batch into this many sub-batches (gradient accumulation)")
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-classes", type=int, default=8)
    p.add_argument("--class-names", nargs="+", default=DEFAULT_CLASSES)
    p.add_argument("--prior", type=float, default=4000.0, help="loss class-count prior")
    p.add_argument("--net-mode", choices=["matlab", "toric"], default="matlab",
                   help="matlab: faithful FCN-32s (pad 100); toric: exactly shift-equivariant variant")
    p.add_argument("--no-pretrained", action="store_true", help="do not load ImageNet VGG16 weights")
    p.add_argument("--augment", action="store_true", help="random toric shifts / rot90")
    p.add_argument("--amp", action="store_true", help="mixed precision (CUDA)")
    p.add_argument("--workers", type=int, default=2, help="DataLoader workers")
    p.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    return p


def last_checkpoint(exp_dir: Path) -> tuple[int, Path | None]:
    best, path = 0, None
    for f in exp_dir.glob("net-epoch-*.pt"):
        m = re.fullmatch(r"net-epoch-(\d+)\.pt", f.name)
        if m and int(m.group(1)) > best:
            best, path = int(m.group(1)), f
    return best, path


def load_model(exp_dir, epoch: int | None = None, device="cpu"):
    """Load ``(model, stats, config, epoch)`` from an experiment directory."""
    import torch

    from .nn import ToricFCN32

    exp_dir = Path(exp_dir)
    if epoch is None:
        epoch, path = last_checkpoint(exp_dir)
        if path is None:
            raise FileNotFoundError(f"no checkpoints in {exp_dir}")
    else:
        path = exp_dir / f"net-epoch-{epoch}.pt"
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = ToricFCN32(cfg["in_channels"], cfg["num_classes"], pretrained=False,
                       mode=cfg.get("net_mode", "matlab"))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    with np.load(exp_dir / "stats.npz", allow_pickle=False) as d:
        stats = {k: d[k] for k in d.files}
    return model, stats, cfg, epoch


def _run_epoch(model, loader, opt, scaler, args, device, train: bool):
    import torch

    from .nn import ConfusionMeter, class_scaled_cross_entropy

    model.train(train)
    meter = ConfusionMeter(args.num_classes)
    total_loss, n_img = 0.0, 0
    amp = bool(args.amp and device.type == "cuda")
    for data, seg in loader:
        data = data.to(device, non_blocking=True)
        seg = seg.to(device, non_blocking=True)
        bs = data.shape[0]
        if train:
            opt.zero_grad(set_to_none=True)
        n_sub = max(1, min(args.sub_batches, bs))
        for d_sub, s_sub in zip(data.chunk(n_sub), seg.chunk(n_sub)):
            with torch.set_grad_enabled(train), torch.autocast(device.type, enabled=amp):
                logits = model(d_sub)
            loss = class_scaled_cross_entropy(logits.float(), s_sub, prior=args.prior)
            if train:
                # cnn_train divides the accumulated derivative by the batch size
                scaler.scale(loss / bs).backward()
            total_loss += float(loss.detach())
            meter.update(logits.detach(), s_sub)
        if train:
            scaler.step(opt)
            scaler.update()
        n_img += bs
    return dict(loss=total_loss / max(1, n_img), pixel_acc=meter.pixel_acc,
                mean_acc=meter.mean_acc, miou=meter.mean_iou)


def train(args) -> Path:
    import torch
    from torch.utils.data import DataLoader

    from .data.dataset import ToricDataset, compute_stats, list_triplet_files, split_train_val
    from .nn import ToricFCN32

    exp = Path(args.exp_dir)
    exp.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if len(args.class_names) != args.num_classes:
        args.class_names = [f"class{i}" for i in range(args.num_classes)] \
            if args.class_names == DEFAULT_CLASSES else args.class_names
        if len(args.class_names) != args.num_classes:
            raise ValueError("--class-names must have --num-classes entries")

    split_path = exp / "split.json"
    if split_path.exists():
        split = json.loads(split_path.read_text())
    else:
        files = list_triplet_files(args.data, args.pattern)
        if not files:
            raise FileNotFoundError(f"no triplet files matching {args.pattern} in {args.data}")
        tr, va = split_train_val([str(Path(f).resolve()) for f in files], args.val_frac, args.seed)
        split = dict(train=tr, val=va)
        split_path.write_text(json.dumps(split, indent=1))

    stats_path = exp / "stats.npz"
    if not stats_path.exists():
        print(f"computing dataset statistics over {len(split['train'])} training images ...")
        np.savez(stats_path, **compute_stats(split["train"], args.num_classes))
    with np.load(stats_path, allow_pickle=False) as d:
        mean = d["mean"]
    in_channels = int(len(mean))

    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    cfg.update(in_channels=in_channels, num_classes=args.num_classes)
    (exp / "config.json").write_text(json.dumps(cfg, indent=1))

    ds_tr = ToricDataset(files=split["train"], mean=mean, augment=args.augment, seed=args.seed)
    ds_va = ToricDataset(files=split["val"], mean=mean)
    pin = device.type == "cuda"
    gen = torch.Generator().manual_seed(args.seed)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                       pin_memory=pin, generator=gen, persistent_workers=args.workers > 0)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.workers, pin_memory=pin)

    model = ToricFCN32(in_channels, args.num_classes, pretrained=not args.no_pretrained,
                       mode=args.net_mode).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=bool(args.amp and device.type == "cuda"))

    start, ckpt_path = last_checkpoint(exp)
    if ckpt_path is not None:
        print(f"resuming from {ckpt_path.name}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        torch.set_rng_state(ckpt["torch_rng"].cpu())
        gen.set_state(ckpt["loader_rng"].cpu())

    log_path = exp / "log.csv"
    fields = ["epoch", "time"] + [f"{s}_{m}" for s in ("train", "val")
                                  for m in ("loss", "pixel_acc", "mean_acc", "miou")]
    if not log_path.exists() or start == 0:
        with open(log_path, "w", newline="") as fh:
            csv.writer(fh).writerow(fields)

    for epoch in range(start + 1, args.epochs + 1):
        t0 = time.time()
        tr = _run_epoch(model, dl_tr, opt, scaler, args, device, train=True)
        va = _run_epoch(model, dl_va, None, scaler, args, device, train=False) if len(ds_va) else \
            dict(loss=float("nan"), pixel_acc=float("nan"), mean_acc=float("nan"), miou=float("nan"))
        dt = time.time() - t0
        row = [epoch, f"{dt:.1f}"] + [f"{r[m]:.6g}" for r in (tr, va)
                                      for m in ("loss", "pixel_acc", "mean_acc", "miou")]
        with open(log_path, "a", newline="") as fh:
            csv.writer(fh).writerow(row)
        print(f"epoch {epoch:3d} ({dt:.0f}s) train loss {tr['loss']:.4g} acc {tr['pixel_acc']:.3f} "
              f"macc {tr['mean_acc']:.3f} miou {tr['miou']:.3f} | val loss {va['loss']:.4g} "
              f"acc {va['pixel_acc']:.3f} macc {va['mean_acc']:.3f} miou {va['miou']:.3f}")
        torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(),
                        scaler=scaler.state_dict(), epoch=epoch, config=cfg,
                        torch_rng=torch.get_rng_state(), loader_rng=gen.get_state()),
                   exp / f"net-epoch-{epoch}.pt")
    return exp


def main(argv=None) -> None:
    train(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
