"""PyTorch dataset over ``triplets/*.npz`` produced by ``toriccnn-make-data``."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

IGNORE_INDEX = -1  # seg values < 0 are ignored in statistics / metrics


def list_triplet_files(root, pattern: str = "*.npz") -> list[Path]:
    """Triplet files under ``root`` (either the processed dir or its ``triplets/``)."""
    root = Path(root)
    if (root / "triplets").is_dir():
        root = root / "triplets"
    return sorted(root.glob(pattern))


def mesh_name(path) -> str:
    """``train_foo_tri_12.npz`` -> ``train_foo``."""
    stem = Path(path).stem
    if stem.startswith("pred_"):
        stem = stem[len("pred_"):]
    head, sep, tail = stem.rpartition("_tri_")
    return head if sep and tail.isdigit() else stem


def split_train_val(files, val_frac: float = 0.05, seed: int = 0) -> tuple[list[str], list[str]]:
    """Random image-level split (as ``createIMDB.m``); returns sorted name lists."""
    files = sorted(str(f) for f in files)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(files))
    n_val = int(round(val_frac * len(files)))
    if val_frac > 0 and len(files) > 1:
        n_val = max(1, n_val)
    n_val = min(n_val, max(0, len(files) - 1))
    val = sorted(files[i] for i in perm[:n_val])
    train = sorted(files[i] for i in perm[n_val:])
    return train, val


def compute_stats(files, num_classes: int) -> dict:
    """Per-channel mean / covariance and class pixel counts (``getDatasetStatistics.m``)."""
    m1, m2 = [], []
    counts = np.zeros(num_classes, dtype=np.int64)
    for f in files:
        with np.load(f, allow_pickle=False) as d:
            x = d["data"].astype(np.float64)
            seg = d["seg"].astype(np.int64)
        z = x.reshape(-1, x.shape[-1])
        m1.append(z.mean(0))
        m2.append(z.T @ z / len(z))
        ok = (seg >= 0) & (seg < num_classes)
        counts += np.bincount(seg[ok], minlength=num_classes)[:num_classes]
    mean = np.mean(m1, 0)
    cov = np.mean(m2, 0) - np.outer(mean, mean)
    return dict(mean=mean.astype(np.float32), covariance=cov.astype(np.float32),
                class_counts=counts, num_channels=np.int64(len(mean)),
                num_classes=np.int64(num_classes))


class ToricDataset(Dataset):
    """Yields ``(data (C,H,W) float32 minus channel mean, seg (H,W) int64)``.

    ``files`` defaults to every npz in ``root``/triplets. ``size`` optionally
    resizes (bilinear for data, nearest for seg). ``augment`` enables random
    circular shifts and 90-degree rotations, which are symmetries of the
    periodic toric image (off by default, as in the original training).
    """

    def __init__(self, root=None, files=None, mean=None, size: int | None = None,
                 augment: bool = False, seed: int | None = None):
        if files is None:
            files = list_triplet_files(root)
        elif root is not None:
            files = [f if Path(f).is_absolute() or Path(f).exists() else Path(root) / f for f in files]
        self.files = [Path(f) for f in files]
        self.mean = None if mean is None else torch.as_tensor(np.asarray(mean, np.float32)).view(-1, 1, 1)
        self.size = size
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i):
        with np.load(self.files[i], allow_pickle=False) as d:
            data = torch.from_numpy(d["data"].astype(np.float32)).permute(2, 0, 1).contiguous()
            seg = torch.from_numpy(d["seg"].astype(np.int64))
        if self.mean is not None:
            data = data - self.mean
        if self.size is not None and tuple(data.shape[1:]) != (self.size, self.size):
            import torch.nn.functional as Fn
            data = Fn.interpolate(data[None], size=(self.size, self.size), mode="bilinear",
                                  align_corners=False)[0]
            seg = Fn.interpolate(seg[None, None].float(), size=(self.size, self.size),
                                 mode="nearest")[0, 0].long()
        if self.augment:
            k = int(self.rng.integers(4))
            sy, sx = (int(v) for v in self.rng.integers(0, seg.shape, size=2))
            data = torch.roll(torch.rot90(data, k, (1, 2)), (sy, sx), (1, 2))
            seg = torch.roll(torch.rot90(seg, k, (0, 1)), (sy, sx), (0, 1))
        return data, seg
