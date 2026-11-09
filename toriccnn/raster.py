"""Rasterise functions on a square-orbifold flattening into toric images.

Port of the rendering part of ``make_tiling.m`` (+ ``getframe``). The flat mesh
is replicated 4 times (rotations by ``-i*pi/2`` about the origin followed by
fixed translations); together the 4 copies form one fundamental domain of the
4-periodic tiling of the plane. Instead of drawing the 9 translated tilings
we wrap pixel indices modulo the image size, which is equivalent.

Viewport: x, y in [-3, 1]^2, row 0 is y = +1 (top), column 0 is x = -3.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage

PERIOD = 4.0
X0, Y1 = -3.0, 1.0
_COPY_TRANSLATIONS = np.array([[0.0, 0.0], [0.0, -2.0], [-2.0, -2.0], [-2.0, 0.0]])


def plane_to_pixel(xy: np.ndarray, size) -> tuple[np.ndarray, np.ndarray]:
    """Continuous (row, col) pixel coordinates; integers are pixel centres."""
    H, W = (size, size) if np.isscalar(size) else size
    xy = np.asarray(xy, dtype=np.float64)
    col = (xy[..., 0] - X0) / PERIOD * W - 0.5
    row = (Y1 - xy[..., 1]) / PERIOD * H - 0.5
    return row, col


def torus_copies(V_flat: np.ndarray) -> list[np.ndarray]:
    """The 4 rotated/translated copies of the flat mesh (``make_tiling.m``)."""
    out = []
    for i in range(4):
        a = -i * np.pi / 2
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        out.append(V_flat @ R.T + _COPY_TRANSLATIONS[i])
    return out


def _device(device):
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _fill_cracks(img: np.ndarray, covered: np.ndarray) -> np.ndarray:
    """Fill uncovered pixels with the nearest covered pixel (toric distance)."""
    if covered.all():
        return img
    if not covered.any():
        raise RuntimeError("rasterisation covered no pixel")
    H, W = covered.shape
    big = np.tile(~covered, (3, 3))
    _, (ri, ci) = ndimage.distance_transform_edt(big, return_indices=True)
    ri = ri[H:2 * H, W:2 * W] % H
    ci = ci[H:2 * H, W:2 * W] % W
    return img[ri, ci]


def _rasterize(flat, size, device, eps=1e-6):
    """Yield, for every covered (triangle, pixel) pair: face index, barycentrics, linear pixel index.

    Returns torch tensors ``(face (N,), bary (N,3), pix (N,))`` on ``device``.
    """
    H, W = (size, size) if np.isscalar(size) else size
    T = np.asarray(flat.cut.T, dtype=np.int64)
    m = len(T)
    P = []
    for Vc in torus_copies(np.asarray(flat.V_flat, dtype=np.float64)):
        r, c = plane_to_pixel(Vc, (H, W))
        P.append(np.stack([r, c], 1)[T])                    # (m, 3, 2)
    P = torch.as_tensor(np.concatenate(P), dtype=torch.float64, device=device)  # (4m, 3, 2)
    face = torch.arange(4 * m, device=device) % m

    lo = torch.ceil(P.amin(1) - eps).long()                 # (4m, 2) row/col
    hi = torch.floor(P.amax(1) + eps).long()
    ext = (hi - lo + 1).clamp(min=0)
    counts = ext[:, 0] * ext[:, 1]
    keep = counts > 0
    P, face, lo, ext, counts = P[keep], face[keep], lo[keep], ext[keep], counts[keep]

    tri = torch.repeat_interleave(torch.arange(len(P), device=device), counts)
    starts = torch.cumsum(counts, 0) - counts
    local = torch.arange(len(tri), device=device) - starts[tri]
    w = ext[tri, 1]
    rr = lo[tri, 0] + local // w
    cc = lo[tri, 1] + local % w

    p0, p1, p2 = P[:, 0], P[:, 1], P[:, 2]
    d1, d2 = p1 - p0, p2 - p0
    det = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    good = det.abs() > 1e-14
    inv = torch.where(good, 1.0 / torch.where(good, det, torch.ones_like(det)), torch.zeros_like(det))

    q0 = rr.double() - p0[tri, 0]
    q1 = cc.double() - p0[tri, 1]
    it = inv[tri]
    b1 = (q0 * d2[tri, 1] - q1 * d2[tri, 0]) * it
    b2 = (d1[tri, 0] * q1 - d1[tri, 1] * q0) * it
    b0 = 1.0 - b1 - b2
    inside = (b0 >= -eps) & (b1 >= -eps) & (b2 >= -eps) & good[tri]

    tri, rr, cc = tri[inside], rr[inside], cc[inside]
    bary = torch.stack([b0[inside], b1[inside], b2[inside]], 1)
    pix = (rr % H) * W + (cc % W)
    return face[tri], bary, pix


def rasterize_vertex_functions(flat, values: np.ndarray, size: int = 512, device=None) -> np.ndarray:
    """Barycentric-linear rendering of per-(uncut-)vertex functions, (H, W, C) float32."""
    dev = _device(device)
    H, W = (size, size) if np.isscalar(size) else size
    values = np.asarray(values)
    squeeze = values.ndim == 1
    if squeeze:
        values = values[:, None]
    C = values.shape[1]
    vals_cut = torch.as_tensor(values[flat.cut.cut_to_uncut], dtype=torch.float32, device=dev)
    T = torch.as_tensor(np.asarray(flat.cut.T, dtype=np.int64), device=dev)

    face, bary, pix = _rasterize(flat, (H, W), dev)
    tv = T[face]                                           # (N, 3)
    b = bary.float()
    out = torch.zeros(H * W, C, dtype=torch.float32, device=dev)
    cnt = torch.zeros(H * W, dtype=torch.float32, device=dev)
    chunk = max(1, int(2 ** 25 // max(C, 1)))
    for s in range(0, len(pix), chunk):
        sl = slice(s, s + chunk)
        v = (vals_cut[tv[sl, 0]] * b[sl, 0:1] + vals_cut[tv[sl, 1]] * b[sl, 1:2]
             + vals_cut[tv[sl, 2]] * b[sl, 2:3])
        out.index_add_(0, pix[sl], v)
        cnt.index_add_(0, pix[sl], torch.ones_like(pix[sl], dtype=torch.float32))
    covered = cnt > 0
    out[covered] /= cnt[covered, None]
    img = out.reshape(H, W, C).cpu().numpy()
    img = _fill_cracks(img, covered.reshape(H, W).cpu().numpy())
    return img[..., 0] if squeeze else img


def rasterize_face_labels(flat, labels: np.ndarray, size: int = 512, device=None) -> np.ndarray:
    """Render a per-face label (flat shading), (H, W) int64."""
    dev = _device(device)
    H, W = (size, size) if np.isscalar(size) else size
    labels_t = torch.as_tensor(np.asarray(labels, dtype=np.int64), device=dev)
    face, _, pix = _rasterize(flat, (H, W), dev)
    out = torch.full((H * W,), -1, dtype=torch.int64, device=dev)
    # deterministic tie-break on shared edges: largest face index wins
    best = torch.full((H * W,), -1, dtype=torch.int64, device=dev)
    best.scatter_reduce_(0, pix, face, reduce="amax")
    covered = best >= 0
    out[covered] = labels_t[best[covered]]
    img = out.reshape(H, W).cpu().numpy()
    return _fill_cracks(img, covered.reshape(H, W).cpu().numpy())
