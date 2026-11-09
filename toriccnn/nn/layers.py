"""Toric building blocks: circular padding (``cPad.m``) and rotation max-projection
(``MaxProjection.m`` / ``doMaxProjectionForward.m``)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def circular_pad2d(x: torch.Tensor, pad: int) -> torch.Tensor:
    """Pad the last two dims of ``x`` periodically by ``pad`` on every side.

    Equivalent to MATLAB ``padarray(x, [pad pad], 'circular')``. Unlike
    ``F.pad(mode="circular")`` this also works when ``pad`` exceeds the
    spatial size (the periodic image is tiled as many times as needed).
    """
    if pad == 0:
        return x
    H, W = x.shape[-2:]
    if pad <= H and pad <= W:
        return F.pad(x, (pad, pad, pad, pad), mode="circular")
    iy = torch.arange(-pad, H + pad, device=x.device) % H
    ix = torch.arange(-pad, W + pad, device=x.device) % W
    return x.index_select(-2, iy).index_select(-1, ix)


class CircularPad2d(nn.Module):
    """Periodic (toric) padding of ``pad`` pixels on each spatial side."""

    def __init__(self, pad: int):
        super().__init__()
        self.pad = int(pad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return circular_pad2d(x, self.pad)

    def extra_repr(self) -> str:
        return f"pad={self.pad}"


class MaxProjection(nn.Module):
    """``out = max(x, rot90 x, rot180 x, rot270 x)`` over the spatial dims.

    Requires square maps (H == W). The output is invariant to 90-degree
    rotations of the input. The gradient is given by autograd (routed to the
    arg-max rotation), replacing MATLAB's hand-written approximate backward.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        if H != W:
            raise ValueError(f"MaxProjection needs square maps, got {H}x{W}")
        r1 = torch.rot90(x, 1, dims=(-2, -1))
        r2 = torch.rot90(x, 2, dims=(-2, -1))
        r3 = torch.rot90(x, 3, dims=(-2, -1))
        return torch.maximum(torch.maximum(x, r1), torch.maximum(r2, r3))
