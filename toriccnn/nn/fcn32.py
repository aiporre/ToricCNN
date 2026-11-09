"""Toric FCN-32s (``fcnInitializeModel32s.m``): VGG16 FCN-32s where every
convolution's zero padding is replaced by circular padding, followed by a
rotation max-projection.

Two geometries are provided:

``mode="matlab"`` (default, faithful port)
    conv1_1 is preceded by a circular pad of 100 (FCN-32s convention), the
    other 3x3 convs by circular pad 1, fc6 (7x7) / fc7 (1x1) are unpadded,
    pools are 2x2/2 with ceil mode, the 64/32 deconvolution output is cropped
    with offset 19 (the standard FCN-32s offset; the circular pad has exactly
    the same geometry as the zero pad, so the alignment is unchanged). For a
    512 input: 712 -> 710 -> 355 -> 178 -> 89 -> 45 -> 23 -> fc6 17 -> upscore
    576 -> crop [19:531] = 512. Because the 100-pixel pad is a *window* of the
    periodic image (not a torus of size H+200), and the receptive field of
    fc7 (404 px) exceeds that window, this network is translation-equivariant
    only approximately (exact in the interior, not near the borders).

``mode="toric"`` (exactly toric variant, not in the MATLAB code)
    every 3x3 conv uses circular pad 1 (including conv1_1), fc6 uses circular
    pad 3, pools are exact (H, W must be multiples of 32), and the upsampling
    deconvolution is performed on the torus (overlapping contributions are
    wrapped around) with centred offset 16. The whole map x -> f(x) then
    commutes exactly with circular shifts by multiples of 32 pixels.

Note on input scaling: torchvision's VGG16 weights expect inputs roughly in
[0, 1] scale normalised by ImageNet mean/std, whereas the MatConvNet model used
0..255 inputs with mean subtraction; normalise the data accordingly.
"""
from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import CircularPad2d, MaxProjection

_VGG16_CFG = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M",
              512, 512, 512, "M", 512, 512, 512, "M"]
_VGG16_CONV_IDX = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]


def bilinear_kernel(num_classes: int, size: int = 64) -> torch.Tensor:
    """(K, K, size, size) bilinear upsampling kernel, diagonal over classes."""
    factor = (size + 1) // 2
    center = factor - 1 if size % 2 == 1 else factor - 0.5
    og = torch.arange(size, dtype=torch.float64)
    f1 = 1 - (og - center).abs() / factor
    filt = (f1[:, None] * f1[None, :]).float()
    w = torch.zeros(num_classes, num_classes, size, size)
    w[range(num_classes), range(num_classes)] = filt
    return w


def tile_input_kernels(w: torch.Tensor, in_channels: int) -> torch.Tensor:
    """MATLAB rule: repeat the RGB kernels floor(C/3) times along the input
    dim, then append the first C - 3*floor(C/3) channels of the result."""
    reps = in_channels // w.shape[1]
    out = w.repeat(1, reps, 1, 1)
    diff = in_channels - out.shape[1]
    if diff > 0:
        src = out if out.shape[1] > 0 else w
        out = torch.cat([out, src[:, :diff]], dim=1)
    return out


def _wrap_add(t: torch.Tensor, n: int, offset: int, dim: int) -> torch.Tensor:
    """Fold a length-L axis onto a torus of length n: out[(i-offset) % n] += t[i]."""
    L = t.shape[dim]
    idx = (torch.arange(L, device=t.device) - offset) % n
    shape = list(t.shape)
    shape[dim] = n
    return t.new_zeros(shape).index_add(dim, idx, t)


class ToricFCN32(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, pretrained: bool = True,
                 mode: str = "matlab", input_scale: float = 1.0 / (255.0 * 0.226)):
        super().__init__()
        # Inputs are mean-subtracted functions on a [0, 255] scale (as in MatConvNet);
        # torchvision's ImageNet VGG16 expects ~unit-std inputs, so rescale (stored
        # in the state_dict so prediction always matches training).
        self.register_buffer("input_scale", torch.tensor(float(input_scale)))
        if mode not in ("matlab", "toric"):
            raise ValueError(f"unknown mode {mode!r}")
        self.mode = mode
        self.in_channels = in_channels
        self.num_classes = num_classes

        layers: list[nn.Module] = []
        convs: list[nn.Conv2d] = []
        c_in = in_channels
        first = True
        for v in _VGG16_CFG:
            if v == "M":
                layers.append(nn.MaxPool2d(2, 2, ceil_mode=True))
                continue
            pad = 100 if (first and mode == "matlab") else 1
            conv = nn.Conv2d(c_in, v, 3, padding=0)
            layers += [CircularPad2d(pad), conv, nn.ReLU(inplace=True)]
            convs.append(conv)
            c_in = v
            first = False
        self.features = nn.Sequential(*layers)

        self.fc6 = nn.Conv2d(512, 4096, 7, padding=0)
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.head = nn.Sequential(
            CircularPad2d(3 if mode == "toric" else 0), self.fc6, nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            self.fc7, nn.ReLU(inplace=True), nn.Dropout(0.5),
        )
        self.score_fr = nn.Conv2d(4096, num_classes, 1)
        nn.init.zeros_(self.score_fr.weight)
        nn.init.zeros_(self.score_fr.bias)
        self.upscore = nn.ConvTranspose2d(num_classes, num_classes, 64, stride=32, bias=False)
        with torch.no_grad():
            self.upscore.weight.copy_(bilinear_kernel(num_classes, 64))
        self.proj = MaxProjection()

        if pretrained:
            self._load_vgg16(convs)

    # ------------------------------------------------------------------ init
    def _load_vgg16(self, convs: list[nn.Conv2d]) -> None:
        try:
            from torchvision.models import VGG16_Weights, vgg16
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        except Exception as e:  # offline, etc.
            warnings.warn(f"ToricFCN32: could not load pretrained VGG16 ({e!r}); "
                          "using random initialisation.")
            return
        with torch.no_grad():
            for conv, idx in zip(convs, _VGG16_CONV_IDX):
                src = vgg.features[idx]
                w = src.weight
                if conv is convs[0]:
                    w = tile_input_kernels(w, self.in_channels)
                conv.weight.copy_(w)
                conv.bias.copy_(src.bias)
            self.fc6.weight.copy_(vgg.classifier[0].weight.view(4096, 512, 7, 7))
            self.fc6.bias.copy_(vgg.classifier[0].bias)
            self.fc7.weight.copy_(vgg.classifier[3].weight.view(4096, 4096, 1, 1))
            self.fc7.bias.copy_(vgg.classifier[3].bias)

    # --------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> max-projected logits (B, K, H, W)."""
        return self.proj(self.forward_unprojected(x))

    def forward_unprojected(self, x: torch.Tensor) -> torch.Tensor:
        """Logits before the rotation max-projection (B, K, H, W).

        Note: the max-projection itself does not commute with circular shifts
        (rot90 turns a (s, s) shift into a (-s, s) one), so translation
        equivariance is a property of this pre-projection map.
        """
        H, W = x.shape[-2:]
        if self.mode == "toric" and (H % 32 or W % 32):
            raise ValueError("mode='toric' needs H and W to be multiples of 32")
        y = self.score_fr(self.head(self.features(x * self.input_scale)))
        y = self.upscore(y)
        if self.mode == "matlab":
            off = 19
            if y.shape[-2] < off + H or y.shape[-1] < off + W:
                raise RuntimeError(f"upscore output {tuple(y.shape[-2:])} too small to crop {H}x{W}")
            y = y[..., off:off + H, off:off + W]
        else:
            # bilinear 64/32 kernel centred at 31.5 maps coarse cell j (fine pixels
            # 32j..32j+31, centre 32j+15.5) to 32j+31.5 -> offset 16, wrapped on torus.
            y = _wrap_add(_wrap_add(y, H, 16, -2), W, 16, -1)
        return y
