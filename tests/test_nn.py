import math

import pytest
import torch
import torch.nn.functional as F

from toriccnn.nn import (CircularPad2d, ConfusionMeter, MaxProjection, ToricFCN32,
                         circular_pad2d, class_scaled_cross_entropy, tile_input_kernels)

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- layers
def test_maxprojection_rot_invariant_and_grad():
    x = torch.randn(2, 3, 9, 9, requires_grad=True)
    mp = MaxProjection()
    y = mp(x)
    for k in (1, 2, 3):
        assert torch.equal(mp(torch.rot90(x, k, dims=(-2, -1))), y)
    # output itself is rotation symmetric
    assert torch.equal(torch.rot90(y, 1, dims=(-2, -1)), y)
    y.sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    with pytest.raises(ValueError):
        mp(torch.randn(1, 1, 4, 5))


def test_circular_pad_matches_padarray_and_large_pad():
    x = torch.randn(1, 2, 5, 6)
    ref = F.pad(x, (2, 2, 2, 2), mode="circular")
    assert torch.equal(CircularPad2d(2)(x), ref)
    big = circular_pad2d(x, 13)  # pad > size: tiling
    assert big.shape[-2:] == (5 + 26, 6 + 26)
    iy = torch.arange(-13, 5 + 13) % 5
    ix = torch.arange(-13, 6 + 13) % 6
    assert torch.equal(big, x[..., iy, :][..., ix])


@pytest.mark.parametrize("p", [1, 2, 3])
def test_circular_conv_equals_roll_convolution(p):
    torch.manual_seed(0)
    x = torch.randn(2, 3, 11, 13, dtype=torch.float64)
    w = torch.randn(4, 3, 2 * p + 1, 2 * p + 1, dtype=torch.float64)
    out = F.conv2d(CircularPad2d(p)(x), w, padding=0)
    ref = torch.zeros(2, 4, 11, 13, dtype=torch.float64)
    for u in range(2 * p + 1):
        for v in range(2 * p + 1):
            xs = torch.roll(x, shifts=(p - u, p - v), dims=(-2, -1))  # x[i+u-p, j+v-p]
            ref += torch.einsum("oc,bchw->bohw", w[:, :, u, v], xs)
    assert torch.allclose(out, ref, atol=1e-10)
    # and the FFT view: circular cross-correlation == ifft(fft(x) * conj(fft(kernel)))
    k = torch.zeros(4, 3, 11, 13, dtype=torch.float64)
    for u in range(2 * p + 1):
        for v in range(2 * p + 1):
            k[:, :, (u - p) % 11, (v - p) % 13] = w[:, :, u, v]
    fx = torch.fft.fft2(x)[:, None]
    fk = torch.fft.fft2(k)[None]
    ref_fft = torch.fft.ifft2(fx * fk.conj()).real.sum(2)
    assert torch.allclose(out, ref_fft, atol=1e-9)


# ---------------------------------------------------------------- network
def test_conv1_tiling_rule_c26():
    w = torch.randn(64, 3, 3, 3)
    t = tile_input_kernels(w, 26)
    assert t.shape == (64, 26, 3, 3)
    for r in range(8):
        assert torch.equal(t[:, 3 * r:3 * r + 3], w)
    assert torch.equal(t[:, 24:26], w[:, :2])
    assert torch.equal(tile_input_kernels(w, 2), w[:, :2])


def _net(mode="matlab"):
    torch.manual_seed(0)
    m = ToricFCN32(26, 8, pretrained=False, mode=mode).to(DEV).eval()
    torch.nn.init.normal_(m.score_fr.weight, std=0.01)  # non-trivial output
    return m


@pytest.mark.parametrize("mode", ["matlab", "toric"])
def test_fcn_shape_and_projection(mode):
    m = _net(mode)
    x = torch.randn(1, 26, 128, 128, device=DEV)
    with torch.no_grad():
        y = m(x)
    assert y.shape == (1, 8, 128, 128)
    assert torch.equal(torch.rot90(y, 1, dims=(-2, -1)), y)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="512 forward needs GPU")
def test_fcn_shape_512():
    m = _net()
    with torch.no_grad():
        y = m(torch.randn(1, 26, 512, 512, device=DEV))
    assert y.shape == (1, 8, 512, 512)


def _shift_err(m, H):
    x = torch.randn(1, 26, H, H, device=DEV)
    with torch.no_grad():
        y = m.forward_unprojected(x)
        ys = m.forward_unprojected(torch.roll(x, (32, 32), (-2, -1)))
    e = (ys - torch.roll(y, (32, 32), (-2, -1))).abs()
    return e, y.std()


def test_toric_mode_exact_shift_equivariance():
    H = 256 if DEV == "cuda" else 128
    e, s = _shift_err(_net("toric"), H)
    assert e.max() / s < 1e-3


def test_matlab_mode_shift_equivariance_interior_only():
    # pad-100 window of the periodic image is smaller than the receptive field,
    # so the faithful MATLAB geometry is equivariant only away from the border.
    H = 256 if DEV == "cuda" else 128
    e, s = _shift_err(_net("matlab"), H)
    c = H // 2
    assert e[..., c - 32:c + 32, c - 32:c + 32].max() / s < 1e-3


# ---------------------------------------------------------------- loss / metrics
def test_class_scaled_cross_entropy_manual():
    torch.manual_seed(0)
    logits = torch.randn(1, 3, 2, 2, dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([[[0, 1], [1, -1]]])
    prior = 4000.0
    loss = class_scaled_cross_entropy(logits, labels, prior=prior)
    ls = torch.log_softmax(logits, 1)[0]
    n = {0: 1, 1: 2}
    manual = sum(-ls[labels[0, i, j], i, j] / (prior + n[int(labels[0, i, j])])
                 for i in range(2) for j in range(2) if labels[0, i, j] >= 0)
    assert torch.allclose(loss, manual)
    loss.backward()
    assert logits.grad[0, :, 1, 1].abs().sum() == 0  # ignored pixel


def test_confusion_meter():
    cm = ConfusionMeter(3)
    labels = torch.tensor([[0, 0, 1, 1, 2, -1]])
    preds = torch.tensor([[0, 1, 1, 1, 0, 2]])
    cm.update(preds, labels)
    # confusion rows=gt: [[1,1,0],[0,2,0],[1,0,0]]
    assert math.isclose(cm.pixel_acc, 3 / 5)
    assert math.isclose(cm.mean_acc, (1 / 2 + 1 + 0) / 3)
    # IoU: c0 tp1 pos2 res2 -> 1/3; c1 tp2 pos2 res3 -> 2/3; c2 0
    assert math.isclose(cm.mean_iou, (1 / 3 + 2 / 3 + 0) / 3)
    # logits path accumulates
    logits = F.one_hot(preds.clamp(min=0), 3).permute(0, 2, 1).float()
    cm.update(logits, labels)
    assert torch.equal(cm.confusion, 2 * torch.tensor([[1., 1, 0], [0, 2, 0], [1, 0, 0]], dtype=torch.float64))
    cm.reset()
    assert cm.confusion.sum() == 0 and cm.pixel_acc == 0
