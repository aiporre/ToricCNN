"""Loss and metrics: ``SegmentationLossClassDistributionScaling.m`` and
``SegmentationAccuracy.m``."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def class_scaled_cross_entropy(logits: torch.Tensor, labels: torch.Tensor,
                               prior: float = 4000.0,
                               ignore_index: int = -1) -> torch.Tensor:
    """Class-frequency-weighted softmax-log loss with sum reduction.

    ``logits``: (B, K, H, W); ``labels``: int64 (B, H, W), 0-based, pixels equal
    to ``ignore_index`` are ignored. With ``n_c`` the number of (non-ignored)
    pixels of class ``c`` in the whole batch, every pixel gets weight
    ``1 / (prior + n_c[label])`` and the loss is ``sum(weight * CE_pixel)``
    (like ``vl_nnloss(..., 'instanceWeights', mass)``).

    Note: the MATLAB layer used ``prior = 4000`` in forward but ``4`` in its
    backward (so its gradient was not the gradient of its reported loss). We use
    ``prior`` (default 4000) consistently and let autograd differentiate it.
    """
    K = logits.shape[1]
    labels = labels.long()
    valid = labels != ignore_index
    safe = torch.where(valid, labels, torch.zeros_like(labels))
    counts = torch.bincount(safe[valid].reshape(-1), minlength=K).to(logits.dtype)
    weights = 1.0 / (prior + counts)
    w_pix = weights[safe] * valid.to(logits.dtype)
    ce = F.cross_entropy(logits, safe, reduction="none")
    return (w_pix * ce).sum()


class ConfusionMeter:
    """Accumulated confusion matrix with ``SegmentationAccuracy.m`` statistics.

    Rows are ground-truth classes, columns predictions. Labels < 0 are ignored.
    Implements the multi-class path of the MATLAB code (the 2-class
    "head segmentation" special case for pixel accuracy is not reproduced).
    """

    def __init__(self, num_classes: int):
        self.num_classes = int(num_classes)
        self.reset()

    def reset(self) -> None:
        self.confusion = torch.zeros(self.num_classes, self.num_classes, dtype=torch.float64)

    @torch.no_grad()
    def update(self, logits_or_preds: torch.Tensor, labels: torch.Tensor) -> None:
        """``logits_or_preds``: (B, K, H, W) scores or (B, H, W) integer predictions."""
        if logits_or_preds.dim() == labels.dim() + 1:
            preds = logits_or_preds.argmax(dim=1)
        else:
            preds = logits_or_preds
        preds = preds.reshape(-1).long().cpu()
        labels = labels.reshape(-1).long().cpu()
        ok = labels >= 0
        K = self.num_classes
        idx = labels[ok] * K + preds[ok]
        self.confusion += torch.bincount(idx, minlength=K * K).reshape(K, K).double()

    @property
    def _stats(self):
        c = self.confusion
        return c.sum(1), c.sum(0), torch.diag(c)

    @property
    def pixel_acc(self) -> float:
        _, _, tp = self._stats
        return float(tp.sum() / max(1.0, float(self.confusion.sum())))

    @property
    def mean_acc(self) -> float:
        pos, _, tp = self._stats
        return float((tp / pos.clamp(min=1)).mean())

    @property
    def mean_iou(self) -> float:
        pos, res, tp = self._stats
        return float((tp / (pos + res - tp).clamp(min=1)).mean())
