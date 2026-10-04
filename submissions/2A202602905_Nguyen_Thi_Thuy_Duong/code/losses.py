"""Losses, class balancing, Mixup and area-corrected CutMix."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class LabelSmoothingCE(nn.CrossEntropyLoss):
    def __init__(self, smoothing=.1):
        super().__init__(label_smoothing=smoothing)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2., alpha=None):
        super().__init__()
        if gamma < 0:
            raise ValueError('gamma must be nonnegative')
        self.gamma = gamma
        self.register_buffer('alpha', None if alpha is None else torch.as_tensor(alpha, dtype=torch.float32))
    def forward(self, logits, target):
        logpt = F.log_softmax(logits, dim=1).gather(1, target[:, None]).squeeze(1)
        values = -(1 - logpt.exp()).pow(self.gamma) * logpt
        if self.alpha is not None:
            values = values * self.alpha[target]
        return values.mean()

def build_criterion(kind='ce', **kw):
    if kind == 'ce':
        return nn.CrossEntropyLoss()
    if kind == 'ls':
        return LabelSmoothingCE(kw.get('smoothing', .1))
    if kind == 'focal':
        return FocalLoss(kw.get('gamma', 2.), kw.get('alpha'))
    if kind == 'ce_weighted':
        if kw.get('weight') is None:
            raise ValueError('ce_weighted requires train class weights')
        return nn.CrossEntropyLoss(weight=kw['weight'])
    raise ValueError(f'Unknown loss: {kind}')

def class_weights(counts, beta=0.):
    n = torch.as_tensor(counts, dtype=torch.float64)
    if (n <= 0).any() or not 0 <= beta < 1:
        raise ValueError('Counts must be positive and beta in [0, 1)')
    w = 1 / n if beta == 0 else (1 - beta) / (-torch.expm1(n * np.log(beta)))
    return (w / w.mean()).float()

def mix_batch(x, y, alpha=1., mode='cutmix'):
    if mode not in ('mixup', 'cutmix'):
        raise ValueError(f'Unknown mixing mode: {mode}')
    if alpha <= 0:
        return x, (y, y, 1.)
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(len(x), device=x.device)
    if mode == 'mixup':
        mixed = lam * x + (1 - lam) * x[perm]
    else:
        h, w = x.shape[-2:]
        cut_h, cut_w = int(h * np.sqrt(1 - lam)), int(w * np.sqrt(1 - lam))
        cy, cx = np.random.randint(h), np.random.randint(w)
        y1, y2 = max(0, cy - cut_h // 2), min(h, cy + (cut_h + 1) // 2)
        x1, x2 = max(0, cx - cut_w // 2), min(w, cx + (cut_w + 1) // 2)
        mixed = x.clone()
        mixed[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
        lam = 1 - (y2 - y1) * (x2 - x1) / (h * w)
    return mixed, (y, y[perm], lam)

def mixed_loss(criterion, logits, targets):
    a, b, lam = targets
    return lam * criterion(logits, a) + (1 - lam) * criterion(logits, b)
