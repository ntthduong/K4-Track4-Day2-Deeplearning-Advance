"""Validation-only method selection; temperature learned on validation."""
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def softmax(logits):
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def predict_logits(model, loader, device, view=None):
    model.eval()
    names, labels, logits = [], [], []
    with torch.inference_mode():
        for x, y, filenames in loader:
            x = x.to(device=device, dtype=next(model.parameters()).dtype)
            logits.append(model(x if view is None else view(x)).float().cpu().numpy())
            labels.append(y.numpy())
            names.extend(filenames)
    if not labels:
        raise ValueError('Empty evaluation loader')
    return names, np.concatenate(labels), np.concatenate(logits)

def view_identity(x):
    return x

def view_hflip(x):
    return x.flip(-1)

def views_multicrop(x, crop):
    h, w = x.shape[-2:]
    if not 0 < crop <= min(h, w):
        raise ValueError('Crop must fit inside input')
    return [x[..., y:y+crop, z:z+crop] for y, z in
            ((0, 0), (0, w-crop), (h-crop, 0), (h-crop, w-crop), ((h-crop)//2, (w-crop)//2))]

def views_multiscale(x, sizes):
    return [F.interpolate(x, size=(s, s), mode='bilinear', align_corners=False, antialias=True) for s in sizes]

def aggregate_views(logits_per_view, space='prob'):
    if not logits_per_view:
        raise ValueError('At least one view is required')
    z = np.stack(logits_per_view)
    if space == 'prob':
        return np.mean([softmax(v) for v in z], axis=0)
    if space == 'logit':
        return softmax(z.mean(axis=0))
    raise ValueError('space must be prob or logit')

def ensemble_probs(list_of_probs):
    p = np.stack(list_of_probs).mean(axis=0)
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1, atol=1e-5):
        raise ValueError('Ensemble requires normalized probabilities')
    return p

def fit_temperature(val_logits, val_labels):
    z = torch.as_tensor(val_logits, dtype=torch.float64)
    y = torch.as_tensor(val_labels, dtype=torch.long)
    if z.ndim != 2 or len(z) != len(y) or not len(y) or not torch.isfinite(z).all():
        raise ValueError('Invalid validation logits/labels')
    # Bounded scalar search is deterministic and cannot diverge like unconstrained LBFGS.
    a, b = np.log(.05), np.log(20.)
    ratio = (np.sqrt(5) - 1) / 2
    def objective(logt):
        return F.cross_entropy(z / np.exp(logt), y).item()
    c, d = b-ratio*(b-a), a+ratio*(b-a)
    fc, fd = objective(c), objective(d)
    for _ in range(60):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b-ratio*(b-a)
            fc = objective(c)
        else:
            a, c, fc = c, d, fd
            d = a+ratio*(b-a)
            fd = objective(d)
    candidates = [0., a, b, (a+b)/2]
    return float(np.exp(min(candidates, key=objective)))

def apply_temperature(logits, T):
    if not np.isfinite(T) or T <= 0:
        raise ValueError('Temperature must be finite and positive')
    return softmax(np.asarray(logits) / T)

def fuse_conv_bn(model):
    fused = copy.deepcopy(model).eval()
    def visit(module):
        # Only fuse pairs with established forward adjacency: Sequential and timm ResNet blocks.
        if isinstance(module, nn.Sequential):
            children = list(module._modules.items())
            for (a, conv), (b, bn) in zip(children, children[1:]):
                if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
                    module._modules[a] = torch.nn.utils.fuse_conv_bn_eval(conv, bn)
                    module._modules[b] = nn.Identity()
        if module.__class__.__module__.startswith('timm.models.resnet'):
            for a, b in (('conv1', 'bn1'), ('conv2', 'bn2'), ('conv3', 'bn3')):
                conv, bn = getattr(module, a, None), getattr(module, b, None)
                if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
                    setattr(module, a, torch.nn.utils.fuse_conv_bn_eval(conv, bn))
                    setattr(module, b, nn.Identity())
        for child in module.children():
            visit(child)
    visit(fused)
    return fused
