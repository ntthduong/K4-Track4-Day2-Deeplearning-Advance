"""CPU tests for mathematical invariants; run python -m unittest test_core -v."""
import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
from torch import nn
from losses import FocalLoss, LabelSmoothingCE, mix_batch, class_weights
from inference import fuse_conv_bn, fit_temperature, apply_temperature, aggregate_views
from model import param_groups
from train import Config, EMA, build_optimizer, build_scheduler, train_one_epoch, evaluate, parse_overrides

class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        np.random.seed(4)
    def test_focal_zero_and_smoothing_zero_equal_ce(self):
        z, y = torch.randn(16, 9), torch.randint(9, (16,))
        expected = nn.CrossEntropyLoss()(z, y)
        torch.testing.assert_close(FocalLoss(0)(z, y), expected, atol=1e-6, rtol=0)
        torch.testing.assert_close(LabelSmoothingCE(0)(z, y), expected)
    def test_cutmix_lambda_matches_actual_area(self):
        x = torch.stack([torch.full((3, 11, 13), float(i)) for i in range(8)])
        y = torch.arange(8)
        mixed, (a, b, lam) = mix_batch(x, y)
        for i in range(8):
            if a[i] != b[i]:
                fraction = (mixed[i] != x[i]).float().mean().item()
                self.assertAlmostEqual(fraction, 1-lam, places=6)
        torch.testing.assert_close(x[:, 0, 0, 0], y.float())
    def test_fusion_preserves_output_and_original(self):
        model = nn.Sequential(nn.Conv2d(3, 5, 3), nn.BatchNorm2d(5), nn.ReLU()).eval()
        model[1].running_mean.copy_(torch.randn(5))
        model[1].running_var.copy_(torch.rand(5)+.2)
        x = torch.randn(2, 3, 8, 8)
        fused = fuse_conv_bn(model)
        torch.testing.assert_close(model(x), fused(x), atol=1e-5, rtol=1e-5)
        self.assertIsInstance(model[1], nn.BatchNorm2d)
        self.assertIsInstance(fused[1], nn.Identity)
    def test_temperature_does_not_change_argmax_and_reduces_nll(self):
        rng = np.random.default_rng(2)
        z, y = rng.normal(size=(100, 9))*5, rng.integers(9, size=100)
        t = fit_temperature(z, y)
        p, before = apply_temperature(z, t), apply_temperature(z, 1)
        np.testing.assert_array_equal(p.argmax(1), z.argmax(1))
        self.assertLessEqual(-np.log(p[np.arange(100), y]).mean(), -np.log(before[np.arange(100), y]).mean()+1e-8)
    def test_weight_normalization_and_tta(self):
        self.assertAlmostEqual(class_weights(range(1, 10), .999).mean().item(), 1., places=6)
        z = np.random.randn(10, 9)
        for space in ('prob', 'logit'):
            p = aggregate_views([z, z], space)
            np.testing.assert_allclose(p.sum(1), 1)
    def test_parameter_groups_cover_once_and_exclude_bias_decay(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Linear(3, 4)
                self.head = nn.Linear(4, 9)
            def get_classifier(self):
                return self.head
        model = Model()
        groups = param_groups(model, .001, .01, .05)
        ids = [id(p) for g in groups for p in g['params']]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(p) for p in model.parameters()})
        for g in groups:
            for p in g['params']:
                if p.ndim == 1:
                    self.assertEqual(g['weight_decay'], 0)
    def test_training_updates_head_and_keeps_frozen_bn(self):
        class Toy(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4))
                self.head = nn.Linear(4, 9)
            def get_classifier(self):
                return self.head
            def forward(self, x):
                return self.head(self.backbone(x))
        from model import freeze_backbone
        model = Toy()
        freeze_backbone(model)
        before = model.head.weight.detach().clone()
        bn_before = model.backbone[1].running_mean.clone()
        loader = [(torch.randn(8, 3), torch.arange(8), [f'{i}.jpg' for i in range(8)])]
        cfg = Config(init='frozen', epochs=2, amp=False)
        optimizer = build_optimizer(model, cfg)
        scheduler = build_scheduler(optimizer, cfg, len(loader))
        ema = EMA(model, .9)
        scaler = torch.amp.GradScaler('cuda', enabled=False)
        row = train_one_epoch(model, loader, nn.CrossEntropyLoss(), optimizer, scheduler, scaler, cfg, torch.device('cpu'), ema)
        self.assertTrue(np.isfinite(row['train_loss']))
        self.assertFalse(torch.equal(before, model.head.weight))
        torch.testing.assert_close(bn_before, model.backbone[1].running_mean)
        ns, y, z, loss = evaluate(ema.model, loader, nn.CrossEntropyLoss(), torch.device('cpu'))
        self.assertEqual(z.shape, (8, 9))
        self.assertEqual(ns, loader[0][2])
        self.assertTrue(np.isfinite(loss))
    def test_cli_types(self):
        values = parse_overrides(['seed=2', 'amp=false', 'ema_decay=0.99', 'sampler=none'])
        self.assertEqual(values, {'seed': 2, 'amp': False, 'ema_decay': .99, 'sampler': None})
        with self.assertRaises(ValueError):
            parse_overrides(['amp=nope'])

if __name__ == '__main__':
    unittest.main()
