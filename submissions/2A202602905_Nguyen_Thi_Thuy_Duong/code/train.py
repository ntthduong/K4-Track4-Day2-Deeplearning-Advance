"""One configurable training loop for B/T/F experiments; test is opt-in."""
from dataclasses import dataclass, asdict, fields
from pathlib import Path
import argparse
import copy
import json
import math
import platform
import random
import time
import sys
import numpy as np
import pandas as pd
import torch
from model import build_model, param_groups, count_params, count_gmacs
from losses import build_criterion, class_weights, mix_batch, mixed_loss
from inference import softmax
if not (Path(__file__).parent / 'eval.py').exists():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval import compute_metrics, save_predictions

@dataclass
class Config:
    exp_id: str = 'T00'
    seed: int = 0
    fold: int = 0
    backbone: str = 'resnet50'
    init: str = 'finetune'
    drop_rate: float = 0.
    img_size: int = 224
    aug: str = 'basic'
    sampler: str | None = None
    mix: str | None = None
    mix_alpha: float = 1.
    loss: str = 'ce'
    label_smoothing: float = .1
    focal_gamma: float = 2.
    class_weight_beta: float | None = None
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = .05
    warmup_epochs: float = 1.
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    images_dir: str = 'data/images'
    labels_dir: str = 'data/labels'
    out_dir: str = 'runs'
    pred_dir: str = 'predictions'
    curves_dir: str = 'curves'
    save_test_predictions: bool = False
    pretrained: bool = True

def run_dir(cfg):
    return Path(cfg.out_dir) / cfg.exp_id / f'seed{cfg.seed}'

def pred_path(cfg, split):
    if split not in ('val', 'test'):
        raise ValueError('split must be val or test')
    return Path(cfg.pred_dir) / f'{cfg.exp_id}_seed{cfg.seed}_{split}.csv'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def build_optimizer(model, cfg):
    return torch.optim.AdamW(param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay))

def build_scheduler(optimizer, cfg, steps_per_epoch):
    total = cfg.epochs * steps_per_epoch
    warmup = min(round(cfg.warmup_epochs * steps_per_epoch), total)
    def factor(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = min(1., (step - warmup) / max(1, total - warmup))
        return .5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)

class EMA:
    def __init__(self, model, decay):
        if not 0 <= decay < 1:
            raise ValueError('EMA decay must be in [0,1)')
        self.model, self.decay = copy.deepcopy(model).eval(), decay
        self.model.requires_grad_(False)
    @torch.no_grad()
    def update(self, model):
        current = dict(model.named_parameters())
        for name, p in self.model.named_parameters():
            p.lerp_(current[name].detach(), 1-self.decay)
        # Copy BN buffers from the live model; do not average integer counters.
        buffers = dict(model.named_buffers())
        for name, b in self.model.named_buffers():
            b.copy_(buffers[name])

def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg, device, ema=None):
    model.train()
    if cfg.init == 'frozen':
        model.eval()
        model.get_classifier().train()
    total, n = 0., 0
    for x, y, _ in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        targets = y
        if cfg.mix:
            x, targets = mix_batch(x, y, cfg.mix_alpha, cfg.mix)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, enabled=cfg.amp and device.type == 'cuda'):
            logits = model(x)
            loss = mixed_loss(criterion, logits, targets) if cfg.mix else criterion(logits, targets)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite training loss')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        # A scaler decrease means the optimizer step was skipped for overflow.
        if scaler.get_scale() >= old_scale:
            scheduler.step()
            if ema:
                ema.update(model)
        total += loss.item() * len(y)
        n += len(y)
    if n == 0:
        raise ValueError('Empty training loader')
    return {'train_loss': total/n, 'lr': optimizer.param_groups[0]['lr']}

def evaluate(model, loader, criterion, device):
    model.eval()
    names, ys, zs, total = [], [], [], 0.
    with torch.inference_mode():
        for x, y, filenames in loader:
            x, y = x.to(device), y.to(device)
            z = model(x)
            total += criterion(z, y).item() * len(y)
            names.extend(filenames)
            ys.append(y.cpu().numpy())
            zs.append(z.float().cpu().numpy())
    if not ys:
        raise ValueError('Empty evaluation loader')
    return names, np.concatenate(ys), np.concatenate(zs), total/len(names)

def plot_curves(history, path, title):
    import matplotlib.pyplot as plt
    df = pd.DataFrame(history)
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    axs[0].plot(df.epoch, df.train_loss, label='train')
    axs[0].plot(df.epoch, df.val_loss, label='val')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[1].plot(df.epoch, df.macro_f1, label='val macro-F1')
    axs[1].plot(df.epoch, df.top1, label='val top-1')
    axs[1].legend()
    axs[2].plot(df.epoch, df.lr)
    axs[2].set_ylabel('Backbone LR')
    for ax in axs:
        ax.set_xlabel('Epoch')
        ax.grid(alpha=.2)
    fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)

def run(cfg):
    from dataset import load_split, check_split, make_loader, build_transforms
    import timm
    if cfg.epochs < 1 or cfg.batch_size < 1:
        raise ValueError('epochs and batch_size must be positive')
    target = run_dir(cfg)
    if (target / 'history.csv').exists():
        raise FileExistsError(f'Run already exists: {target}; choose another exp_id')
    if cfg.save_test_predictions and pred_path(cfg, 'test').exists():
        raise FileExistsError('Test predictions already exist; do not repeat final evaluation')
    set_seed(cfg.seed)
    target.mkdir(parents=True, exist_ok=True)
    train, val, test = load_split(cfg.labels_dir, cfg.fold)
    checks = check_split(train, val, test, cfg.images_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(cfg.backbone, cfg.pretrained, drop_rate=cfg.drop_rate, init=cfg.init)
    data_cfg = timm.data.resolve_model_data_config(model)
    metadata = {**asdict(cfg), 'python': platform.python_version(), 'torch': str(torch.__version__),
                'timm': timm.__version__, 'pretrained_cfg': model.pretrained_cfg, 'data_cfg': data_cfg,
                'split_checks': checks, 'device': str(device), 'gmac_tool': 'thop (attention ops may be incomplete)'}
    (target / 'config.json').write_text(json.dumps(metadata, indent=2, default=str), encoding='utf-8')
    transforms = {s: build_transforms(s == 'train', cfg.img_size, cfg.aug, data_cfg['mean'], data_cfg['std'])
                  for s in ('train', 'val')}
    def loader(df, training=False):
        return make_loader(df, cfg.images_dir, transforms['train' if training else 'val'], cfg.batch_size,
                           training, cfg.sampler if training else None, cfg.num_workers, cfg.seed)
    train_loader, val_loader = loader(train, True), loader(val)
    params = count_params(model)
    try:
        gmacs = count_gmacs(model, cfg.img_size)
    except (ImportError, RuntimeError, AssertionError) as exc:
        gmacs = None
        print('GMAC unavailable:', exc)
    model = model.to(device)
    weights = None
    if cfg.loss == 'ce_weighted' or cfg.class_weight_beta is not None:
        weights = class_weights(train.Label.value_counts().reindex(range(9), fill_value=0).values,
                                cfg.class_weight_beta or 0.).to(device)
    criterion = build_criterion(cfg.loss, smoothing=cfg.label_smoothing, gamma=cfg.focal_gamma,
                                alpha=weights, weight=weights).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = torch.amp.GradScaler('cuda', enabled=cfg.amp and device.type == 'cuda')
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay is not None else None
    history, best, best_epoch = [], -1., 0
    start = time.perf_counter()
    for epoch in range(1, cfg.epochs+1):
        epoch_start = time.perf_counter()
        row = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, cfg, device, ema)
        selected = ema.model if ema else model
        names, y, z, val_loss = evaluate(selected, val_loader, criterion, device)
        p = softmax(z)
        metric = compute_metrics(y, p.argmax(1), p)
        row.update(epoch=epoch, val_loss=val_loss, macro_f1=metric['macro_f1'], top1=metric['top1'],
                   ece=metric['ece'], seconds=time.perf_counter()-epoch_start)
        history.append(row)
        pd.DataFrame(history).to_csv(target / 'history.csv', index=False)
        if metric['macro_f1'] > best:
            best, best_epoch = metric['macro_f1'], epoch
            torch.save({'model': selected.state_dict(), 'epoch': epoch, 'config': asdict(cfg)}, target / 'best.pt')
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict(), 'ema': ema.model.state_dict() if ema else None,
                    'epoch': epoch, 'config': asdict(cfg)}, target / 'last.pt')
        plot_curves(history, Path(cfg.curves_dir) / f'{cfg.exp_id}_seed{cfg.seed}.png', f'{cfg.exp_id} / {cfg.backbone}')
        print(f"Epoch {epoch}/{cfg.epochs}: loss={row['train_loss']:.4f} val F1={best:.4f} (best)", flush=True)
    model.load_state_dict(torch.load(target / 'best.pt', map_location=device, weights_only=True)['model'])
    for split, df in [('val', val)] + ([('test', test)] if cfg.save_test_predictions else []):
        names, y, z, _ = evaluate(model, loader(df), criterion, device)
        np.savez_compressed(target / f'{split}_logits.npz', filenames=np.array(names), y_true=y, logits=z)
        save_predictions(pred_path(cfg, split), names, y, softmax(z))
    result = {'exp_id': cfg.exp_id, 'backbone': cfg.backbone, 'seed': cfg.seed, 'best_epoch': best_epoch,
              'macro_f1': best, 'top1': history[best_epoch-1]['top1'], 'params_m': params, 'gmacs': gmacs,
              'train_seconds': time.perf_counter()-start, 'seconds_per_epoch': np.mean([r['seconds'] for r in history]),
              'checkpoint': str(target / 'best.pt')}
    (target / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result

def parse_overrides(pairs):
    defaults = asdict(Config())
    optional_numeric = {'ema_decay', 'class_weight_beta'}
    result = {}
    for pair in pairs:
        key, sep, value = pair.partition('=')
        if not sep or key not in defaults:
            raise ValueError(f'Invalid Config override: {pair}')
        default = defaults[key]
        if value.lower() in ('none', 'null'):
            if default is not None:
                raise ValueError(f'{key} does not accept None')
            result[key] = None
        elif isinstance(default, bool):
            if value.lower() not in ('true', 'false'):
                raise ValueError(f'{key} requires true/false')
            result[key] = value.lower() == 'true'
        elif isinstance(default, int):
            result[key] = int(value)
        elif isinstance(default, float) or key in optional_numeric:
            result[key] = float(value)
        else:
            result[key] = value
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set', nargs='*', default=[])
    args = parser.parse_args()
    print(json.dumps(run(Config(**parse_overrides(args.set))), indent=2))

if __name__ == '__main__':
    main()
