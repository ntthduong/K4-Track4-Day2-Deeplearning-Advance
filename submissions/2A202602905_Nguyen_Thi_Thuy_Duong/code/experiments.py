"""Reusable validation experiments and gated final inference."""
from dataclasses import replace, fields
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from train import run, run_dir, Config
from model import build_model
from inference import predict_logits, view_hflip, aggregate_views, fit_temperature, apply_temperature, fuse_conv_bn, softmax
from benchmark import latency_report
from eval import compute_metrics, save_predictions

BACKBONES = ['resnet50', 'resnext50_32x4d', 'convnext_tiny', 'deit_small_patch16_224', 'efficientnet_b0']
TRAINING_RECIPES = {
    'T00': {}, 'T01_scratch': {'init': 'scratch'}, 'T02_frozen': {'init': 'frozen'},
    'T03_color': {'aug': 'color'}, 'T04_cutmix': {'mix': 'cutmix'}, 'T05_mixup': {'mix': 'mixup'},
    'T06_ls': {'loss': 'ls', 'label_smoothing': .1}, 'T07_focal': {'loss': 'focal'},
    'T08_weighted': {'loss': 'ce_weighted'}, 'T09_ema': {'ema_decay': .999},
}

def train_suite(base, stage='backbones'):
    configs = ([replace(base, exp_id=f'B{i:02}', backbone=name) for i, name in enumerate(BACKBONES, 1)]
               if stage == 'backbones' else [replace(base, exp_id=k, **v) for k, v in TRAINING_RECIPES.items()])
    if base.save_test_predictions:
        raise ValueError('Validation suites must not evaluate test')
    results = []
    for cfg in configs:
        summary = run_dir(cfg) / 'summary.json'
        if summary.exists():
            # Resume orchestration only if saved configuration matches this experiment.
            old = json.loads((run_dir(cfg)/'config.json').read_text())
            from dataclasses import asdict
            if any(old.get(k) != v for k, v in asdict(cfg).items()):
                raise ValueError(f'Existing run has another configuration: {cfg.exp_id}')
            results.append(json.loads(summary.read_text()))
        else:
            results.append(run(cfg))
        pd.DataFrame(results).to_csv(f'{stage}_results.csv', index=False)
    return pd.DataFrame(results)

def load_checkpoint(cfg, device='cuda'):
    model = build_model(cfg.backbone, pretrained=False, drop_rate=cfg.drop_rate, init='scratch')
    state = torch.load(run_dir(cfg)/'best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(state['model'])
    return model.to(device).eval()

def restore_config(exp_id, seed=0, out_dir='runs'):
    """Recover the full recipe instead of guessing from a checkpoint name."""
    saved = json.loads((Path(out_dir)/exp_id/f'seed{seed}'/'config.json').read_text())
    return Config(**{field.name: saved[field.name] for field in fields(Config)})

def inference_suite(cfg, val_loader, device='cuda'):
    model = load_checkpoint(cfg, device)
    names, y, z = predict_logits(model, val_loader, device)
    flipped_names, flipped_y, flipped = predict_logits(model, val_loader, device, view_hflip)
    if names != flipped_names or not np.array_equal(y, flipped_y):
        raise ValueError('View filenames/labels differ')
    T = fit_temperature(z, y)
    candidates = {'I00': (softmax(z), 1, 'fp32'),
                  'I01_hflip_prob': (aggregate_views([z, flipped], 'prob'), 2, 'fp32'),
                  'I02_hflip_logit': (aggregate_views([z, flipped], 'logit'), 2, 'fp32'),
                  'I03_temperature': (apply_temperature(z, T), 1, 'fp32')}
    fused = fuse_conv_bn(model)
    fn, fy, fz = predict_logits(fused, val_loader, device)
    if fn != names or not np.array_equal(fy, y):
        raise ValueError('Fused model filenames differ')
    fusion_error = float(np.max(np.abs(z-fz)))
    candidates['I04_fused'] = (softmax(fz), 1, 'fp32')
    import copy
    half_model = copy.deepcopy(model).half()
    hn, hy, hz = predict_logits(half_model, val_loader, device)
    if hn != names or not np.array_equal(hy, y):
        raise ValueError('FP16 filenames/labels differ')
    candidates['I05_fp16'] = (softmax(hz), 1, 'fp16')
    del half_model
    rows = []
    for exp_id, (p, k, dtype) in candidates.items():
        metrics = compute_metrics(y, p.argmax(1), p)
        timings = latency_report(fused if exp_id == 'I04_fused' else model, 1, cfg.img_size,
                                 dtype=dtype, device=device, k_views=k)
        rows.append({'exp_id': exp_id, 'checkpoint': str(run_dir(cfg)/'best.pt'),
                     **{key: metrics[key] for key in ('macro_f1', 'top1', 'ece')}, **timings,
                     'temperature': T if exp_id == 'I03_temperature' else 1., 'fusion_max_abs': fusion_error})
        save_predictions(Path(cfg.pred_dir)/f'{exp_id}_seed{cfg.seed}_val.csv', names, y, p)
    result = pd.DataFrame(rows)
    result.to_csv('inference_results.csv', index=False)
    return result

def final_inference(cfg, val_loader, test_loader, method='identity', calibrated=True, device='cuda'):
    """Call only after locking the recipe on val. One test pass per view and seed."""
    if method not in ('identity', 'hflip_prob', 'hflip_logit'):
        raise ValueError('Unknown final method')
    path = Path(cfg.pred_dir)/f'{cfg.exp_id}_seed{cfg.seed}_test.csv'
    if path.exists():
        raise FileExistsError('Final predictions already exist; do not repeat test selection')
    model = load_checkpoint(cfg, device)
    def collect(loader):
        names, y, z = predict_logits(model, loader, device)
        if method == 'identity':
            return names, y, z
        ns, ys, flip = predict_logits(model, loader, device, view_hflip)
        if names != ns or not np.array_equal(y, ys):
            raise ValueError('Mismatched TTA order')
        # Prob-space aggregation represented as log probabilities for scalar calibration.
        return names, y, (z+flip)/2 if method == 'hflip_logit' else np.log(aggregate_views([z, flip], 'prob').clip(1e-12))
    vn, vy, vz = collect(val_loader)
    T = fit_temperature(vz, vy) if calibrated else 1.
    names, y, z = collect(test_loader)
    p = apply_temperature(z, T)
    save_predictions(path, names, y, p)
    save_predictions(Path(cfg.pred_dir)/f'{cfg.exp_id}uncal_seed{cfg.seed}_test.csv', names, y, softmax(z))
    save_predictions(Path(cfg.pred_dir)/f'{cfg.exp_id}_seed{cfg.seed}_val.csv', vn, vy, apply_temperature(vz, T))
    result = {key: value for key, value in compute_metrics(y, p.argmax(1), p).items()
              if key in ('top1', 'macro_f1', 'ece', 'balanced_acc')}
    result.update(exp_id=cfg.exp_id, seed=cfg.seed, temperature=T, method=method)
    return result

def export_results(path='results.xlsx'):
    # Only real measured results are exported; empty sheets never invent scores.
    mapping = {'Backbones': 'backbones_results.csv', 'Training': 'training_results.csv',
               'Inference': 'inference_results.csv', 'Final': 'final_results.csv',
               'PerClass': 'per_class_results.csv', 'Latency': 'latency_results.csv'}
    tables = {name: pd.read_csv(file) if Path(file).exists() else pd.DataFrame() for name, file in mapping.items()}
    sources = [df.assign(stage=name) for name, df in tables.items() if 'macro_f1' in df and name != 'Final']
    summary = pd.concat(sources, ignore_index=True) if sources else pd.DataFrame()
    tables['Summary'] = summary.sort_values('macro_f1', ascending=False).head(10) if len(summary) else pd.DataFrame()
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        for name, table in tables.items():
            table.to_excel(writer, sheet_name=name, index=False)
            ws = writer.sheets[name]
            ws.freeze_panes = 'A2'
            if len(table):
                ws.auto_filter.ref = ws.dimensions
            for column in ws.columns:
                ws.column_dimensions[column[0].column_letter].width = min(45, max(14, max(len(str(c.value or '')) for c in column)+2))
    return path
