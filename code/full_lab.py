"""Run all required stages, making every selection on validation only."""
import json
import shutil
from dataclasses import asdict, replace
from pathlib import Path
import numpy as np
import pandas as pd
from train import run, run_dir
from experiments import (train_suite, restore_config, TRAINING_RECIPES,
                         inference_suite, final_inference, export_results)
from dataset import make_loader, build_transforms
from eval import main as eval_main, compute_metrics


def import_previous_output(base):
    """Only import backbone runs; never import final test files from another run."""
    sources = sorted(Path('/kaggle/input').rglob('backbones_results.csv'))
    if len(sources) > 1:
        raise ValueError('Attach only one previous DeepWeeds notebook output')
    if not sources:
        print('No previous output attached: backbone suite will train from scratch.', flush=True)
        return
    source = sources[0].parent
    for exp in range(1, 6):
        relative = Path('runs') / f'B{exp:02}' / 'seed0'
        if not (source / relative / 'summary.json').exists():
            raise ValueError(f'Incomplete previous backbone: {relative}')
        if not relative.exists():
            shutil.copytree(source / relative, relative)
            config_path = relative / 'config.json'
            metadata = json.loads(config_path.read_text())
            metadata['imported_from'] = str(source / relative)
            metadata['original_data_paths'] = {key: metadata[key] for key in ('images_dir', 'labels_dir')}
            metadata.update(images_dir=base.images_dir, labels_dir=base.labels_dir)
            config_path.write_text(json.dumps(metadata, indent=2))
    for curve in (source / 'curves').glob('B*.png'):
        Path('curves').mkdir(exist_ok=True)
        if not (Path('curves') / curve.name).exists():
            shutil.copy2(curve, Path('curves') / curve.name)
    print('Imported previous five backbone checkpoints:', source, flush=True)


def reuse_run(source_cfg, target_cfg):
    """Reuse an identical seed/recipe with explicit provenance."""
    a, b = asdict(source_cfg), asdict(target_cfg)
    for key in a:
        if key != 'exp_id' and a[key] != b[key]:
            raise ValueError(f'Cannot reuse a different recipe: {key}')
    target = run_dir(target_cfg)
    if target.exists():
        return
    shutil.copytree(run_dir(source_cfg), target)
    meta = json.loads((target / 'config.json').read_text())
    meta.update(asdict(target_cfg), reused_from=str(run_dir(source_cfg)))
    (target / 'config.json').write_text(json.dumps(meta, indent=2))
    summary = json.loads((target / 'summary.json').read_text())
    summary.update(exp_id=target_cfg.exp_id, checkpoint=str(target / 'best.pt'),
                   reused_from=str(run_dir(source_cfg)))
    (target / 'summary.json').write_text(json.dumps(summary, indent=2))
    from train import plot_curves
    history = pd.read_csv(target / 'history.csv').to_dict('records')
    plot_curves(history, Path(target_cfg.curves_dir) / f'{target_cfg.exp_id}_seed{target_cfg.seed}.png',
                f'{target_cfg.exp_id} / reused identical seed and recipe')


def loaders(cfg, frames):
    data = json.loads((run_dir(cfg) / 'config.json').read_text())['data_cfg']
    transform = build_transforms(False, cfg.img_size, mean=data['mean'], std=data['std'])
    return [make_loader(df, cfg.images_dir, transform, cfg.batch_size, False,
                        num_workers=cfg.num_workers) for df in frames]


def run_all(base, val_df, test_df):
    import_previous_output(base)
    backbones = train_suite(base, 'backbones')
    winner = backbones.sort_values(['macro_f1', 'exp_id'], ascending=[False, True]).iloc[0]
    source = restore_config(winner.exp_id, base.seed)
    selected = replace(base, backbone=source.backbone)
    reuse_run(source, replace(selected, exp_id='T00'))
    print('Selected backbone on validation:', selected.backbone, flush=True)
    training = train_suite(selected, 'training')
    baseline_f1 = float(training.loc[training.exp_id == 'T00', 'macro_f1'].iloc[0])
    # Choose at most one winning value per axis, without combining incompatible losses/mixes.
    changes = {}
    axes = [('T03_color',), ('T04_cutmix', 'T05_mixup'),
            ('T06_ls', 'T07_focal', 'T08_weighted'), ('T09_ema',)]
    for ids in axes:
        best_axis = training[training.exp_id.isin(ids)].sort_values(
            ['macro_f1', 'exp_id'], ascending=[False, True]).iloc[0]
        if best_axis.macro_f1 > baseline_f1:
            changes.update(TRAINING_RECIPES[best_axis.exp_id])
    combined = replace(selected, exp_id='T10_combined', **changes)
    if not (run_dir(combined) / 'summary.json').exists():
        if changes:
            run(combined)
        else:
            reuse_run(replace(selected, exp_id='T00'), combined)
    combined_row = json.loads((run_dir(combined) / 'summary.json').read_text())
    training = pd.concat([training, pd.DataFrame([combined_row])], ignore_index=True)
    training.to_csv('training_results.csv', index=False)
    best_row = training.sort_values(['macro_f1', 'exp_id'], ascending=[False, True]).iloc[0]
    best = restore_config(best_row.exp_id, base.seed)
    inference = inference_suite(best, loaders(best, [val_df])[0])
    inference[['exp_id', 'gpu', 'dtype', 'batch', 'p50', 'p95', 'p99', 'images_per_s']].to_csv(
        'latency_results.csv', index=False)
    import matplotlib.pyplot as plt
    ax = inference.plot.scatter(x='p95', y='macro_f1', title='Validation F1 vs latency')
    for _, row in inference.iterrows():
        ax.annotate(row.exp_id, (row.p95, row.macro_f1))
    ax.figure.savefig('curves/inference_tradeoff.png', dpi=150)
    plt.close(ax.figure)
    # Lock a supported final method before touching test. Prefer low ECE then latency on F1 ties.
    eligible = inference[inference.exp_id.isin(['I00', 'I01_hflip_prob', 'I02_hflip_logit', 'I03_temperature'])]
    choice = eligible.sort_values(['macro_f1', 'ece', 'p95', 'exp_id'],
                                  ascending=[False, True, True, True]).iloc[0]
    method = {'I00': 'identity', 'I01_hflip_prob': 'hflip_prob',
              'I02_hflip_logit': 'hflip_logit', 'I03_temperature': 'identity'}[choice.exp_id]
    lock = {'recipe': asdict(best), 'method': method, 'calibrated': True,
            'inference_choice': choice.exp_id, 'seeds': [0, 1, 2],
            'selection_split': 'val', 'test_used_for_selection': False}
    lock_path = Path('selection.json')
    if lock_path.exists() and json.loads(lock_path.read_text()) != lock:
        raise ValueError('Selection is already locked; do not change after test')
    lock_path.write_text(json.dumps(lock, indent=2))
    print('Locked selection:', lock, flush=True)
    rows = []
    for seed in (0, 1, 2):
        for original, exp_id, infer_method, calibrated in (
                (best, 'F01', method, True),
                (replace(selected, exp_id='T00'), 'T00_final', 'identity', False)):
            cfg = replace(original, exp_id=exp_id, seed=seed, save_test_predictions=False)
            if not (run_dir(cfg) / 'summary.json').exists():
                if seed == base.seed:
                    reuse_run(original, cfg)
                else:
                    run(cfg)
            result_path = run_dir(cfg) / 'final_metrics.json'
            prediction_path = Path(cfg.pred_dir) / f'{exp_id}_seed{seed}_test.csv'
            if result_path.exists():
                result = json.loads(result_path.read_text())
            elif prediction_path.exists():
                # Recover metrics from saved predictions without rerunning test inference.
                df = pd.read_csv(prediction_path)
                metrics = compute_metrics(df.y_true.to_numpy(), df.y_pred.to_numpy(),
                                          df[[f'p{i}' for i in range(9)]].to_numpy())
                result = {k: metrics[k] for k in ('top1', 'macro_f1', 'ece', 'balanced_acc')}
                result.update(exp_id=exp_id, seed=seed, method=infer_method)
                result_path.write_text(json.dumps(result, indent=2))
            else:
                result = final_inference(cfg, *loaders(cfg, [val_df, test_df]),
                                         method=infer_method, calibrated=calibrated)
                result_path.write_text(json.dumps(result, indent=2))
            rows.append(result)
            pd.DataFrame(rows).to_csv('final_results.csv', index=False)
    for tag in ('F01', 'T00_final'):
        assert eval_main(['score', '--pred', f'predictions/{tag}_seed*_test.csv',
                          '--test-csv', f'{base.labels_dir}/test_subset0.csv',
                          '--labels', f'{base.labels_dir}/labels.csv', '--tag', tag,
                          '--out', 'eval_out']) == 0
    assert eval_main(['grade', '--final', 'predictions/F01_seed*_test.csv',
                      '--baseline', 'predictions/T00_final_seed*_test.csv',
                      '--uncal', 'predictions/F01uncal_seed*_test.csv',
                      '--final-val', 'predictions/F01_seed*_val.csv',
                      '--test-csv', f'{base.labels_dir}/test_subset0.csv',
                      '--val-csv', f'{base.labels_dir}/val_subset0.csv',
                      '--latency-p95-ms', str(choice.p95), '--out', 'eval_out']) == 0
    pd.read_csv('eval_out/F01_per_class.csv').to_csv('per_class_results.csv', index=False)
    export_results()
    final = pd.DataFrame(rows)
    stats = final.groupby('exp_id')[['macro_f1', 'top1', 'ece', 'balanced_acc']].agg(['mean', 'std'])
    stats.to_csv('final_mean_std.csv')
    # Render confusion matrices from the actual saved predictions.
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    for seed in (0, 1, 2):
        df = pd.read_csv(f'predictions/F01_seed{seed}_test.csv')
        fig, ax = plt.subplots(figsize=(8, 8))
        ConfusionMatrixDisplay(confusion_matrix(df.y_true, df.y_pred, labels=list(range(9)))).plot(ax=ax)
        fig.savefig(f'curves/F01_seed{seed}_confusion.png', dpi=150)
        plt.close(fig)
    report = ('# DeepWeeds — kết quả chạy đầy đủ\n\n'
              'Fold 0 chính thức; ảnh 224×224; batch 64; 12 epoch; AMP; chọn checkpoint bằng macro-F1 val. '
              'Chọn backbone, recipe và suy luận chỉ trên val; test một lần cho mỗi cấu hình/seed.\n\n'
              f'Backbone đã chọn: **{selected.backbone}**. Recipe thắng: **{best.exp_id}**. '
              f'Suy luận: **{method}**, temperature khớp riêng trên val của từng seed.\n\n'
              '## Backbone (validation)\n\n' + backbones.to_csv(index=False) + '\n'
              '## Ablation (validation)\n\n' + training.to_csv(index=False) + '\n'
              '## Suy luận (validation)\n\n' + inference.to_csv(index=False) + '\n'
              '## Chung kết — mean và std mẫu qua 3 seed\n\n' + stats.to_string() + '\n\n'
              '## Hạn chế\n\n12 epoch có thể chưa đủ cho scratch. Dữ liệu mất cân bằng; '
              'macro-F1 được ưu tiên hơn accuracy. Split ngẫu nhiên chưa đo tổng quát sang địa điểm mới. '
              'Độ trễ chỉ gồm forward, chưa gồm đọc ảnh và gộp xác suất. '
              'ConvNeXt dùng LayerNorm nên phép gộp Conv-BN có thể là no-op; FP16 được đo riêng. '
              'GMAC từ thop có thể thiếu phép toán attention. Seed 0 được tái sử dụng khi recipe giống hệt, '
              'nguồn ghi trong config.json; không phải một lần chạy độc lập bổ sung.\n')
    Path('report.md').write_text(report, encoding='utf-8')
    Path('COMPLETE.json').write_text(json.dumps({'status': 'complete', 'final_runs': len(rows),
                                              'selection': lock}, indent=2))
    print(stats, flush=True)
    print('ALL STAGES COMPLETE: results.xlsx, report.md, predictions/, curves/, eval_out/', flush=True)
    return final
