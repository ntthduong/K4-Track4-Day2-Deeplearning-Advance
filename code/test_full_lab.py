"""Exercise orchestration with synthetic metrics, without GPU training."""
import json
import os
import sys
import tempfile
import unittest
import types
import io
from contextlib import redirect_stdout
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
try:
    import torchvision
except ImportError:
    # This orchestration test does not read images; GPU notebook tests use the real dataset module.
    dataset_stub = types.ModuleType('dataset')
    dataset_stub.make_loader = lambda *a, **k: None
    dataset_stub.build_transforms = lambda *a, **k: None
    sys.modules['dataset'] = dataset_stub
import full_lab
from train import Config, run_dir
from experiments import TRAINING_RECIPES
from eval import save_predictions


class FullLabTests(unittest.TestCase):
    def test_selection_before_test_and_no_repeat_test(self):
        initial = Path.cwd()
        with tempfile.TemporaryDirectory(dir=initial) as tmp:
            os.chdir(tmp)
            try:
                Path('data/labels').mkdir(parents=True)
                Path('curves').mkdir()
                labels = pd.DataFrame({'Filename': [f'x{i}.jpg' for i in range(9)],
                                       'Label': range(9), 'Species': [str(i) for i in range(9)]})
                for filename in ('labels.csv', 'val_subset0.csv', 'test_subset0.csv'):
                    labels.to_csv(Path('data/labels') / filename, index=False)
                base = Config()
                calls = []
                def fake_run(cfg):
                    directory = run_dir(cfg)
                    directory.mkdir(parents=True, exist_ok=True)
                    meta = asdict(cfg)
                    meta['data_cfg'] = {'mean': [0, 0, 0], 'std': [1, 1, 1]}
                    (directory / 'config.json').write_text(json.dumps(meta))
                    (directory / 'best.pt').write_bytes(b'fixture')
                    pd.DataFrame([{'epoch': 1}]).to_csv(directory / 'history.csv', index=False)
                    row = {'exp_id': cfg.exp_id, 'backbone': cfg.backbone, 'seed': cfg.seed,
                           'macro_f1': .97 if cfg.exp_id == 'T04_cutmix' else .9,
                           'checkpoint': str(directory / 'best.pt')}
                    (directory / 'summary.json').write_text(json.dumps(row))
                    return row
                def fake_suite(cfg, stage):
                    configs = ([replace(cfg, exp_id='B01', backbone='convnext_tiny')]
                               if stage == 'backbones' else
                               [replace(cfg, exp_id=k, **v) for k, v in TRAINING_RECIPES.items()])
                    table = pd.DataFrame([fake_run(c) for c in configs])
                    table.to_csv(f'{stage}_results.csv', index=False)
                    return table
                def fake_inference(cfg, loader):
                    table = pd.DataFrame([dict(exp_id=exp, macro_f1=.97 if 'hflip_prob' in exp else .95,
                                               ece=.05, p95=10., p50=9., p99=11., images_per_s=100.,
                                               dtype='fp32', batch=1, gpu='fixture')
                                          for exp in ('I00', 'I01_hflip_prob', 'I02_hflip_logit', 'I03_temperature')])
                    table.to_csv('inference_results.csv', index=False)
                    return table
                def fake_final(cfg, *args, method, calibrated):
                    lock = json.loads(Path('selection.json').read_text())
                    self.assertEqual(lock['method'], 'hflip_prob')
                    self.assertEqual(lock['recipe']['exp_id'], 'T04_cutmix')
                    calls.append((cfg.exp_id, cfg.seed))
                    Path('predictions').mkdir(exist_ok=True)
                    p = np.eye(9) * .9 + np.ones((9, 9)) / 90
                    for tag, split in ((cfg.exp_id, 'test'), (cfg.exp_id + 'uncal', 'test'), (cfg.exp_id, 'val')):
                        save_predictions(f'predictions/{tag}_seed{cfg.seed}_{split}.csv',
                                         labels.Filename.tolist(), labels.Label.to_numpy(), p)
                    return {'exp_id': cfg.exp_id, 'seed': cfg.seed, 'top1': 1., 'macro_f1': 1.,
                            'ece': .08888889, 'balanced_acc': 1., 'method': method}
                metrics_stub = types.ModuleType('sklearn.metrics')
                metrics_stub.confusion_matrix = lambda *a, **k: np.eye(9)
                class DisplayStub:
                    def __init__(self, matrix):
                        pass
                    def plot(self, ax):
                        return self
                metrics_stub.ConfusionMatrixDisplay = DisplayStub
                with patch.dict(sys.modules, {'sklearn.metrics': metrics_stub}), \
                     redirect_stdout(io.StringIO()), \
                     patch.object(full_lab, 'import_previous_output'), \
                     patch.object(full_lab, 'train_suite', side_effect=fake_suite), \
                     patch.object(full_lab, 'run', side_effect=fake_run), \
                     patch.object(full_lab, 'loaders', return_value=[None, None]), \
                     patch.object(full_lab, 'inference_suite', side_effect=fake_inference), \
                     patch.object(full_lab, 'final_inference', side_effect=fake_final), \
                     patch('train.plot_curves'):
                    full_lab.run_all(base, labels, labels)
                    self.assertEqual(len(calls), 6)
                    self.assertTrue(Path('results.xlsx').exists())
                    self.assertTrue(Path('COMPLETE.json').exists())
                    full_lab.run_all(base, labels, labels)
                    self.assertEqual(len(calls), 6, 'Saved test outputs must not be inferred again')
            finally:
                os.chdir(initial)


if __name__ == '__main__':
    unittest.main()
