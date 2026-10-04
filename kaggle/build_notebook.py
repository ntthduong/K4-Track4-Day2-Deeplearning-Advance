"""Build a portable Kaggle notebook embedding the implementation and original eval.py."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
cells = []
def markdown(source):
    cells.append(dict(cell_type='markdown', metadata={}, source=source))
def code(source):
    cells.append(dict(cell_type='code', metadata={}, source=source, execution_count=None, outputs=[]))

markdown('''# K4 Track4 Day2 — DeepWeeds
Notebook triển khai các phần chính của bài lab. GPU T4 x2, Internet bật để lấy trọng số ImageNet.
Dùng **dữ liệu đã tải sẵn**, không tải lại ảnh. Chọn mọi cấu hình trên val; test chỉ ở chung kết.
Code được nhúng trong notebook và ghi vào `/kaggle/working/day2_lab_code`.
Chạy đầy đủ tự động: backbone, ablation, công thức kết hợp, suy luận, chung kết 3 seed và xuất kết quả.
Gắn Output của Version 4 bằng Add Input để tái sử dụng 5 backbone đã chạy.
''')
sources = {p.name: p.read_text(encoding='utf-8') for p in (ROOT/'code').glob('*.py')}
sources['eval.py'] = (ROOT/'eval.py').read_text(encoding='utf-8')
code('''import os, sys, json, subprocess, importlib
from pathlib import Path
os.chdir('/kaggle/working')
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'timm', 'thop', 'openpyxl'], check=True)
CODE_DIR = Path('/kaggle/working/day2_lab_code')
CODE_DIR.mkdir(exist_ok=True)
SOURCES = ''' + repr(sources) + '''
for name, content in SOURCES.items():
    (CODE_DIR/name).write_text(content, encoding='utf-8')
sys.path.insert(0, str(CODE_DIR))
for name in ('dataset', 'model', 'losses', 'inference', 'benchmark', 'train', 'experiments', 'eval', 'test_core'):
    sys.modules.pop(name, None)
importlib.invalidate_caches()
import torch, timm, numpy as np, pandas as pd
print('torch', torch.__version__, 'timm', timm.__version__)
print('GPU:', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
print('Implementation:', CODE_DIR)
''')
markdown('''## 1. Dữ liệu có sẵn và kiểm tra fold 0
Gắn dataset bằng **Add Input** nếu dữ liệu nằm trong dataset Kaggle.
Notebook tự tìm bộ CSV chính thức và ảnh trong `/kaggle/input`, `/kaggle/working/data`.
Nếu có nhiều bản, đặt hai đường dẫn bên dưới. Ảnh cần được giải nén sẵn.
''')
code('''from dataset import load_split, check_split, CLASS_NAMES
# Điền đường dẫn ở đây khi cần. Không dùng CSV đã tự chia lại.
IMAGES_DIR = None
LABELS_DIR = None
roots = [Path('/kaggle/input'), Path('/kaggle/working/data')]
if LABELS_DIR is None:
    candidates = sorted({p.parent for root in roots if root.exists() for p in root.rglob('train_subset0.csv')
                         if all((p.parent/f'{s}_subset0.csv').exists() for s in ('val', 'test'))})
    if not candidates:
        # The uploaded images.zip contains JPEGs only. Fetch the unchanged official metadata.
        import urllib.request
        labels_path = Path('/kaggle/working/data/labels')
        labels_path.mkdir(parents=True, exist_ok=True)
        label_base = 'https://raw.githubusercontent.com/AlexOlsen/DeepWeeds/master/labels'
        for filename in ('labels.csv', 'train_subset0.csv', 'val_subset0.csv', 'test_subset0.csv'):
            urllib.request.urlretrieve(f'{label_base}/{filename}', labels_path/filename)
        candidates = [labels_path]
    if len(candidates) != 1:
        raise ValueError(f'Found label directories: {candidates}. Attach your dataset and set LABELS_DIR.')
    LABELS_DIR = str(candidates[0])
train_df, val_df, test_df = load_split(LABELS_DIR)
if IMAGES_DIR is None:
    sample = str(train_df.iloc[0].Filename)
    candidates = sorted({p.parent for root in roots if root.exists() for p in root.rglob(sample)})
    if len(candidates) != 1:
        raise ValueError(f'Found image directories: {candidates}. Set IMAGES_DIR to extracted JPEGs.')
    IMAGES_DIR = str(candidates[0])
checks = check_split(train_df, val_df, test_df, IMAGES_DIR)
print('Images:', IMAGES_DIR, 'Labels:', LABELS_DIR)
''')
markdown('## 2. EDA và kiểm tra code')
code('''import matplotlib.pyplot as plt
from PIL import Image
counts = pd.DataFrame({s: df.Label.value_counts().reindex(range(9), fill_value=0)
                       for s, df in zip(('train', 'val', 'test'), (train_df, val_df, test_df))})
counts.index = CLASS_NAMES
display(counts)
Path('curves').mkdir(exist_ok=True)
ax = counts.plot.bar(figsize=(12,5), title='DeepWeeds fold 0')
ax.set_ylabel('Images'); ax.figure.tight_layout()
ax.figure.savefig('curves/EDA_class_distribution.png', dpi=150)
plt.show()
fig, axs = plt.subplots(9,3, figsize=(9,24))
for k in range(9):
    examples = train_df[train_df.Label == k].sample(3, random_state=0)
    for ax, filename in zip(axs[k], examples.Filename):
        with Image.open(Path(IMAGES_DIR)/filename) as im:
            ax.imshow(im.convert('RGB'))
        ax.set_title(CLASS_NAMES[k]); ax.axis('off')
fig.tight_layout(); fig.savefig('curves/EDA_samples.png', dpi=100); plt.show()
''')
code('''# Kiểm tra loss, diện tích CutMix, gộp BN, temperature, optimizer, frozen BN và train/eval.
import unittest, test_core
suite = unittest.defaultTestLoader.loadTestsFromModule(test_core)
test_result = unittest.TextTestRunner(verbosity=2).run(suite)
assert test_result.wasSuccessful()
''')
code('''from dataclasses import replace
from train import Config, run, run_dir
from dataset import build_transforms, make_loader
from experiments import train_suite, inference_suite, load_checkpoint, final_inference, export_results, restore_config
from inference import predict_logits
from eval import compute_metrics, save_predictions
BASE = Config(images_dir=IMAGES_DIR, labels_dir=LABELS_DIR, epochs=12, batch_size=64, seed=0)
print(BASE)
# Kiểm tra augmentation bằng mắt.
from torchvision.utils import make_grid
x, y, filenames = next(iter(make_loader(train_df, IMAGES_DIR, build_transforms(True), 8, True, num_workers=0)))
display_image = x * torch.tensor([.229,.224,.225])[None,:,None,None] + torch.tensor([.485,.456,.406])[None,:,None,None]
plt.figure(figsize=(12,4)); plt.imshow(make_grid(display_image.clamp(0,1), nrow=4).permute(1,2,0)); plt.axis('off'); plt.show()
print('Labels:', y.tolist())
''')
markdown('''## 3. Kiểm tra pipeline trước khi chạy thật
Bật `RUN_SANITY` để overfit một batch ảnh cố định. Ghi lại loss và ảnh augmentation vào báo cáo.
Loss của logits đồng đều bằng ln(9); loss head ngẫu nhiên chỉ xấp xỉ giá trị này.
''')
code('''RUN_SANITY = True
if RUN_SANITY:
    from model import build_model
    from train import set_seed
    set_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sanity = build_model('efficientnet_b0', pretrained=False, init='scratch').to(device)
    sanity.train()
    images, labels, _ = next(iter(make_loader(train_df, IMAGES_DIR, build_transforms(False), 8, False, num_workers=0)))
    images, labels = images.to(device), labels.to(device)
    opt = torch.optim.Adam(sanity.parameters(), lr=1e-3)
    print('Uniform loss:', torch.nn.functional.cross_entropy(torch.zeros(8,9,device=device), labels).item())
    losses = []
    for step in range(100):
        opt.zero_grad(); z = sanity(images); loss = torch.nn.functional.cross_entropy(z, labels)
        loss.backward(); opt.step(); losses.append(loss.item())
        if step % 20 == 0: print(step, loss.item())
    plt.plot(losses); plt.xlabel('Step'); plt.ylabel('One-batch loss'); plt.show()
    print('Final batch accuracy:', (sanity(images).argmax(1) == labels).float().mean().item())
    del sanity, opt
    if torch.cuda.is_available(): torch.cuda.empty_cache()
''')
markdown('## 4. So sánh 5 backbone với cùng công thức nền')
code('''RUN_BACKBONES = True
if RUN_BACKBONES:
    backbone_results = train_suite(BASE, 'backbones')
    display(backbone_results)
''')
markdown('''## 5. Ablation huấn luyện: khởi tạo, augmentation, loss, EMA
Chọn backbone bằng kết quả val, rồi đặt `SELECTED_BACKBONE`.
Mỗi recipe T chỉ đổi một yếu tố so với T00. Chạy kết hợp sau khi xem các ablation.
''')
code('''SELECTED_BACKBONE = 'resnet50'
SELECTED = replace(BASE, backbone=SELECTED_BACKBONE)
RUN_TRAINING = False
if RUN_TRAINING:
    training_results = train_suite(SELECTED, 'training')
    display(training_results)
# Chỉ đặt những yếu tố đã cho kết quả tốt trên val vào COMBINED.
COMBINED = replace(SELECTED, exp_id='T10_combined', aug='color', mix='cutmix', loss='ls', label_smoothing=.1)
RUN_COMBINED = False
if RUN_COMBINED:
    print(run(COMBINED))
''')
markdown('''## 6. Suy luận và độ trễ trên val
1-view, TTA lật (prob/logit), temperature scaling và gộp BN. Đây là 4 phương pháp ngoài mốc.
Độ trễ có >=10 warmup, >=50 lần đo, đồng bộ CUDA; báo p50/p95/p99.
Phần đo tính forward và phép lật; chưa gồm đọc ảnh hoặc gộp xác suất.
''')
code('''BEST_EXP_ID = 'T00'  # đổi sang exp_id thắng trên val
BEST = replace(SELECTED, exp_id=BEST_EXP_ID)
RUN_INFERENCE = False
if RUN_INFERENCE:
    BEST = restore_config(BEST_EXP_ID, BASE.seed)
    metadata = json.loads((run_dir(BEST)/'config.json').read_text())
    data_cfg = metadata['data_cfg']
    val_loader = make_loader(val_df, IMAGES_DIR, build_transforms(False, BEST.img_size,
                             mean=data_cfg['mean'], std=data_cfg['std']), BEST.batch_size, False, num_workers=2)
    inference_results = inference_suite(BEST, val_loader)
    display(inference_results)
    inference_results[['exp_id','gpu','dtype','batch','p50','p95','p99','images_per_s']].to_csv('latency_results.csv', index=False)
    ax = inference_results.plot.scatter(x='p95', y='macro_f1', title='Validation F1 vs latency')
    for _, r in inference_results.iterrows(): ax.annotate(r.exp_id, (r.p95,r.macro_f1))
    ax.figure.savefig('curves/inference_tradeoff.png', dpi=150)
''')
markdown('''## 7. Chung kết — chỉ bật sau khi chốt mọi lựa chọn trên val
Huấn luyện final và baseline với seed 0/1/2. Chọn phương pháp suy luận trước khi mở test.
Không quay lại sửa cấu hình sau khi xem test. Code từ chối ghi đè prediction test.
''')
code('''RUN_FINAL = False
FINAL_METHOD = 'identity'  # identity | hflip_prob | hflip_logit; chọn trên val
FINAL_CALIBRATED = True
if RUN_FINAL:
    BEST = restore_config(BEST_EXP_ID, BASE.seed)
    final_rows = []
    for seed in (0,1,2):
        for cfg, method, calibrated in [(replace(BEST, exp_id='F01', seed=seed, save_test_predictions=False), FINAL_METHOD, FINAL_CALIBRATED),
                                         (replace(SELECTED, exp_id='T00_final', seed=seed, save_test_predictions=False), 'identity', False)]:
            if not (run_dir(cfg)/'summary.json').exists(): run(cfg)
            data_cfg = json.loads((run_dir(cfg)/'config.json').read_text())['data_cfg']
            transform = build_transforms(False, cfg.img_size, mean=data_cfg['mean'], std=data_cfg['std'])
            loaders = [make_loader(df, IMAGES_DIR, transform, cfg.batch_size, False, num_workers=2) for df in (val_df,test_df)]
            final_rows.append(final_inference(cfg, *loaders, method=method, calibrated=calibrated))
            pd.DataFrame(final_rows).to_csv('final_results.csv', index=False)
    display(pd.DataFrame(final_rows).groupby('exp_id')[['macro_f1','top1','ece']].agg(['mean','std']))
''')
markdown('## 8. Tính điểm và xuất bảng — chỉ khi đã có prediction thật')
code('''RUN_SCORE = False
if RUN_SCORE:
    from eval import main as eval_main
    for tag in ('F01','T00_final'):
        assert eval_main(['score','--pred',f'predictions/{tag}_seed*_test.csv','--test-csv',f'{LABELS_DIR}/test_subset0.csv',
                          '--tag',tag,'--out','eval_out']) == 0
    assert eval_main(['grade','--final','predictions/F01_seed*_test.csv','--baseline','predictions/T00_final_seed*_test.csv',
                      '--uncal','predictions/F01uncal_seed*_test.csv','--final-val','predictions/F01_seed*_val.csv',
                      '--test-csv',f'{LABELS_DIR}/test_subset0.csv','--out','eval_out']) == 0
    pc = pd.read_csv('eval_out/F01_per_class.csv')
    pc.to_csv('per_class_results.csv', index=False)
    print(export_results())
# Lưu output bằng Save Version khi hoàn tất. Báo cáo cần phân tích lỗi và đối chiếu rubric.
''')
# Replace manual stage switches with one validation-driven orchestration cell.
stage_index = next(i for i, cell in enumerate(cells)
                   if cell['cell_type'] == 'markdown' and cell['source'].startswith('## 4.'))
cells = cells[:stage_index]
markdown('''## 4–8. Chạy đầy đủ các thí nghiệm và chung kết
Tự chọn backbone bằng macro-F1 val; chạy 10 ablation và kết hợp các yếu tố thắng.
So sánh 1-view, TTA prob/logit, temperature, fusion và FP16.
Lưu `selection.json` trước test; chung kết và baseline với seed 0/1/2.
Tái sử dụng các lần train giống hệt, ghi rõ nguồn; không suy luận test lại nếu đã có predictions.
Kết quả: `results.xlsx`, `report.md`, `final_mean_std.csv`, `curves/`, `predictions/`, `eval_out/`.
Chỉ khi có `COMPLETE.json` thì tất cả các phần đã hoàn tất.
''')
code('''assert torch.cuda.is_available(), 'Enable GPU before running the full lab'
from full_lab import run_all
final_results = run_all(BASE, val_df, test_df)
display(final_results)
''')
nb = dict(cells=cells, metadata={'kernelspec': {'display_name':'Python 3','language':'python','name':'python3'},
                               'language_info': {'name':'python'}}, nbformat=4, nbformat_minor=5)
out = ROOT/'kaggle'/'lab_day2_kaggle.ipynb'
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print(out)
