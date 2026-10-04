# DeepWeeds Day 2 implementation

Author: Nguyễn Thị Thùy Dương — student ID 2A202602905.

Use `lab_day2_kaggle.ipynb` beside this README inside a submission package, or
`kaggle/lab_day2_kaggle.ipynb` from the repository root. It embeds every module
in this directory plus the unchanged official `eval.py`, so no GitHub push is needed.
Attach the already downloaded DeepWeeds dataset to Kaggle and set `IMAGES_DIR` and
`LABELS_DIR` if automatic discovery finds zero or multiple directories. Use the
official fold-0 train/val/test CSV files, not a new random split.

The notebook now runs all stages automatically after setup, split checks, EDA and
CPU checks. Attach the saved Version 4 notebook output using Add Input to reuse the
five backbone runs. `full_lab.py` selects backbone, recipe and inference from
validation only, writes `selection.json` before test, evaluates final and baseline
with seeds 0/1/2, then exports metrics, Excel and a report. A `COMPLETE.json` file
marks success. Identical seed-0 runs are reused with explicit provenance. Test
predictions already written are read for metrics rather than inferred again.

For a local CLI run, place the repository root and this `code/` directory on
`PYTHONPATH`, then run:

```text
python code/train.py --set exp_id=T00 images_dir=data/images labels_dir=data/labels
python code/test_core.py -v
```

Dependencies: torch >=2.3, torchvision matched to torch, timm >=1.0,
numpy, pandas, Pillow, matplotlib, thop, openpyxl. Kaggle provides torch/torchvision;
the setup cell installs timm, thop and openpyxl without replacing CUDA packages.
Only a single GPU is used by the training loop; the second T4 is available but DDP
is not implemented. This keeps experiments and seed handling simple.

Checkpoint selection uses macro-F1 validation, choosing the earlier epoch on ties.
EMA averages parameters and copies live BatchNorm buffers. Loss curves for weighted
CE aggregate batch losses weighted by batch size. CPU tests cover focal gamma=0,
label smoothing epsilon=0, CutMix area, Conv-BN equivalence, temperature invariance,
optimizer grouping and frozen-BN training.

Latency is forward-only with warmup and synchronization. TTA latency includes flip
and repeated forward passes but excludes probability aggregation. thop counts can
omit attention operations; transformer GMACs must be checked with an appropriate
profiler before reporting them as exact. Fusion is limited to Sequential Conv/BN
pairs and timm ResNet blocks with established adjacency.

The Excel exporter creates the seven requested sheets using available real result
files. The completed submission is in
`submissions/2A202602905_Nguyen_Thi_Thuy_Duong/`. Its RUBRIC_CHECK.md documents
verified requirements and remaining limitations; results come from Kaggle version 5.
Save Kaggle output after runs to retain checkpoints, histories and predictions.
