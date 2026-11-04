# Blood cell detection with YOLO26

Fine-tunes a pretrained **YOLO26** detector to locate and classify cells in blood smear
images. The three classes are **WBC** (white blood cells), **RBC** (red blood cells) and
**Platelets**. Training runs on a **Kaggle GPU** (T4), and results are monitored through a
local HTML dashboard.

## Repository layout

| Path | Contents |
| --- | --- |
| `data/TXL-PBC/` | Dataset: 1,260 images, 18,143 YOLO-format annotations, train/val/test split |
| `scripts/analyze_dataset.py` | Validates the labels and writes the dataset report |
| `scripts/kaggle_run.py` | Uploads the dataset to Kaggle, starts the GPU kernel, checks status, pulls outputs |
| `scripts/build_reports.py` | Builds the training/performance reports and the dashboard data |
| `kaggle/train_yolo26.py` | Training script that runs on Kaggle (fine-tune, log each epoch, evaluate) |
| `dashboard/index.html` | Monitoring dashboard (open in a browser; refreshes every 30 s) |
| `reports/dataset/dataset_report.html` | Dataset report |
| `reports/training_report.html` | Training progress report |
| `reports/performance_report.html` | Model performance report (val and test) |

## Setup

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

The Kaggle API token goes in `.env` as `KAGGLE_API_TOKEN=...`. The file is git-ignored and
is never printed or copied into the repository.

## Workflow

```bash
# 1. Dataset report
./.venv/bin/python scripts/analyze_dataset.py

# 2. Upload the dataset to Kaggle (private dataset)
./.venv/bin/python scripts/kaggle_run.py upload-dataset

# 3. Start training on a Kaggle GPU
./.venv/bin/python scripts/kaggle_run.py push --model yolo26s.pt --epochs 100

# 4. Monitor
./.venv/bin/python scripts/kaggle_run.py status
./.venv/bin/python scripts/kaggle_run.py pull          # copies outputs into runs/kaggle_output/
./.venv/bin/python scripts/build_reports.py            # refreshes dashboard + reports
open dashboard/index.html
```

Kaggle returns a kernel's outputs when the run finishes, so the dashboard shows
per-epoch curves after a `pull`. While the run is in progress, `status` shows whether
the kernel is queued, running, complete, or failed.

## Method

- **Model:** YOLO26 (Ultralytics), COCO-pretrained, fine-tuned end to end.
- **Input:** 640 px, with Ultralytics' default augmentation.
- **Split:** the dataset's own train / val / test split (882 / 252 / 126 images). Test images
  are used only for the final evaluation.
- **Seed:** 42, deterministic mode on.
- **Selection:** the checkpoint with the best validation fitness (`best.pt`) is evaluated on
  val and test.

## Results

YOLO26s, 100 epochs, 640 px, batch 16, on a Kaggle GPU (about 27 minutes of training).
Best checkpoint: epoch 86.

| Split | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| Validation | 0.959 | 0.951 | 0.984 | 0.865 |
| **Test** | **0.974** | **0.948** | **0.990** | **0.877** |

Test results by class:

| Class | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| WBC | 0.996 | 0.992 | 0.995 | 0.900 |
| RBC | 0.980 | 0.933 | 0.991 | 0.907 |
| Platelets | 0.945 | 0.918 | 0.983 | 0.823 |

Platelets are the hardest class: they are the smallest objects and the rarest class in the
data. Full details are in `reports/performance_report.html`, and training curves are in
`reports/training_report.html`.
