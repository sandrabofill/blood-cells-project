# Blood cell detection with YOLO26

> **Disclaimer:** This is a student project for a master's course, for educational purposes
> only. It is not a medical device and has not been validated for clinical use or diagnosis.
> Do not use its output to make medical decisions.

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

## Local web app

The trained weights are copied to `models/best.pt` (git-ignored; they are large binaries).

```bash
./.venv/bin/python app/server.py
# open http://127.0.0.1:8000
```

Upload a smear image, then move the confidence slider (default 0.25). The boxes, class
labels and per-class counts update without re-running the model. Only images up to 20 MB
are accepted, and the server listens on 127.0.0.1 only.

### WBC types

Each detected WBC is also classified into one of six types (basophil, eosinophil, immature
granulocytes, lymphocyte, monocyte, neutrophil) by a ResNet18 trained on BloodMNIST
(`models/resnet18-bloodmnist-wbc.pt`, git-ignored). The app labels each WBC box with its type
and shows the count per type. Without that file the app runs as a plain detector.

How a WBC is cropped (`app/wbc_classifier.py`):

- The classifier was trained on the **64 px** BloodMNIST images upscaled to 128 px. It scores
  0.995 on 64 px val images but only 0.88 on 224 px ones, so every crop is downsampled to
  64 px before the upscale.
- BloodMNIST images are a fixed 200 px window of the PBC micrographs, so a cell's size in the
  frame is informative (lymphocytes are small, monocytes are large). The crop is a square
  window of fixed *physical* size, centred on the WBC box: 2.46 × the median RBC box side.
  RBCs serve as the magnification reference.

### Evaluation on TXL-PBC

TXL-PBC has no WBC-type labels. Its 360 × 363 images, however, are PBC micrographs that also
appear in BloodMNIST (`scripts/match_txl_bloodmnist.py` matches all 486 by pixel
comparison). That gives them labels, and it also means that 332 of those images are
BloodMNIST *training* images and must not be used for testing. The remaining **154 images**
(twin in BloodMNIST val/test) test the whole pipeline: detector → crop → classifier.

```bash
./.venv/bin/python scripts/match_txl_bloodmnist.py   # TXL-PBC <-> BloodMNIST twins
./.venv/bin/python scripts/test_wbc_on_txl.py        # pipeline evaluation
```

| Crop | Accuracy on the 154 images |
| --- | ---: |
| **App: detector box, window = 2.46 × median RBC** | **0.974** (150/154, 95% CI 0.935–0.990) |
| Same, ground-truth box instead of detector box | 0.974 |
| Window scaled to each WBC's own box (1.5×), which loses cell size | 0.831 |
| Fixed 200 px window (valid only at PBC magnification) | 0.987 |

The 2.46 ratio was calibrated on the 332 training-twin images only. All 4 errors involve
monocytes or neutrophils: 2 monocytes predicted as immature granulocytes, 1 monocyte as
lymphocyte, 1 neutrophil as immature granulocyte. The matched TXL-PBC images contain no
immature granulocytes, so that class is not tested here. The other TXL-PBC sources (575 × 575,
640 × 480, 256 × 256) are at different magnifications and have no labels. Results are in
`reports/wbc_classifier/`.

`scripts/convert_bloodmnist.py` converts the 224 px MedMNIST+ BloodMNIST release
(`data/bloodmnist_224_raw/`) into a 6-class WBC dataset in the YOLO classification layout
(`data/bloodmnist_wbc_yolo/`). It was used to check the classifier's input resolution.

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
