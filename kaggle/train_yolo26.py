"""Fine-tune YOLO26 on TXL-PBC inside a Kaggle GPU kernel.

This script is executed by Kaggle (see scripts/kaggle_run.py). It:
  1. copies the dataset from /kaggle/input into /kaggle/working (writable),
  2. rewrites data.yaml with absolute paths,
  3. fine-tunes a pretrained YOLO26 checkpoint,
  4. writes per-epoch progress to progress.json (for the dashboard),
  5. evaluates the best checkpoint on val and test, saving metrics.json.

All outputs go to /kaggle/working/outputs, which Kaggle returns to the local
machine with `kaggle kernels output`.

Settings come from the environment so the same file serves quick smoke tests
and full runs:  BC_MODEL, BC_EPOCHS, BC_IMGSZ, BC_BATCH, BC_DATASET_DIR.
"""


import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

# Pinned to a release at least two weeks old. YOLO26 weights are included in this version.
ULTRALYTICS_VERSION = "8.4.183"
MODEL = os.environ.get("BC_MODEL", "yolo26s.pt")
EPOCHS = int(os.environ.get("BC_EPOCHS", "100"))
IMGSZ = int(os.environ.get("BC_IMGSZ", "640"))
BATCH = int(os.environ.get("BC_BATCH", "16"))
PATIENCE = int(os.environ.get("BC_PATIENCE", "25"))
SEED = 42

WORK = Path("/kaggle/working")
OUT = WORK / "outputs"
RUNS = WORK / "runs"
OUT.mkdir(parents=True, exist_ok=True)


def find_dataset_root() -> Path:
    """Locate the folder that contains data.yaml under /kaggle/input."""
    override = os.environ.get("BC_DATASET_DIR")
    if override:
        return Path(override)
    for path in Path("/kaggle/input").rglob("data.yaml"):
        has_images = (path.parent / "images").is_dir() or (path.parent / "images.zip").exists()
        if has_images:
            return path.parent
    raise FileNotFoundError("No TXL-PBC data.yaml found under /kaggle/input")


def stage_dataset() -> Path:
    """Copy the dataset into writable storage, extracting zip archives if Kaggle stored them as zips."""
    src = find_dataset_root()
    dst = Path("/tmp/TXL-PBC")  # kept out of /kaggle/working so it is not returned as output
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(src / "data.yaml", dst / "data.yaml")
    for sub in ("images", "labels"):
        target = dst / sub
        if not target.exists():
            if (src / sub).is_dir():
                shutil.copytree(src / sub, target)
            elif (src / f"{sub}.zip").exists():
                shutil.unpack_archive(str(src / f"{sub}.zip"), str(target))
                # A zip made from a folder may hold a single top-level folder with the same name
                nested = target / sub
                if nested.is_dir():
                    for item in nested.iterdir():
                        shutil.move(str(item), str(target / item.name))
                    nested.rmdir()
            else:
                raise FileNotFoundError(f"{sub} not found in {src}")
    cfg = yaml.safe_load((dst / "data.yaml").read_text())
    cfg["path"] = str(dst)
    cfg["train"] = "images/train"
    cfg["val"] = "images/val"
    cfg["test"] = "images/test"
    (dst / "data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    return dst / "data.yaml"


class ProgressLogger:
    """Writes one JSON record per epoch, so progress can be read while training runs."""

    def __init__(self, path: Path, total_epochs: int):
        self.path = path
        self.total = total_epochs
        self.start = time.time()
        self.records: list[dict] = []

    def on_fit_epoch_end(self, trainer) -> None:
        metrics = {k: float(v) for k, v in trainer.metrics.items()}
        record = {
            "epoch": int(trainer.epoch) + 1,
            "total_epochs": self.total,
            "elapsed_s": round(time.time() - self.start, 1),
            "metrics": metrics,
            "lr": [float(trainer.optimizer.param_groups[0]["lr"])],
        }
        self.records.append(record)
        self.path.write_text(json.dumps({"status": "training", "records": self.records}, indent=2))


def evaluate(model_path: Path, data_yaml: Path, split: str) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    res = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=IMGSZ,
        batch=BATCH,
        plots=True,
        project=str(RUNS),
        name=f"eval_{split}",
        exist_ok=True,
        verbose=False,
    )
    names = res.names
    per_class = []
    for idx, cls_id in enumerate(res.ap_class_index):
        per_class.append(
            {
                "class": names[int(cls_id)],
                "precision": float(res.box.p[idx]),
                "recall": float(res.box.r[idx]),
                "map50": float(res.box.all_ap[idx, 0]),
                "map50_95": float(res.box.all_ap[idx].mean()),
            }
        )
    return {
        "split": split,
        "precision": float(res.box.mp),
        "recall": float(res.box.mr),
        "map50": float(res.box.map50),
        "map50_95": float(res.box.map),
        "per_class": per_class,
        "speed_ms": {k: float(v) for k, v in res.speed.items()},
    }


def install_ultralytics() -> None:
    """Install the pinned Ultralytics version (Kaggle images ship a different one)."""
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", f"ultralytics=={ULTRALYTICS_VERSION}"],
        check=True,
    )


def main() -> None:
    install_ultralytics()
    import torch
    from ultralytics import YOLO

    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu", torch.cuda.get_device_name(0))

    data_yaml = stage_dataset()
    progress_path = OUT / "progress.json"
    logger = ProgressLogger(progress_path, EPOCHS)

    model = YOLO(MODEL)
    model.add_callback("on_fit_epoch_end", logger.on_fit_epoch_end)

    t0 = time.time()
    model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        patience=PATIENCE,
        seed=SEED,
        deterministic=True,
        device=0 if torch.cuda.is_available() else "cpu",
        project=str(RUNS),
        name="yolo26_txl_pbc",
        exist_ok=True,
        plots=True,
        pretrained=True,
        workers=4,
        verbose=True,
    )
    train_seconds = time.time() - t0

    run_dir = RUNS / "yolo26_txl_pbc"
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"

    # Copy the training curves and artefacts to the output folder
    for name in ["results.csv", "args.yaml", "confusion_matrix.png", "confusion_matrix_normalized.png",
                 "BoxPR_curve.png", "BoxF1_curve.png", "BoxP_curve.png", "BoxR_curve.png",
                 "results.png", "labels.jpg", "train_batch0.jpg", "val_batch0_pred.jpg"]:
        src = run_dir / name
        if src.exists():
            shutil.copy(src, OUT / name)
    if best.exists():
        shutil.copy(best, OUT / "best.pt")

    print("evaluating best checkpoint")
    results = {
        "model": MODEL,
        "imgsz": IMGSZ,
        "epochs_requested": EPOCHS,
        "epochs_completed": len(logger.records),
        "train_seconds": round(train_seconds, 1),
        "splits": {},
    }
    for split in ["val", "test"]:
        results["splits"][split] = evaluate(best if best.exists() else last, data_yaml, split)
    (OUT / "metrics.json").write_text(json.dumps(results, indent=2))

    # Save a sample of test predictions for the report
    test_dir = Path("/tmp/TXL-PBC") / "images" / "test"
    sample = sorted(test_dir.glob("*.png"))[:12]
    if sample and best.exists():
        pred_dir = OUT / "test_predictions"
        pred_dir.mkdir(exist_ok=True)
        model = YOLO(str(best))
        for img in sample:
            r = model.predict(str(img), imgsz=IMGSZ, conf=0.25, verbose=False)[0]
            r.save(filename=str(pred_dir / img.name))

    progress = json.loads(progress_path.read_text())
    progress["status"] = "finished"
    progress_path.write_text(json.dumps(progress, indent=2))
    print("done; outputs in", OUT)


if __name__ == "__main__":
    main()
