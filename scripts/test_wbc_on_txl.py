"""Evaluate the full detector -> crop -> WBC classifier pipeline on labelled TXL-PBC images.

TXL-PBC has no WBC-type labels, but its 360 x 363 images come from the PBC dataset, BloodMNIST's
source. scripts/match_txl_bloodmnist.py links each one to its BloodMNIST twin and therefore to a
WBC-type label. The labelled cell is the one at the image centre (BloodMNIST is a centre crop).

Groups (by the twin's BloodMNIST split):
  test set   twin in BloodMNIST val/test -> never seen by the classifier in training (154 images)
  tuning set twin in BloodMNIST train    -> seen in training; used only to calibrate the crop
             window (WINDOW_PER_RBC = 200 px / median detected RBC side)

Pipeline per image, exactly as in the web app: detector (models/best.pt, 640 px, conf 0.25,
IoU 0.7) -> WBC box nearest the image centre -> square window of WINDOW_PER_RBC x median RBC
side -> classifier. An image where the detector finds no WBC counts as an error ("missed").
For comparison the script also scores the ground-truth WBC box, a box-relative crop
(CROP_SCALE x WBC box, which loses cell size) and a fixed 200 px window (BloodMNIST's own window,
only valid at PBC magnification).

Writes reports/wbc_classifier/txl_pipeline_eval.json, confusion-matrix PNG and a contact sheet.

Usage:
  ./.venv/bin/python scripts/test_wbc_on_txl.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
import server  # noqa: E402  (detector settings + loader, so the evaluation matches the app)
import wbc_classifier  # noqa: E402

TXL = ROOT / "data" / "TXL-PBC"
OUT = ROOT / "reports" / "wbc_classifier"
MATCHES = OUT / "txl_bloodmnist_matches.json"
APP_CONF = 0.25  # the app's default slider value
MISSED = "missed"


def gt_wbc_boxes(label_file: Path, w: int, h: int) -> list[list[float]]:
    boxes = []
    for line in label_file.read_text().split("\n"):
        parts = line.split()
        if len(parts) == 5 and int(parts[0]) == 0:  # class 0 = WBC
            cx, cy, bw, bh = (float(v) for v in parts[1:])
            boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return boxes


def nearest_centre(boxes: list[list[float]], w: int, h: int) -> list[float] | None:
    if not boxes:
        return None
    return min(boxes, key=lambda b: ((b[0] + b[2]) / 2 - w / 2) ** 2 + ((b[1] + b[3]) / 2 - h / 2) ** 2)


def load_items(names: list[str]) -> tuple[list[dict], Counter]:
    detector = server.get_model()
    items, excluded = [], Counter()
    for m in json.loads(MATCHES.read_text()):
        if not m["matched"]:
            excluded["no BloodMNIST twin"] += 1
            continue
        if m["bm_class"] not in names:
            excluded[f"twin is {m['bm_class']} (not a WBC type)"] += 1
            continue
        image = Image.open(TXL / "images" / m["txl_split"] / m["txl_image"]).convert("RGB")
        det = detector.predict(image, imgsz=server.IMGSZ, conf=APP_CONF, iou=server.IOU, verbose=False)[0]
        det_wbc = [b for b, c in zip(det.boxes.xyxy.tolist(), det.boxes.cls.tolist()) if int(c) == 0]
        det_rbc = [b for b, c in zip(det.boxes.xyxy.tolist(), det.boxes.cls.tolist()) if int(c) == 1]
        gt = gt_wbc_boxes(TXL / "labels" / m["txl_split"] / f"{Path(m['txl_image']).stem}.txt", *image.size)
        items.append({**m, "image": image, "truth": m["bm_class"], "n_det_wbc": len(det_wbc), "n_gt_wbc": len(gt),
                      "rbc_size": wbc_classifier.median_rbc_size(det_rbc),
                      "det_box": nearest_centre(det_wbc, *image.size), "gt_box": nearest_centre(gt, *image.size)})
    return items, excluded


CROPS = {  # name -> window side for (item, box); "app" is what the web app does
    "app": lambda it, b: wbc_classifier.crop_side(b, it["rbc_size"]),
    "box_relative": lambda it, b: wbc_classifier.crop_side(b, None),
    "fixed_200px": lambda it, b: 200.0,
}


def run(items: list[dict], key: str, crop: str = "app") -> list[dict]:
    preds = []
    for it in items:
        if it[key] is None:
            preds.append({"type": MISSED, "confidence": 0.0})
        else:
            preds.append(wbc_classifier.classify_boxes(
                it["image"], [it[key]], side_fn=lambda b, it=it: CROPS[crop](it, b))[0])
    return preds


def score(items: list[dict], preds: list[dict], names: list[str]) -> dict:
    cols = names + [MISSED]
    cm = np.zeros((len(names), len(cols)), dtype=int)  # rows: true type, cols: predicted (+ missed)
    for it, p in zip(items, preds):
        cm[names.index(it["truth"]), cols.index(p["type"])] += 1
    n = int(cm.sum())
    per_class = {}
    for i, name in enumerate(names):
        tp, support, predicted = cm[i, i], cm[i].sum(), cm[:, i].sum()
        rec = tp / support if support else None
        prec = tp / predicted if predicted else None
        per_class[name] = {"support": int(support), "recall": None if rec is None else round(float(rec), 4),
                           "precision": None if prec is None else round(float(prec), 4)}
    recalls = [v["recall"] for v in per_class.values() if v["recall"] is not None]
    return {"n": n, "correct": int(np.trace(cm[:, :len(names)])), "accuracy": round(float(np.trace(cm) / n), 4),
            "balanced_accuracy": round(float(np.mean(recalls)), 4), "missed": int(cm[:, -1].sum()),
            "per_class": per_class, "confusion_matrix": {"rows_true": names, "cols_pred": cols, "counts": cm.tolist()}}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(float(c - h), 4), round(float(c + h), 4)


def print_cm(s: dict) -> None:
    cm = s["confusion_matrix"]
    short = {"basophil": "baso", "eosinophil": "eos", "immature_granulocytes": "IG", "lymphocyte": "lymph",
             "monocyte": "mono", "neutrophil": "neut", MISSED: "missed"}
    print("    true \\ pred           " + "".join(short[c].rjust(7) for c in cm["cols_pred"]) + "    n  recall")
    for name, row in zip(cm["rows_true"], cm["counts"]):
        r = s["per_class"][name]["recall"]
        print(f"    {name:22s}" + "".join(str(v).rjust(7) for v in row) + f"{sum(row):5d}  {'' if r is None else f'{r:.3f}'}")


def plot_cm(s: dict, title: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.array(s["confusion_matrix"]["counts"])
    rows, cols = s["confusion_matrix"]["rows_true"], s["confusion_matrix"]["cols_pred"]
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(cols)), [c.replace("_", "\n") for c in cols], rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(rows)), [r.replace("_", "\n") for r in rows], fontsize=9)
    for i in range(len(rows)):
        for j in range(len(cols)):
            if cm[i, j]:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=10,
                        color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted (pipeline)")
    ax.set_ylabel("True (BloodMNIST twin label)")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def contact_sheet(items: list[dict], preds: list[dict], path: Path, tile: int = 150, cols: int = 8) -> None:
    order = sorted(range(len(items)), key=lambda i: preds[i]["type"] == items[i]["truth"])  # errors first
    rows = (len(order) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile, rows * (tile + 30)), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
    except OSError:
        font = ImageFont.load_default()
    for k, i in enumerate(order):
        it, p = items[i], preds[i]
        x, y = (k % cols) * tile, (k // cols) * (tile + 30)
        box = it["det_box"] or it["gt_box"]
        side = wbc_classifier.crop_side(box, it["rbc_size"])
        sheet.paste(wbc_classifier.square_crop(it["image"], box, side).resize((tile, tile)), (x, y))
        ok = p["type"] == it["truth"]
        draw.text((x + 3, y + tile + 2), f"pred {p['type'][:13]} {p['confidence']:.2f}", fill="#14532d" if ok else "#b91c1c", font=font)
        draw.text((x + 3, y + tile + 15), f"true {it['truth'][:13]}", fill="#333", font=font)
    sheet.save(path, quality=88)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()

    cls = wbc_classifier.get_cls_model()
    if cls is None:
        sys.exit(f"Missing classifier weights: {wbc_classifier.CLS_MODEL_PATH}")
    names = cls.names
    items, excluded = load_items(names)
    test = [it for it in items if it["bm_split"] in ("val", "test")]
    tune = [it for it in items if it["bm_split"] == "train"]
    report: dict = {"classifier": cls.path.name, "detector": server.MODEL_PATH.name, "detector_conf": APP_CONF,
                    "excluded_images": dict(excluded),
                    "test_set": {"n_images": len(test), "twin_splits": dict(Counter(it["bm_split"] for it in test)),
                                 "txl_splits": dict(Counter(it["txl_split"] for it in test)),
                                 "true_class_counts": dict(Counter(it["truth"] for it in test)),
                                 "images_with_more_than_one_gt_wbc": sum(it["n_gt_wbc"] > 1 for it in test)},
                    "tuning_set": {"n_images": len(tune)}}
    print(f"Excluded: {dict(excluded)}")
    print(f"Test set: {len(test)} images (twin in BloodMNIST val/test); tuning set: {len(tune)} (twin in train)")

    # 1. Calibrate the crop window on the tuning (BloodMNIST-train) images only
    rbc = [it["rbc_size"] for it in tune if it["rbc_size"]]
    ratio = 200.0 / float(np.median(rbc))
    report["calibration"] = {"median_rbc_side_px_tuning": round(float(np.median(rbc)), 1),
                             "window_per_rbc_calibrated": round(ratio, 3),
                             "window_per_rbc_in_app": wbc_classifier.WINDOW_PER_RBC,
                             "test_images_without_rbc": sum(it["rbc_size"] is None for it in test)}
    print(f"Calibration on tuning set: median RBC side {np.median(rbc):.1f} px -> window/RBC {ratio:.3f} "
          f"(app uses {wbc_classifier.WINDOW_PER_RBC})")
    if abs(ratio - wbc_classifier.WINDOW_PER_RBC) > 0.01:
        print("  WARNING: app constant WINDOW_PER_RBC differs from this calibration")

    # 2. Evaluate on the 154 never-trained-on images
    runs = [("det_box", "app", "pipeline: detector box, RBC-sized window (as in the app)"),
            ("gt_box", "app", "ground-truth box, RBC-sized window"),
            ("det_box", "box_relative", "detector box, box-relative window (1.5x box)"),
            ("det_box", "fixed_200px", "detector box, fixed 200 px window (PBC magnification only)")]
    report["results"] = {}
    for key, crop, label in runs:
        preds = run(test, key, crop)
        s = score(test, preds, names)
        s["description"] = label
        s["accuracy_95ci"] = wilson(s["correct"], s["n"])
        s["errors"] = [{"txl_image": f"{it['txl_split']}/{it['txl_image']}", "true": it["truth"], "pred": p["type"],
                        "confidence": p["confidence"]} for it, p in zip(test, preds) if p["type"] != it["truth"]]
        report["results"][f"{key.replace('_box', '')}/{crop}"] = s
        print(f"\n{label}: accuracy {s['accuracy']:.4f} ({s['correct']}/{s['n']}, "
              f"95% CI {s['accuracy_95ci'][0]:.3f}-{s['accuracy_95ci'][1]:.3f}), "
              f"balanced {s['balanced_accuracy']:.4f}, detector missed {s['missed']}")
        print_cm(s)
        if (key, crop) == ("det_box", "app"):
            plot_cm(s, f"TXL-PBC full pipeline, {s['n']} unseen images, accuracy {s['accuracy']:.3f}",
                    OUT / "txl_pipeline_confusion_matrix.png")
            contact_sheet(test, preds, OUT / "txl_pipeline_contact_sheet.jpg")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "txl_pipeline_eval.json").write_text(json.dumps(report, indent=2))
    print(f"Wrote {OUT / 'txl_pipeline_eval.json'}")


if __name__ == "__main__":
    main()
