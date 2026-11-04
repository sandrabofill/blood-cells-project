"""Build the training and performance reports and the data behind the dashboard.

Reads what the Kaggle run produced (copied into runs/kaggle_output/ by
`kaggle_run.py pull`) and the dataset statistics (reports/dataset/), then writes:

  dashboard/data.js                     data loaded by dashboard/index.html
  dashboard/assets/                     figures and sample predictions used by the dashboard
  reports/training_report.html          training progress, curves and epoch log
  reports/performance_report.html       evaluation on val and test, per class

Run after every pull:
  python scripts/build_reports.py
"""

from __future__ import annotations

import base64
import csv
import datetime as dt
import html
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_RUN = ROOT / "runs" / "kaggle_output" / "outputs"
DASH = ROOT / "dashboard"
ASSETS = DASH / "assets"
REPORTS = ROOT / "reports"
DATASET_STATS = REPORTS / "dataset" / "dataset_stats.json"
KERNEL_STATUS = ROOT / "runs" / "kernel_status.json"

FIGURES = [
    ("results.png", "Training curves (Ultralytics)"),
    ("confusion_matrix_normalized.png", "Normalised confusion matrix (test)"),
    ("BoxPR_curve.png", "Precision-recall curve"),
    ("BoxF1_curve.png", "F1 vs confidence"),
    ("labels.jpg", "Training label distribution"),
    ("val_batch0_pred.jpg", "Validation predictions"),
]


def load_json(path: Path, default):
    if path.exists():
        return json.loads(path.read_text())
    return default


def b64_img(path: Path) -> str:
    mime = "image/png" if path.suffix == ".png" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def read_epoch_csv(path: Path) -> list[dict]:
    """results.csv from Ultralytics is the authoritative per-epoch log."""
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            rec = {k.strip(): float(v) for k, v in r.items() if k and v not in ("", None)}
            rows.append(rec)
    return rows


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    progress = load_json(OUT_RUN / "progress.json", {"status": "not started", "records": []})
    metrics = load_json(OUT_RUN / "metrics.json", None)
    dataset = load_json(DATASET_STATS, None)
    kernel = load_json(KERNEL_STATUS, {"status": "unknown"})

    # Per-epoch records: prefer results.csv (it has every field), fall back to progress.json
    csv_rows = read_epoch_csv(OUT_RUN / "results.csv")
    records = []
    if csv_rows:
        for i, row in enumerate(csv_rows, start=1):
            metrics_row = {k: v for k, v in row.items() if k != "epoch"}
            records.append({"epoch": int(row.get("epoch", i)), "metrics": metrics_row,
                            "lr": [metrics_row.get("lr/pg0", None)], "elapsed_s": None})
    else:
        records = progress.get("records", [])

    total = (metrics or {}).get("epochs_requested") or progress.get("total_epochs")
    completed = len(records)
    best = {}
    if records:
        best_rec = max(records, key=lambda r: r["metrics"].get("metrics/mAP50-95(B)", 0))
        best = {
            "epoch": best_rec["epoch"],
            "map50": best_rec["metrics"].get("metrics/mAP50(B)"),
            "map50_95": best_rec["metrics"].get("metrics/mAP50-95(B)"),
        }

    # Copy figures and predictions into the dashboard's assets folder
    figures = []
    for fname, label in FIGURES:
        src = OUT_RUN / fname
        if src.exists():
            shutil.copy(src, ASSETS / fname)
            figures.append({"src": f"assets/{fname}", "label": label})
    predictions = []
    pred_src = OUT_RUN / "test_predictions"
    if pred_src.exists():
        pred_dst = ASSETS / "predictions"
        pred_dst.mkdir(exist_ok=True)
        for img in sorted(pred_src.glob("*")):
            shutil.copy(img, pred_dst / img.name)
            predictions.append({"src": f"assets/predictions/{img.name}", "name": img.name})

    splits = (metrics or {}).get("splits", {})
    test = splits.get("test")
    val = splits.get("val")
    if test and dataset:
        test["images"] = dataset["splits"]["test"]["images"]
    generated_at = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    data = {
        "generated_at": generated_at,
        "model": (metrics or {}).get("model", "yolo26s.pt"),
        "imgsz": (metrics or {}).get("imgsz", 640),
        "train_seconds": (metrics or {}).get("train_seconds"),
        "kernel": kernel,
        "progress": {"status": progress.get("status"), "completed": completed, "total": total},
        "records": records,
        "best": best,
        "val": val,
        "test": test,
        "dataset": dataset,
        "figures": figures,
        "predictions": predictions,
    }
    (DASH / "data.js").write_text("window.DASHBOARD_DATA = " + json.dumps(data, indent=1) + ";\n")
    print("dashboard data written:", DASH / "data.js")

    write_training_report(data, records)
    write_performance_report(data)


def _table(headers: list[str], rows: list[list]) -> str:
    head = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _f(x, d=3):
    return "–" if x is None else f"{float(x):.{d}f}"


CSS = """
body{font-family:-apple-system,Helvetica,Arial,sans-serif;margin:0;background:#f7f8fb;color:#1d2330}
main{max-width:1080px;margin:0 auto;padding:30px 20px 60px}
h1{margin-bottom:4px} h2{margin-top:34px;border-bottom:1px solid #d9dde7;padding-bottom:6px}
.note{color:#667085;font-size:14px}
.kpis{display:flex;gap:12px;flex-wrap:wrap;margin:16px 0}
.kpi{background:#fff;border:1px solid #e1e5ee;border-radius:10px;padding:12px 16px;min-width:150px}
.kpi b{display:block;font-size:22px} .kpi span{color:#667085;font-size:13px}
table{border-collapse:collapse;width:100%;background:#fff;font-size:14px;margin:8px 0}
th,td{border:1px solid #e1e5ee;padding:6px 9px;text-align:right} th:first-child,td:first-child{text-align:left}
th{background:#eef1f7}
img{max-width:100%;background:#fff;border:1px solid #e1e5ee;border-radius:8px;padding:6px;margin:6px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px}
.scroll{overflow-x:auto}
"""


def _page(title: str, body: str) -> str:
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>{CSS}</style></head><body><main>{body}</main></body></html>"""


def write_training_report(data: dict, records: list[dict]) -> None:
    prog = data["progress"]
    best = data["best"]
    body = [f"<h1>Training report: YOLO26 on TXL-PBC</h1>"
            f"<p class='note'>Generated {data['generated_at']}. Model {html.escape(data['model'])}, "
            f"image size {data['imgsz']} px.</p>"]
    body.append('<div class="kpis">'
                f"<div class='kpi'><span>Status</span><b>{html.escape(str(prog['status']))}</b></div>"
                f"<div class='kpi'><span>Epochs</span><b>{prog['completed']} / {prog['total'] or '–'}</b></div>"
                f"<div class='kpi'><span>Best epoch</span><b>{best.get('epoch', '–')}</b></div>"
                f"<div class='kpi'><span>Best val mAP50-95</span><b>{_f(best.get('map50_95'))}</b></div>"
                f"<div class='kpi'><span>Training time</span><b>{_f((data['train_seconds'] or 0) / 60, 1)} min</b></div>"
                "</div>")
    if not records:
        body.append("<p>No epochs recorded yet. Run training on Kaggle and pull the outputs.</p>")
    else:
        body.append("<h2>Curves</h2><div class='grid'>")
        for fname, label in [("results.png", "Ultralytics training curves")]:
            p = OUT_RUN / fname
            if p.exists():
                body.append(f"<figure style='margin:0'><img src='{b64_img(p)}' alt='{label}'><figcaption class='note'>{label}</figcaption></figure>")
        body.append("</div>")
        body.append("<h2>Epoch log</h2><div class='scroll'>")
        rows = []
        for r in records:
            m = r["metrics"]
            rows.append([r["epoch"], _f(m.get("train/box_loss")), _f(m.get("train/cls_loss")),
                         _f(m.get("val/box_loss")), _f(m.get("metrics/precision(B)")),
                         _f(m.get("metrics/recall(B)")), _f(m.get("metrics/mAP50(B)")),
                         _f(m.get("metrics/mAP50-95(B)"))])
        body.append(_table(["Epoch", "Train box loss", "Train cls loss", "Val box loss",
                            "Precision", "Recall", "mAP50", "mAP50-95"], rows))
        body.append("</div>")
    (REPORTS / "training_report.html").write_text(_page("YOLO26 training report", "".join(body)))
    print("wrote", REPORTS / "training_report.html")


def write_performance_report(data: dict) -> None:
    body = [f"<h1>Performance report: YOLO26 on TXL-PBC</h1>"
            f"<p class='note'>Generated {data['generated_at']}. Scores are from Ultralytics validation "
            "using best.pt, the checkpoint with the best validation fitness (weighted mAP50 and mAP50-95).</p>"]
    test = data.get("test")
    val = data.get("val")
    if not test:
        body.append("<p>No evaluation results yet. Pull the Kaggle outputs after training finishes.</p>")
    else:
        body.append('<div class="kpis">'
                    f"<div class='kpi'><span>Test mAP50</span><b>{_f(test['map50'])}</b></div>"
                    f"<div class='kpi'><span>Test mAP50-95</span><b>{_f(test['map50_95'])}</b></div>"
                    f"<div class='kpi'><span>Test precision</span><b>{_f(test['precision'])}</b></div>"
                    f"<div class='kpi'><span>Test recall</span><b>{_f(test['recall'])}</b></div>"
                    "</div>")
        for label, split in (("Test set", test), ("Validation set", val)):
            if not split:
                continue
            body.append(f"<h2>{label}</h2>")
            rows = [[c["class"], _f(c["precision"]), _f(c["recall"]), _f(c["map50"]), _f(c["map50_95"])]
                    for c in split["per_class"]]
            rows.append(["All classes", _f(split["precision"]), _f(split["recall"]),
                         _f(split["map50"]), _f(split["map50_95"])])
            body.append(_table(["Class", "Precision", "Recall", "mAP50", "mAP50-95"], rows))
        body.append("<h2>Figures</h2><div class='grid'>")
        for f in data.get("figures", []):
            src = ROOT / "dashboard" / f["src"]
            if src.exists():
                body.append(f"<figure style='margin:0'><img src='{b64_img(src)}' alt='{html.escape(f['label'])}'>"
                            f"<figcaption class='note'>{html.escape(f['label'])}</figcaption></figure>")
        body.append("</div>")
    if data.get("predictions"):
        body.append("<h2>Sample test predictions</h2><div class='grid'>")
        for p in data["predictions"][:12]:
            src = ROOT / "dashboard" / p["src"]
            body.append(f"<img src='{b64_img(src)}' alt='{html.escape(p['name'])}'>")
        body.append("</div>")
    (REPORTS / "performance_report.html").write_text(_page("YOLO26 performance report", "".join(body)))
    print("wrote", REPORTS / "performance_report.html")


if __name__ == "__main__":
    main()
