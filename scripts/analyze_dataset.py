"""Validate the TXL-PBC dataset and produce dataset statistics.

Outputs (under reports/dataset/):
  dataset_stats.json   machine-readable statistics used by the dashboard
  dataset_report.html  standalone HTML report with figures and tables

Usage:
  python scripts/analyze_dataset.py
"""

from __future__ import annotations

import base64
import io
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "TXL-PBC"
OUT = ROOT / "reports" / "dataset"

SPLITS = ["train", "val", "test"]


def read_labels(path: Path) -> list[tuple[int, float, float, float, float]]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        rows.append((int(float(parts[0])), *map(float, parts[1:5])))
    return rows


def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def main() -> int:
    cfg = yaml.safe_load((DATA / "data.yaml").read_text())
    names = cfg["names"]
    OUT.mkdir(parents=True, exist_ok=True)


    problems: list[str] = []
    split_stats: dict[str, dict] = {}
    all_boxes = defaultdict(list)  # class -> list of (w, h) in fraction units
    image_sizes = Counter()
    modes = Counter()
    per_image_counts = {}
    class_counts_per_split = {}
    cells_per_image_hist = {}
    file_formats = Counter()

    for split in SPLITS:
        img_dir = DATA / "images" / split
        lbl_dir = DATA / "labels" / split
        images = sorted(img_dir.glob("*.png"))
        counts = Counter()
        n_cells = 0
        n_empty = 0
        for img_path in images:
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                problems.append(f"missing label: {split}/{img_path.name}")
                continue
            with Image.open(img_path) as im:
                image_sizes[im.size] += 1
                modes[im.mode] += 1
                file_formats[im.format] += 1
            rows = read_labels(lbl_path)
            if not rows:
                n_empty += 1
            per_image_counts[f"{split}/{img_path.stem}"] = len(rows)
            for cls, xc, yc, w, h in rows:
                if cls not in range(len(names)):
                    problems.append(f"bad class {cls} in {split}/{lbl_path.name}")
                    continue
                if not all(0.0 <= v <= 1.0 for v in (xc, yc, w, h)):
                    problems.append(f"out-of-range value in {split}/{lbl_path.name}")
                counts[cls] += 1
                n_cells += 1
                all_boxes[cls].append((w, h, xc, yc))
        orphan_labels = {p.stem for p in lbl_dir.glob("*.txt")} - {p.stem for p in images}
        for stem in sorted(orphan_labels):
            problems.append(f"label without image: {split}/{stem}.txt")

        split_stats[split] = {
            "images": len(images),
            "cells": n_cells,
            "empty_images": n_empty,
            "class_counts": {names[c]: counts.get(c, 0) for c in range(len(names))},
        }
        class_counts_per_split[split] = [counts.get(c, 0) for c in range(len(names))]

    # Per-image cell count histogram (all splits)
    counts_all = list(per_image_counts.values())
    cells_per_image = Counter(counts_all)
    cells_per_image_hist = {int(k): int(v) for k, v in sorted(cells_per_image.items())}

    # Box statistics per class (in pixels, assuming typical image size)
    box_stats = {}
    for c, name in enumerate(names):
        arr = np.array(all_boxes[c]) if all_boxes[c] else np.zeros((0, 4))
        if len(arr):
            box_stats[name] = {
                "count": int(len(arr)),
                "width_mean": float(arr[:, 0].mean()),
                "height_mean": float(arr[:, 1].mean()),
                "width_median": float(np.median(arr[:, 0])),
                "height_median": float(np.median(arr[:, 1])),
                "width_min": float(arr[:, 0].min()),
                "width_max": float(arr[:, 0].max()),
            }
        else:
            box_stats[name] = {"count": 0}

    total_images = sum(s["images"] for s in split_stats.values())
    total_cells = sum(s["cells"] for s in split_stats.values())
    total_counts = Counter()
    for s in split_stats.values():
        total_counts.update(s["class_counts"])
    imbalance = {
        k: round(v / total_cells, 4) for k, v in total_counts.items()
    }

    # ----- Figures -----
    x = np.arange(len(names))
    width = 0.27
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, split in enumerate(SPLITS):
        ax.bar(x + (i - 1) * width, class_counts_per_split[split], width, label=split)
    ax.set_xticks(x, names)
    ax.set_ylabel("Annotated cells")
    ax.set_title("Cells per class and split")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig_class = fig_to_b64(fig)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    split_names = SPLITS
    img_counts = [split_stats[s]["images"] for s in split_names]
    cell_counts = [split_stats[s]["cells"] for s in split_names]
    ax.bar(split_names, img_counts, color="#4C78A8", label="images")
    ax2 = ax.twinx()
    ax2.plot(split_names, cell_counts, "o-", color="#E45756", label="cells")
    ax.set_ylabel("Images")
    ax2.set_ylabel("Annotated cells")
    ax.set_title("Images and cells per split")
    fig_split = fig_to_b64(fig)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    keys = sorted(cells_per_image_hist)
    ax.bar(keys, [cells_per_image_hist[k] for k in keys], color="#72B7B2")
    ax.set_xlabel("Cells in image")
    ax.set_ylabel("Number of images")
    ax.set_title("Cells per image distribution")
    ax.grid(axis="y", alpha=0.3)
    fig_density = fig_to_b64(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    colors = ["#4C78A8", "#E45756", "#54A24B"]
    for c, name in enumerate(names):
        if not all_boxes[c]:
            continue
        arr = np.array(all_boxes[c])
        axes[0].hist(arr[:, 0], bins=40, alpha=0.55, color=colors[c], label=name)
        axes[1].hist(arr[:, 1], bins=40, alpha=0.55, color=colors[c], label=name)
    axes[0].set_title("Box width (fraction of image)")
    axes[1].set_title("Box height (fraction of image)")
    for a in axes:
        a.legend()
        a.grid(alpha=0.3)
    fig_boxes = fig_to_b64(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    for c, name in enumerate(names):
        if not all_boxes[c]:
            continue
        arr = np.array(all_boxes[c])
        ax.scatter(arr[:, 2], arr[:, 3], s=3, alpha=0.35, color=colors[c], label=name)
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_xlabel("x center (fraction)")
    ax.set_ylabel("y center (fraction)")
    ax.set_title("Cell center locations")
    ax.legend(markerscale=4)
    fig_centers = fig_to_b64(fig)

    stats = {
        "dataset": "TXL-PBC (local copy)",
        "classes": names,
        "total_images": total_images,
        "total_cells": total_cells,
        "splits": split_stats,
        "total_class_counts": dict(total_counts),
        "class_share": imbalance,
        "image_sizes": {f"{w}x{h}": n for (w, h), n in image_sizes.most_common()},
        "image_modes": dict(modes),
        "file_formats_detected": dict(file_formats),
        "cells_per_image_histogram": cells_per_image_hist,
        "box_stats": box_stats,
        "problems": problems,
        "problem_count": len(problems),
    }
    (OUT / "dataset_stats.json").write_text(json.dumps(stats, indent=2))

    def table(rows, headers):
        head = "".join(f"<th>{h}</th>" for h in headers)
        body = "".join(
            "<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows
        )
        return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"

    split_rows = []
    for s in SPLITS:
        st = split_stats[s]
        split_rows.append(
            [s, st["images"], st["cells"], st["empty_images"]]
            + [st["class_counts"][n] for n in names]
        )
    split_table = table(
        split_rows, ["Split", "Images", "Cells", "Empty images"] + names
    )
    box_rows = [
        [
            n,
            box_stats[n]["count"],
            f"{box_stats[n].get('width_median', 0):.3f}",
            f"{box_stats[n].get('height_median', 0):.3f}",
            f"{box_stats[n].get('width_median', 0) * max(w for (w, h) in image_sizes) if image_sizes else 0:.0f}",
        ]
        for n in names
    ]
    box_table = table(
        box_rows,
        [
            "Class",
            "Boxes",
            "Median width (frac)",
            "Median height (frac)",
            "Median width (px, largest image)",
        ],
    )
    size_rows = [[f"{k}", v] for k, v in image_sizes.most_common()]
    size_table = table(size_rows, ["Image size (WxH)", "Images"])
    problem_text = (
        "<p class='ok'>No problems found: every image has a matching label file, "
        "all class IDs are valid and all coordinates are inside [0, 1].</p>"
        if not problems
        else "<ul>" + "".join(f"<li>{p}</li>" for p in problems[:200]) + "</ul>"
    )

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>TXL-PBC Dataset Report</title>
<style>
body{{font-family:-apple-system,Helvetica,Arial,sans-serif;margin:0;color:#1d2330;background:#f7f8fb}}
main{{max-width:1080px;margin:0 auto;padding:32px 20px}}
h1{{margin-bottom:4px}} h2{{margin-top:36px;border-bottom:1px solid #d9dde7;padding-bottom:6px}}
.kpis{{display:flex;gap:14px;flex-wrap:wrap;margin:16px 0}}
.kpi{{background:#fff;border:1px solid #e1e5ee;border-radius:10px;padding:14px 18px;min-width:150px}}
.kpi b{{display:block;font-size:24px}} .kpi span{{color:#667085;font-size:13px}}
table{{border-collapse:collapse;width:100%;background:#fff;font-size:14px}}
th,td{{border:1px solid #e1e5ee;padding:7px 10px;text-align:right}} th:first-child,td:first-child{{text-align:left}}
th{{background:#eef1f7}} img{{max-width:100%;background:#fff;border:1px solid #e1e5ee;border-radius:8px;padding:6px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:18px}}
.ok{{color:#1b7f3b}} .note{{color:#667085;font-size:14px}}
</style></head><body><main>
<h1>TXL-PBC Dataset Report</h1>
<p class="note">Blood-smear images with YOLO bounding-box labels for WBC, RBC and platelets.</p>
<div class="kpis">
 <div class="kpi"><span>Images</span><b>{total_images:,}</b></div>
 <div class="kpi"><span>Annotated cells</span><b>{total_cells:,}</b></div>
 <div class="kpi"><span>Classes</span><b>{len(names)}</b></div>
 <div class="kpi"><span>Label problems</span><b>{len(problems)}</b></div>
</div>

<h2>Splits</h2>
{split_table}
<div class="grid" style="margin-top:18px">
 <img alt="split chart" src="data:image/png;base64,{fig_split}">
 <img alt="class chart" src="data:image/png;base64,{fig_class}">
</div>

<h2>Class statistics</h2>
{box_table}
<p class="note">Class shares overall: {", ".join(f"{k} {v*100:.1f}%" for k, v in imbalance.items())}.
Platelets are the rarest class, so per-class metrics matter more than overall mAP alone.</p>
<img alt="box sizes" src="data:image/png;base64,{fig_boxes}">

<h2>Spatial and density</h2>
<div class="grid">
 <img alt="density" src="data:image/png;base64,{fig_density}">
 <img alt="centers" src="data:image/png;base64,{fig_centers}">
</div>

<h2>Image properties</h2>
{size_table}
<p class="note">Image modes: {dict(modes)}. Files named .png are detected as
{dict(file_formats)} by content, so loaders must read by content rather than by extension.
The loader used for training (OpenCV/PIL) handles this without changes.</p>

<h2>Validation</h2>
{problem_text}
</main></body></html>"""
    (OUT / "dataset_report.html").write_text(html)

    print(f"images={total_images} cells={total_cells} problems={len(problems)}")
    for s in SPLITS:
        print(s, split_stats[s]["images"], split_stats[s]["cells"], split_stats[s]["class_counts"])
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
