"""Convert BloodMNIST (MedMNIST+, 224 x 224) into a YOLO classification dataset of WBC types.

Input : data/bloodmnist_224_raw/bloodmnist_224.npz  (official MedMNIST+ release, Zenodo record 10519652)
Output: data/bloodmnist_wbc_yolo/{train,val,test}/<class>/<split>_<index>.png

Only the six white-blood-cell classes are kept; erythroblasts and platelets are dropped.
The original train / val / test split is preserved. The layout is the one Ultralytics expects
for classification (`yolo classify train data=<dir>`).

Usage:
  ./.venv/bin/python scripts/convert_bloodmnist.py [--src PATH] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = ROOT / "data" / "bloodmnist_224_raw" / "bloodmnist_224.npz"
DEFAULT_OUT = ROOT / "data" / "bloodmnist_wbc_yolo"

# BloodMNIST label id -> output class folder (ids as in data/bloodmnist/classes.json)
WBC_CLASSES = {
    0: "basophil",
    1: "eosinophil",
    3: "immature_granulocyte",
    4: "lymphocyte",
    5: "monocyte",
    6: "neutrophil",
}
DROPPED = {2: "erythroblast", 7: "platelet"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if args.out.exists():
        shutil.rmtree(args.out)

    # allow_pickle=False: the archive only needs plain arrays, so refuse anything else
    npz = np.load(args.src, allow_pickle=False)
    summary: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        images = npz[f"{split}_images"]  # (N, 224, 224, 3) uint8
        labels = npz[f"{split}_labels"].reshape(-1)
        assert images.ndim == 4 and images.shape[-1] == 3, images.shape
        counts = {name: 0 for name in WBC_CLASSES.values()}
        for cls_name in WBC_CLASSES.values():
            (args.out / split / cls_name).mkdir(parents=True, exist_ok=True)
        for idx, (img, label) in enumerate(zip(images, labels)):
            cls_name = WBC_CLASSES.get(int(label))
            if cls_name is None:
                continue
            Image.fromarray(img).save(args.out / split / cls_name / f"{split}_{idx:05d}.png")
            counts[cls_name] += 1
        summary[split] = counts
        print(f"{split}: {sum(counts.values())} images of {images.shape[1]}x{images.shape[2]} px, "
              f"dropped {int(np.isin(labels, list(DROPPED)).sum())} non-WBC  {counts}")
        del images

    (args.out / "summary.json").write_text(json.dumps(
        {"source": str(args.src.name), "resolution": 224, "classes": sorted(WBC_CLASSES.values()),
         "counts": summary}, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
