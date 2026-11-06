"""Find TXL-PBC images that are the same micrographs as BloodMNIST images.

TXL-PBC includes images from the PBC dataset (360 x 363 px), which is also BloodMNIST's source.
BloodMNIST was made by centre-cropping each PBC image to 200 x 200 and resizing. This script
applies the same transform to every TXL-PBC image of that size and finds its nearest BloodMNIST
image (64 px copy in data/bloodmnist). A match below the distance threshold means it is the same
micrograph, which gives
  * a leakage check: TXL-PBC crops matched to BloodMNIST *train* images were seen in training;
  * WBC-type labels for TXL-PBC cells, so the detector-crop pipeline can be scored on matched
    cells that are NOT in the BloodMNIST training split.

Writes reports/wbc_classifier/txl_bloodmnist_matches.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BM = ROOT / "data" / "bloodmnist"
TXL = ROOT / "data" / "TXL-PBC"
OUT = ROOT / "reports" / "wbc_classifier" / "txl_bloodmnist_matches.json"
SIDE = 64
THRESH = 6.0  # mean absolute difference in 0..255 grey levels at 64 px; same image ~ 1-3, different > 20


def to_bm(image: Image.Image) -> np.ndarray:
    w, h = image.size
    left, top = (w - 200) // 2, (h - 200) // 2
    crop = image.convert("RGB").crop((left, top, left + 200, top + 200)).resize((SIDE, SIDE), Image.BICUBIC)
    return np.asarray(crop, dtype=np.float32)


def main() -> None:
    files = sorted(BM.glob("*/*/*.png"))
    bank = np.stack([np.asarray(Image.open(f).convert("RGB"), dtype=np.float32) for f in files])
    small = bank.reshape(len(files), 16, 4, 16, 4, 3).mean(axis=(2, 4)).reshape(len(files), -1)
    meta = [(f.parts[-3], f.parts[-2], f.name) for f in files]  # split, class, name

    matches = []
    for split in ("train", "val", "test"):
        for p in sorted((TXL / "images" / split).glob("*.png")):
            image = Image.open(p)
            if image.size != (360, 363):
                continue
            q = to_bm(image)
            qs = q.reshape(16, 4, 16, 4, 3).mean(axis=(1, 3)).reshape(-1)
            cand = np.argsort(np.abs(small - qs).mean(axis=1))[:20]  # coarse search, then exact
            d = np.abs(bank[cand] - q).mean(axis=(1, 2, 3))
            best = int(cand[d.argmin()])
            matches.append({"txl_split": split, "txl_image": p.name, "distance": round(float(d.min()), 2),
                            "matched": bool(d.min() < THRESH), "bm_split": meta[best][0],
                            "bm_class": meta[best][1], "bm_file": meta[best][2]})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(matches, indent=2))
    dist = np.array([m["distance"] for m in matches])
    print(f"{len(matches)} TXL-PBC images of 360x363; distance quantiles "
          f"{np.percentile(dist, [0, 25, 50, 75, 90, 100]).round(2).tolist()}")
    for split in ("train", "val", "test"):
        ms = [m for m in matches if m["txl_split"] == split and m["matched"]]
        by = {s: sum(m["bm_split"] == s for m in ms) for s in ("train", "val", "test")}
        print(f"TXL {split}: {len(ms)} matched -> BloodMNIST splits {by}")


if __name__ == "__main__":
    main()
