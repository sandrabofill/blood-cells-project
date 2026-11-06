"""WBC-type classification of detector boxes, shared by the web app and the evaluation scripts.

Classifier: torchvision ResNet18 trained on the six BloodMNIST WBC classes
(models/resnet18-bloodmnist-wbc.pt). The checkpoint is a dict holding the state_dict plus
metadata; class names, input size and normalisation are read from it, and it is loaded with
weights_only=True so no pickled code can run.

Cropping. BloodMNIST images are a fixed 200 x 200 px centre window of PBC micrographs, so the
cell's apparent size carries information (a lymphocyte is small, a monocyte large). The crop
must keep that: it is a square window of fixed *physical* size centred on the WBC box. The
image's magnification is estimated from the red cells the detector finds (RBCs are nearly
constant in size): window side = WINDOW_PER_RBC x median RBC box side. WINDOW_PER_RBC = 2.46 was
calibrated as 200 px / 81.4 px, the median detected RBC side in PBC-sourced TXL-PBC images
whose BloodMNIST twins are training images (scripts/test_wbc_on_txl.py).
A crop that scales with each WBC's own box (side = CROP_SCALE x box) discards size and scored
0.83 instead of 0.97 on the 154 held-out TXL-PBC images; it is used only when no RBC is found.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CLS_MODEL_PATH = Path(os.environ.get("BC_CLS_MODEL_PATH", ROOT / "models" / "resnet18-bloodmnist-wbc.pt"))
WINDOW_PER_RBC = 2.46  # crop side = WINDOW_PER_RBC * median RBC box side (see module docstring)
CROP_SCALE = 1.5  # fallback when the image has no detected RBC: crop side = CROP_SCALE * WBC box side
BATCH = 64
# The checkpoint stores input_size=128 but not the source resolution. It was trained on the 64 px
# BloodMNIST copy (data/bloodmnist) upscaled to 128: its stored val accuracy (0.9947) reproduces
# exactly on 64 px val images, while 224 px val images fed straight in score only 0.88. Crops are
# therefore downsampled to 64 px (bicubic, 0.993 on 224 px val) before the upscale to input_size.
SOURCE_SIZE = 64

_model = None
_lock = threading.Lock()


class WBCClassifier:
    def __init__(self, path: Path):
        import torch
        import torchvision

        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        arch = ckpt.get("architecture", "resnet18")
        self.names: list[str] = list(ckpt["classes"])
        self.input_size = int(ckpt.get("input_size", 128))
        self.mean = torch.tensor(ckpt.get("mean", [0.485, 0.456, 0.406])).view(1, 3, 1, 1)
        self.std = torch.tensor(ckpt.get("std", [0.229, 0.224, 0.225])).view(1, 3, 1, 1)
        net = getattr(torchvision.models, arch)(weights=None, num_classes=len(self.names))
        net.load_state_dict(ckpt["state_dict"])  # strict: every key must match
        self.net = net.eval()
        self.path = path

    def predict_probs(self, crops: list[Image.Image]):
        """Class probabilities, shape (len(crops), n_classes)."""
        import numpy as np
        import torch

        out = []
        for i in range(0, len(crops), BATCH):
            arr = np.stack([np.asarray(c.convert("RGB").resize((SOURCE_SIZE, SOURCE_SIZE), Image.BICUBIC)
                                       .resize((self.input_size, self.input_size), Image.BILINEAR),
                                       dtype=np.float32) for c in crops[i:i + BATCH]])
            x = torch.from_numpy(arr).permute(0, 3, 1, 2) / 255.0
            x = (x - self.mean) / self.std
            with torch.inference_mode():
                out.append(torch.softmax(self.net(x), dim=1))
        return torch.cat(out).numpy()


def get_cls_model() -> WBCClassifier | None:
    """Load the classifier once, on first use. Returns None if the weights are missing."""
    global _model
    with _lock:
        if _model is None and CLS_MODEL_PATH.exists():
            _model = WBCClassifier(CLS_MODEL_PATH)
    return _model


def crop_side(box: list[float], rbc_size: float | None) -> float:
    """Window side in pixels: fixed physical size if the RBC size is known, else box-relative."""
    if rbc_size:
        return WINDOW_PER_RBC * rbc_size
    return CROP_SCALE * max(box[2] - box[0], box[3] - box[1])


def median_rbc_size(boxes: list[list[float]]) -> float | None:
    """Median longer side of the RBC boxes, the image's magnification reference."""
    sides = sorted(max(b[2] - b[0], b[3] - b[1]) for b in boxes)
    if not sides:
        return None
    mid = len(sides) // 2
    return sides[mid] if len(sides) % 2 else (sides[mid - 1] + sides[mid]) / 2


def square_crop(image: Image.Image, box: list[float], side: float) -> Image.Image:
    """Square crop of the given side centred on box (x1, y1, x2, y2). Parts outside the image are
    padded with the image's mean colour so the cell stays centred, as in BloodMNIST."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(1, round(side))
    left, top = round(cx - side / 2), round(cy - side / 2)
    right, bottom = left + side, top + side
    if left >= 0 and top >= 0 and right <= image.width and bottom <= image.height:
        return image.crop((left, top, right, bottom))
    fill = tuple(int(c) for c in image.resize((1, 1), Image.BOX).getpixel((0, 0)))
    canvas = Image.new("RGB", (side, side), fill)
    inner = image.crop((max(left, 0), max(top, 0), min(right, image.width), min(bottom, image.height)))
    canvas.paste(inner, (max(left, 0) - left, max(top, 0) - top))
    return canvas


def classify_boxes(image: Image.Image, boxes: list[list[float]], rbc_size: float | None = None,
                   side_fn=None) -> list[dict]:
    """Classify each WBC box. rbc_size is the median RBC box side in this image (None if there
    are no RBCs). side_fn(box) overrides the window size (used by the evaluation scripts).
    Returns one dict per box: type, confidence and all class probabilities."""
    model = get_cls_model()
    if model is None or not boxes:
        return []
    sides = [side_fn(b) if side_fn else crop_side(b, rbc_size) for b in boxes]
    probs = model.predict_probs([square_crop(image, b, s) for b, s in zip(boxes, sides)])
    out = []
    for row in probs:
        top = int(row.argmax())
        out.append({
            "type": model.names[top],
            "confidence": round(float(row[top]), 4),
            "probs": {n: round(float(p), 4) for n, p in zip(model.names, row)},
        })
    return out
