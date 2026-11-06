"""Local web app for blood cell detection with the trained YOLO26 model.

Two-stage pipeline: the detector (models/best.pt) finds WBC, RBC and platelets; every WBC box
is then cropped and classified into a WBC type by a ResNet18 trained on BloodMNIST
(models/resnet18-bloodmnist-wbc.pt, see app/wbc_classifier.py). If the classifier weights are missing, the app still
runs as a plain detector.

Start it with:
  ./.venv/bin/python app/server.py
then open http://127.0.0.1:8000

The server runs the model once per upload and returns every candidate detection
above a low floor (MIN_CONF). The browser applies the confidence threshold from
the slider, so changing the slider updates boxes and counts instantly.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, UnidentifiedImageError

try:  # run as `python app/server.py` (app/ on sys.path) or imported as app.server
    import wbc_classifier
except ImportError:  # pragma: no cover
    from app import wbc_classifier

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.environ.get("BC_MODEL_PATH", ROOT / "models" / "best.pt"))
IMGSZ = 640
MIN_CONF = 0.05  # lowest threshold the slider can reach; detections below it are never returned
IOU = 0.7
DEFAULT_CONF = 0.25  # initial slider value; also the threshold for the RBCs that set the WBC crop size
MAX_UPLOAD_MB = 20
HOST = "127.0.0.1"
PORT = int(os.environ.get("PORT", "8000"))

app = Flask(__name__, static_folder=str(Path(__file__).parent / "static"), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

_model = None


def get_model():
    """Load the model once, on first use."""
    global _model
    if _model is None:
        from ultralytics import YOLO

        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model weights not found: {MODEL_PATH}")
        _model = YOLO(str(MODEL_PATH))
    return _model


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/info")
def info():
    model = get_model()
    cls_model = wbc_classifier.get_cls_model()
    return jsonify(
        {
            "model": MODEL_PATH.name,
            "classes": [model.names[i] for i in sorted(model.names)],
            "wbc_model": wbc_classifier.CLS_MODEL_PATH.name if cls_model else None,
            "wbc_types": list(cls_model.names) if cls_model else [],
            "imgsz": IMGSZ,
            "min_conf": MIN_CONF,
            "default_conf": DEFAULT_CONF,
        }
    )


@app.post("/api/detect")
def detect():
    upload = request.files.get("image")
    if upload is None or upload.filename == "":
        return jsonify({"error": "No image uploaded. Send it as form field 'image'."}), 400
    try:
        image = Image.open(io.BytesIO(upload.read())).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return jsonify({"error": "The file is not a readable image."}), 400

    model = get_model()
    result = model.predict(image, imgsz=IMGSZ, conf=MIN_CONF, iou=IOU, verbose=False)[0]

    detections = []
    for box, cls, conf in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist()):
        cls_id = int(cls)
        detections.append(
            {
                "class_id": cls_id,
                "class": model.names[cls_id],
                "confidence": round(float(conf), 4),
                "box": [round(v, 1) for v in box],  # x1, y1, x2, y2 in original image pixels
            }
        )
    detections.sort(key=lambda d: d["confidence"], reverse=True)

    # Stage 2: classify every WBC candidate. The slider filters in the browser, so all
    # candidates are classified up front (a smear has few WBCs, so this is cheap).
    # The crop window is sized from the median RBC (magnification reference), using the RBCs
    # at the app's default threshold so low-confidence junk boxes do not skew it.
    wbcs = [d for d in detections if d["class"] == "WBC"]
    rbc_size = wbc_classifier.median_rbc_size(
        [d["box"] for d in detections if d["class"] == "RBC" and d["confidence"] >= DEFAULT_CONF])
    for d, typ in zip(wbcs, wbc_classifier.classify_boxes(image, [d["box"] for d in wbcs], rbc_size)):
        d["wbc_type"] = typ["type"]
        d["wbc_type_confidence"] = typ["confidence"]
        d["wbc_type_probs"] = typ["probs"]

    width, height = image.size
    return jsonify(
        {
            "filename": upload.filename,
            "width": width,
            "height": height,
            "min_conf": MIN_CONF,
            "rbc_size_px": None if rbc_size is None else round(rbc_size, 1),
            "detections": detections,
        }
    )


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": f"File is larger than {MAX_UPLOAD_MB} MB."}), 413


if __name__ == "__main__":
    print(f"Model: {MODEL_PATH}")
    print(f"WBC classifier: {wbc_classifier.CLS_MODEL_PATH}"
          + ("" if wbc_classifier.CLS_MODEL_PATH.exists() else " (missing, WBC types disabled)"))
    print(f"Open http://{HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=False)
