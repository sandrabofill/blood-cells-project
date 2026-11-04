# TXL-PBC

Blood-smear microscope images for **object detection**: locate each cell with a
bounding box and classify it as a white blood cell (WBC), red blood cell (RBC), or
platelet.

This local copy contains **1,260 PNG images and 18,143 annotated cells**. Image
dimensions vary, and files have RGB or RGBA color channels.

| Split | Images | Annotated cells |
| --- | ---: | ---: |
| `train` | 882 | 12,510 |
| `val` | 252 | 3,752 |
| `test` | 126 | 1,881 |
| **Total** | **1,260** | **18,143** |

## Files and labels

- `images/<split>/<name>.png` contains the image.
- `labels/<split>/<name>.txt` contains its cell annotations, with the same filename
  stem. Each image has a matching label file.
- [data.yaml](data.yaml) defines the split paths and class names.

Each annotation file has **one row per cell**, in YOLO detection format:

```text
class_id x_center y_center width height
```

The box center and size are normalized: horizontal values are divided by image
width, and vertical values by image height. They are fractions of the image size,
rather than pixel coordinates.

Class IDs: **0 = WBC**, **1 = RBC**, **2 = Platelets**.
