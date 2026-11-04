"""Drive Kaggle from the command line: upload the dataset, run training on a GPU, pull results.

Subcommands
  upload-dataset   create/update the private Kaggle dataset with TXL-PBC
  push             start the GPU training kernel (version it if it already exists)
  status           print the kernel status (queued / running / complete / error)
  pull             download the kernel outputs into runs/kaggle_output/

The Kaggle API token is read from the project's .env file (KAGGLE_API_TOKEN).
It is passed to the Kaggle CLI through the environment and never written to disk
or printed.

Examples
  python scripts/kaggle_run.py upload-dataset
  python scripts/kaggle_run.py push --epochs 100 --model yolo26s.pt
  python scripts/kaggle_run.py status
  python scripts/kaggle_run.py pull
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / ".env"
DATA = ROOT / "data" / "TXL-PBC"
STAGING = ROOT / "kaggle" / "staging"
DATASET_SLUG = "txl-pbc-blood-cells"
KERNEL_SLUG = "yolo26-txl-pbc-training"
KERNEL_SCRIPT = ROOT / "kaggle" / "train_yolo26.py"
PULL_DIR = ROOT / "runs" / "kaggle_output"
KAGGLE_BIN = ROOT / ".venv" / "bin" / "kaggle"


def load_env() -> dict:
    env = dict(os.environ)
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            env.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    if "KAGGLE_API_TOKEN" not in env:
        sys.exit("KAGGLE_API_TOKEN not found in .env")
    return env


def kaggle(env: dict, *args: str, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    cmd = [str(KAGGLE_BIN), *args]
    print("$ kaggle", " ".join(args), flush=True)
    return subprocess.run(cmd, env=env, check=check, text=True, capture_output=capture)


def username(env: dict) -> str:
    """Ask Kaggle who the token belongs to (the owner of our own dataset listing)."""
    if env.get("KAGGLE_USERNAME"):
        return env["KAGGLE_USERNAME"]
    res = kaggle(env, "datasets", "list", "--mine", "--csv", check=False, capture=True)
    for line in (res.stdout or "").splitlines()[1:]:
        ref = line.split(",")[0]
        if "/" in ref:
            return ref.split("/")[0]
    res = kaggle(env, "config", "view", check=False, capture=True)
    for line in (res.stdout or "").splitlines():
        if "username" in line.lower() and ":" in line:
            return line.split(":", 1)[1].strip()
    sys.exit("Could not determine Kaggle username; set KAGGLE_USERNAME in .env")


def stage_dataset() -> Path:
    """Copy the dataset into a clean staging folder that Kaggle will zip and upload."""
    target = STAGING / "dataset"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    shutil.copytree(DATA / "images", target / "images")
    shutil.copytree(DATA / "labels", target / "labels")
    shutil.copy(DATA / "data.yaml", target / "data.yaml")
    shutil.copy(DATA / "README.md", target / "README.md")
    return target


def cmd_upload_dataset(args, env):
    user = username(env)
    ds_dir = stage_dataset()
    ref = f"{user}/{DATASET_SLUG}"
    meta = {
        "title": "TXL-PBC blood cell detection dataset",
        "id": ref,
        "licenses": [{"name": "unknown"}],
    }
    (ds_dir / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))
    exists = kaggle(env, "datasets", "status", ref, check=False, capture=True).returncode == 0
    if exists:
        kaggle(env, "datasets", "version", "-p", str(ds_dir), "-m", args.message, "-r", "zip")
    else:
        kaggle(env, "datasets", "create", "-p", str(ds_dir), "-r", "zip")
    print("dataset:", ref)


def write_kernel_metadata(user: str, args) -> Path:
    kdir = STAGING / "kernel"
    if kdir.exists():
        shutil.rmtree(kdir)
    kdir.mkdir(parents=True)
    shutil.copy(KERNEL_SCRIPT, kdir / "train_yolo26.py")
    meta = {
        "id": f"{user}/{KERNEL_SLUG}",
        "title": "YOLO26 TXL-PBC training",
        "code_file": "train_yolo26.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [f"{user}/{DATASET_SLUG}"],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (kdir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    # Pass the training settings to the kernel through its environment
    settings = {
        "BC_MODEL": args.model,
        "BC_EPOCHS": str(args.epochs),
        "BC_IMGSZ": str(args.imgsz),
        "BC_BATCH": str(args.batch),
        "BC_PATIENCE": str(args.patience),
    }
    header = "import os\n" + "".join(f"os.environ.setdefault({k!r}, {v!r})\n" for k, v in settings.items())
    original = (kdir / "train_yolo26.py").read_text()
    (kdir / "train_yolo26.py").write_text(header + "\n" + original)
    (kdir / "run_settings.json").write_text(json.dumps(settings, indent=2))
    return kdir


def cmd_push(args, env):
    user = username(env)
    kdir = write_kernel_metadata(user, args)
    kaggle(env, "kernels", "push", "-p", str(kdir))
    print("kernel:", f"{user}/{KERNEL_SLUG}")


def cmd_status(args, env):
    user = username(env)
    res = kaggle(env, "kernels", "status", f"{user}/{KERNEL_SLUG}", check=False, capture=True)
    text = (res.stdout or "") + (res.stderr or "")
    print(text)
    # Record the state for the dashboard (RUNNING, COMPLETE, ERROR, QUEUED, ...)
    state = "unknown"
    for word in ("COMPLETE", "RUNNING", "QUEUED", "ERROR", "CANCEL", "NEW_SCRIPT"):
        if word in text.upper():
            state = word.lower().replace("new_script", "queued")
            break
    out = ROOT / "runs" / "kernel_status.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"status": state, "raw": text.strip()}, indent=2))


def cmd_pull(args, env):
    user = username(env)
    PULL_DIR.mkdir(parents=True, exist_ok=True)
    kaggle(env, "kernels", "output", f"{user}/{KERNEL_SLUG}", "-p", str(PULL_DIR))
    print("outputs in", PULL_DIR)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    u = sub.add_parser("upload-dataset")
    u.add_argument("--message", default="Update TXL-PBC dataset")
    u.set_defaults(fn=cmd_upload_dataset)

    t = sub.add_parser("push")
    t.add_argument("--model", default="yolo26s.pt")
    t.add_argument("--epochs", type=int, default=100)
    t.add_argument("--imgsz", type=int, default=640)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--patience", type=int, default=25)
    t.set_defaults(fn=cmd_push)

    s = sub.add_parser("status")
    s.set_defaults(fn=cmd_status)

    pl = sub.add_parser("pull")
    pl.set_defaults(fn=cmd_pull)
    return p


def main() -> None:
    args = build_parser().parse_args()
    env = load_env()
    args.fn(args, env)


if __name__ == "__main__":
    main()
