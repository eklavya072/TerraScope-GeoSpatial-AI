"""Seeding, environment capture and split integrity checks.

A benchmark result is only meaningful alongside the environment that produced
it, so every run records its hardware and library versions rather than trusting
the README to stay in sync.
"""

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed every RNG that can influence a training run."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)  # MPS lacks deterministic kernels


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sysctl(key: str) -> str:
    try:
        return subprocess.check_output(["sysctl", "-n", key], text=True).strip()
    except Exception:
        return "unknown"


def environment() -> dict:
    """Hardware + software fingerprint recorded with every result row."""
    import onnxruntime as ort
    import timm

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cpu": _sysctl("machdep.cpu.brand_string"),
        "cpu_cores_physical": _sysctl("hw.physicalcpu"),
        "cpu_cores_logical": _sysctl("hw.logicalcpu"),
        "ram_bytes": _sysctl("hw.memsize"),
        "os": f"{platform.system()} {platform.release()} ({platform.version().split(':')[0]})",
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "timm": timm.__version__,
        "onnxruntime": ort.__version__,
        "numpy": np.__version__,
    }


def verify_split(split_csv: str) -> str:
    """Return the split's sha256, checking it against the committed .sha256 file.

    Every result references the split by hash. If the split file is edited
    without regenerating its sidecar, runs stop here rather than silently
    producing numbers that reference a fold definition nobody can recover.
    """
    digest = sha256_file(split_csv)
    sidecar = split_csv + ".sha256"
    if os.path.exists(sidecar):
        expected = open(sidecar).read().split()[0]
        if expected != digest:
            raise SystemExit(
                f"split file {split_csv} does not match {sidecar}\n"
                f"  expected {expected}\n  actual   {digest}\n"
                "Regenerate with scripts/make_split.py or restore the committed split."
            )
    return digest


def append_jsonl(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
