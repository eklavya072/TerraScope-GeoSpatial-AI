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


def power_state() -> dict:
    """macOS power/thermal state at measurement time.

    Low Power Mode throttles the CPU, and running on battery can engage
    additional power capping, so a latency or energy figure measured under
    either is not comparable to one measured without. Recording the state per
    run means a reader can tell which regime produced a number instead of
    taking the README's word for it.
    """
    state = {"low_power_mode": None, "power_source": None, "thermal_pressure": None}
    try:
        custom = subprocess.check_output(["pmset", "-g", "custom"], text=True)
        section = None
        modes = {}
        for line in custom.splitlines():
            if line.strip().endswith("Power:"):
                section = line.strip().rstrip(":").strip()
            elif "lowpowermode" in line and section:
                modes[section] = line.split()[-1]
        state["low_power_mode"] = modes or None
    except Exception:
        pass
    try:
        batt = subprocess.check_output(["pmset", "-g", "batt"], text=True)
        state["power_source"] = ("AC" if "AC Power" in batt else
                                 "battery" if "Battery Power" in batt else "unknown")
        state["battery_line"] = next(
            (l.strip() for l in batt.splitlines() if "InternalBattery" in l), None)
    except Exception:
        pass
    try:
        therm = subprocess.check_output(
            ["pmset", "-g", "therm"], text=True, stderr=subprocess.DEVNULL)
        # When nothing has throttled, pmset reports "No thermal warning level
        # has been recorded" -- which is itself the evidence we want, so the
        # whole output is kept rather than filtered for a warning that is
        # absent precisely when the run was clean.
        state["thermal_pressure"] = " | ".join(
            l.strip() for l in therm.splitlines() if l.strip()) or None
    except Exception:
        pass
    return state


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
        "power_state": power_state(),
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
