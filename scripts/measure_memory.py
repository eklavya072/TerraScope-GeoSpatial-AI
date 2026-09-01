"""Measure per-model memory footprint in an ISOLATED process.

Why this is separate from bench.benchmark: psutil reports the RESIDENT SET of
the whole process, and the benchmark process also holds the decoded test fold
(~130 MB) plus every InferenceSession it has already built. Peak RSS sampled
there is dominated by harness state and grows monotonically through the run --
it told us MobileNetV3-Small (1.5M params) used more memory than ResNet-50
(23.5M), which is an artefact, not a measurement.

Here each (model, precision) is measured in a fresh subprocess that loads
nothing but the ONNX session and one input, so the delta from baseline is
attributable to the model.

Usage:
    python scripts/measure_memory.py
"""

import json
import os
import subprocess
import sys

sys.path.insert(0, os.getcwd())
from bench.config import MODEL_ZOO, RESULTS_DIR          # noqa: E402
from bench.export_onnx import ONNX_DIR, PRECISIONS       # noqa: E402

OUT = os.path.join(RESULTS_DIR, "memory.jsonl")
SEED = 0        # footprint is a property of the graph, not of the seed

CHILD = r'''
import json, os, sys
import numpy as np, psutil, onnxruntime as ort
path = sys.argv[1]
proc = psutil.Process()
baseline = proc.memory_info().rss
opts = ort.SessionOptions()
opts.intra_op_num_threads = 1
opts.inter_op_num_threads = 1
sess = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
x = np.zeros((1, 3, 64, 64), dtype=np.float32)
peak = 0
for _ in range(50):
    sess.run(None, {"input": x})
    peak = max(peak, proc.memory_info().rss)
print(json.dumps({"baseline_bytes": baseline, "peak_bytes": peak,
                  "model_bytes": peak - baseline}))
'''


def main() -> int:
    rows = []
    for name in MODEL_ZOO:
        for precision in PRECISIONS:
            path = os.path.join(ONNX_DIR, f"{name}_seed{SEED}_{precision}.onnx")
            if not os.path.exists(path):
                continue
            out = subprocess.check_output(
                [sys.executable, "-c", CHILD, path], text=True,
                env={**os.environ, "OMP_NUM_THREADS": "1"})
            data = json.loads(out.strip().splitlines()[-1])
            row = {"kind": "memory", "model": name, "precision": precision,
                   "seed": SEED, "onnx_bytes": os.path.getsize(path),
                   "isolated_process": True, **data}
            rows.append(row)
            print(f"  {name:20s} {precision:13s} "
                  f"rss_delta={data['model_bytes']/1e6:7.1f}MB  "
                  f"peak={data['peak_bytes']/1e6:7.1f}MB  "
                  f"file={row['onnx_bytes']/1e6:6.1f}MB", flush=True)

    with open(OUT, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    print(f"wrote {OUT} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
