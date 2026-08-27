"""Export trained checkpoints to ONNX and produce int8 quantised variants.

Three precisions per model, all evaluated on the same committed test fold:

    fp32          the exported graph, unquantised
    int8_dynamic  weights quantised, activations quantised at run time
    int8_static   weights and activations quantised using a calibration set

The calibration set is drawn deterministically from the TRAIN fold only. Using
val or test images to calibrate would leak the evaluation data into the model
and inflate the post-quantisation accuracy we are trying to measure.

Usage:
    python -m bench.export_onnx --model all --seeds 0,1,2,3,4
"""

import argparse
import os

import numpy as np
import torch
from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType,
                                      quantize_dynamic, quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process

from bench.config import CHECKPOINT_DIR, INPUT_SIZE, MODEL_ZOO, RESULTS_DIR, SPLIT_CSV
from bench.data import EuroSATFold
from bench.models import build

ONNX_DIR = os.path.join(RESULTS_DIR, "onnx")
N_CALIB = 256
PRECISIONS = ("fp32", "int8_dynamic", "int8_static")


class FoldCalibrationReader(CalibrationDataReader):
    """Feeds calibration images from a fixed slice of the train fold."""

    def __init__(self, split_csv: str, n: int = N_CALIB):
        ds = EuroSATFold("train", split_csv)
        rng = np.random.default_rng(12345)          # fixed: calibration is reproducible
        idx = rng.choice(len(ds), size=n, replace=False)
        self.batches = iter([{"input": ds[int(i)][0].unsqueeze(0).numpy()} for i in idx])

    def get_next(self):
        return next(self.batches, None)


def export_one(name: str, seed: int, split_csv: str) -> dict[str, str]:
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{name}_seed{seed}.pt")
    if not os.path.exists(ckpt_path):
        raise SystemExit(f"missing checkpoint {ckpt_path} -- run bench.train first")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model = build(name, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    os.makedirs(ONNX_DIR, exist_ok=True)
    stem = os.path.join(ONNX_DIR, f"{name}_seed{seed}")
    fp32 = f"{stem}_fp32.onnx"

    dummy = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE)
    torch.onnx.export(
        model, dummy, fp32,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17, do_constant_folding=True,
        # dynamo=False selects the TorchScript exporter. torch 2.9 defaults to
        # the dynamo path, whose graphs carry shape metadata that ONNX Runtime's
        # quantiser rejects during shape inference ("Inferred shape and existing
        # shape differ in dimension 0: (2048) vs (10)"). Since every model here
        # must survive int8 quantisation, the exporter that produces
        # quantisable graphs is the one that matters.
        dynamo=False,
    )

    # Dynamic: weights only, no calibration data needed.
    dyn = f"{stem}_int8_dynamic.onnx"
    quantize_dynamic(fp32, dyn, weight_type=QuantType.QInt8)

    # Static: needs shape inference first, then activation calibration.
    prep = f"{stem}_prep.onnx"
    quant_pre_process(fp32, prep, skip_symbolic_shape=False)
    stat = f"{stem}_int8_static.onnx"
    quantize_static(prep, stat, FoldCalibrationReader(split_csv),
                    quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8)
    os.remove(prep)

    sizes = {p: os.path.getsize(f) for p, f in
             (("fp32", fp32), ("int8_dynamic", dyn), ("int8_static", stat))}
    print(f"  {name} seed{seed}: " +
          "  ".join(f"{p}={s/1e6:.1f}MB" for p, s in sizes.items()), flush=True)
    return {"fp32": fp32, "int8_dynamic": dyn, "int8_static": stat}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="all")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--split", default=SPLIT_CSV)
    args = ap.parse_args()

    names = list(MODEL_ZOO) if args.model == "all" else args.model.split(",")
    for name in names:
        for seed in (int(s) for s in args.seeds.split(",")):
            export_one(name, seed, args.split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
