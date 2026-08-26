"""Materialise the EuroSAT RGB corpus to disk with original JPEG bytes intact.

Source: the `giswqs/EuroSAT_RGB` mirror on the Hugging Face Hub, which carries
the 27,000 RGB tiles of EuroSAT (Helber et al.) together with each tile's
original `filename`. We deliberately read the images with `decode=False` so the
bytes written here are byte-identical to the ones in the upstream archive --
re-encoding through PIL would silently change every hash and make the manifest
meaningless.

The mirror's own train/validation/test folds are IGNORED: their provenance is
undocumented and they are not reproducible from a seed. `scripts/make_split.py`
generates the split this benchmark actually uses.

Output:
    data/eurosat_rgb/<ClassName>/<ClassName>_<n>.jpg   (27,000 files)
    data/eurosat_rgb/MANIFEST.sha256                   (one line per file)

Usage:
    python scripts/prepare_data.py
"""

import hashlib
import os
import sys

from datasets import Image as HFImage
from datasets import load_dataset

REPO = "giswqs/EuroSAT_RGB"
OUT_ROOT = os.path.join("data", "eurosat_rgb")
EXPECTED_N = 27_000


def main() -> int:
    print(f"[1/3] loading {REPO} (undecoded bytes) ...", flush=True)
    ds = load_dataset(REPO)
    written, records = 0, []

    print(f"[2/3] writing tiles -> {OUT_ROOT}/ ...", flush=True)
    for split in ("train", "validation", "test"):
        part = ds[split].cast_column("image", HFImage(decode=False))
        classes = part.features["label"].names
        for row in part:
            rel = row["filename"]                      # e.g. AnnualCrop/AnnualCrop_1.jpg
            label = classes[row["label"]]
            assert rel.split("/")[0] == label, f"label/path mismatch: {rel} vs {label}"
            dest = os.path.join(OUT_ROOT, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            raw = row["image"]["bytes"]
            if raw is None:                            # mirror stored a path, not bytes
                with open(row["image"]["path"], "rb") as fh:
                    raw = fh.read()
            with open(dest, "wb") as fh:
                fh.write(raw)
            records.append((hashlib.sha256(raw).hexdigest(), rel))
            written += 1
            if written % 5000 == 0:
                print(f"    {written} / {EXPECTED_N}", flush=True)

    if written != EXPECTED_N:
        print(f"FATAL: wrote {written} tiles, expected {EXPECTED_N}", file=sys.stderr)
        return 1
    if len({rel for _, rel in records}) != EXPECTED_N:
        print("FATAL: duplicate filenames in corpus", file=sys.stderr)
        return 1

    print("[3/3] writing MANIFEST.sha256 ...", flush=True)
    records.sort(key=lambda r: r[1])
    manifest = os.path.join(OUT_ROOT, "MANIFEST.sha256")
    with open(manifest, "w") as fh:
        for digest, rel in records:
            fh.write(f"{digest}  {rel}\n")

    with open(manifest, "rb") as fh:
        corpus_hash = hashlib.sha256(fh.read()).hexdigest()
    print(f"OK  tiles={written}  manifest_sha256={corpus_hash}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
