"""Export the figures the static site displays, straight from the artefacts.

The site is plain HTML and cannot import results/summary.json at runtime the
way the Python app did, so the numbers are exported here instead of being typed
into the markup. Same rule as before, enforced the same way: if a figure is not
in the committed benchmark, it does not reach the page.

    python scripts/build_site_data.py
"""

from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bench.config import CLASSES  # noqa: E402

SUMMARY = os.path.join(ROOT, "results", "summary.json")
SAMPLES = os.path.join(ROOT, "app", "samples", "index.json")
OUT = os.path.join(ROOT, "web", "data", "site.json")

REPORT_THREADS = 1
REPORT_BATCH = 1
NEAR_EQUAL_TOLERANCE = 0.01     # one percentage point, as a fraction


def main() -> None:
    summary = json.load(open(SUMMARY))
    samples = json.load(open(SAMPLES))

    rows = [v for v in summary["measured"].values()
            if v["threads"] == REPORT_THREADS
            and v["batch_size"] == REPORT_BATCH]
    with_energy = [r for r in rows if r.get("energy_j_per_1k")]

    best = max(with_energy, key=lambda r: r["test_acc"]["mean"])
    near = [r for r in with_energy
            if r["test_acc"]["mean"] >= best["test_acc"]["mean"] - NEAR_EQUAL_TOLERANCE]
    cheapest = min(near, key=lambda r: r["energy_j_per_1k"]["mean"])
    dearest = max(near, key=lambda r: r["energy_j_per_1k"]["mean"])
    accs = [v["mean"] * 100 for v in summary["accuracy_over_seeds"].values()]

    def config(r: dict) -> dict:
        return {
            "model": r["model"],
            "precision": r["precision"],
            "label": f'{r["model"]} {r["precision"]}',
            "accuracy": r["test_acc"]["mean"] * 100,
            "accuracy_ci": (r["test_acc"].get("half_width") or 0) * 100,
            "energy": r["energy_j_per_1k"]["mean"] if r.get("energy_j_per_1k") else None,
            "p95": r["latency_p95_ms"]["mean"],
            "size_mb": r["onnx_mb"],
            "co2e_per_1m": (r["co2e_g_per_1m"]["mean"]
                            if r.get("co2e_g_per_1m") else None),
        }

    quant = {}
    for key, d in summary.get("quantisation_delta", {}).items():
        quant[key] = {
            "fp32": d["fp32_mean"] * 100,
            "int8": d["quant_mean"] * 100,
            "delta": d["delta_mean"] * 100,
            "half_width": (d.get("delta_half_width") or 0) * 100,
        }

    ex = summary.get("exclusions", {})
    env = summary.get("environment", {})

    data = {
        "classes": list(CLASSES),
        "finding": {
            "ratio": dearest["energy_j_per_1k"]["mean"] / cheapest["energy_j_per_1k"]["mean"],
            # Two different gaps, and mixing them is the easiest mistake on the
            # site: `gap` is against the most accurate configuration measured,
            # `gap_vs_dearest` is against the one the energy ratio is quoted
            # against. A sentence naming the dearest model must use the latter.
            "gap": (best["test_acc"]["mean"] - cheapest["test_acc"]["mean"]) * 100,
            "gap_vs_dearest": (dearest["test_acc"]["mean"]
                               - cheapest["test_acc"]["mean"]) * 100,
            "spread": max(accs) - min(accs),
            "architectures": len(summary["accuracy_over_seeds"]),
            "seeds": max(v.get("n") or 0
                         for v in summary["accuracy_over_seeds"].values()),
            "cheapest": config(cheapest),
            "dearest": config(dearest),
            "best": config(best),
            "near_equal": [config(r) for r in
                           sorted(near, key=lambda r: r["energy_j_per_1k"]["mean"])],
            # dearest/cheapest are raw summary rows here, not the flattened
            # shape config() returns.
            "latency_ratio": (dearest["latency_p95_ms"]["mean"]
                              / cheapest["latency_p95_ms"]["mean"]),
            "size_ratio": (dearest["onnx_mb"] / cheapest["onnx_mb"]
                           if cheapest["onnx_mb"] else None),
        },
        "configs": [config(r) for r in rows],
        "quantisation": quant,
        "exclusions": {
            "excluded": ex.get("windows_excluded"),
            "total": ex.get("windows_total"),
            "windows": ex.get("excluded_windows", []),
        },
        "provenance": {
            "split_sha256": summary.get("split_sha256"),
            "recipe_hash": summary.get("recipe_hash"),
            "grid_intensity": summary.get("grid_intensity_g_co2e_per_kwh"),
            "grid_intensity_source": summary.get("grid_intensity_source"),
            "energy_note": summary.get("energy_note"),
            "environment": {k: env.get(k) for k in
                            ("cpu", "cpu_cores_logical", "os", "python",
                             "onnxruntime", "timestamp_utc")},
            "confound": summary.get("thread_regime_confound", {}),
        },
        # distinct_answers/models_wrong record how the zoo splits on each
        # tile; the demo says so rather than presenting every tile as equal.
        "tiles": [{"file": t["file"], "label": t["true_label"],
                   "split_path": t["split_path"],
                   "answers": t.get("distinct_answers"),
                   "wrong": t.get("models_wrong")}
                  for t in samples.get("tiles", [])],
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(data, fh, indent=1)

    print(f"wrote {os.path.relpath(OUT, ROOT)}")
    print(f"  energy ratio at equal accuracy: {data['finding']['ratio']:.2f}x")
    print(f"  accuracy given up:              {data['finding']['gap']:.2f} pp")
    print(f"  configurations:                 {len(data['configs'])}")
    print(f"  tiles:                          {len(data['tiles'])}")


if __name__ == "__main__":
    main()
