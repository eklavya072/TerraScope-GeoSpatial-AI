"""Aggregate results into the tables and figures the README publishes.

Reads results/runs.jsonl and results/bench.jsonl, and writes:
    results/summary.json        machine-readable aggregate (CC-BY-4.0 data)
    results/results_table.md    accuracy / latency / memory / energy table
    results/significance.md     which accuracy differences survive correction
    results/pareto.png          accuracy vs energy, frontier marked

No number is computed here that was not measured by bench.train or
bench.benchmark; this module only aggregates.

Usage:
    python -m bench.report
"""

import argparse
import collections
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bench.config import (GRID_INTENSITY_G_CO2E_PER_KWH, GRID_INTENSITY_SOURCE,
                          RESULTS_DIR)
from bench.stats import compare, holm_bonferroni, mean_ci, pareto_frontier

RUNS = os.path.join(RESULTS_DIR, "runs.jsonl")
BENCH = os.path.join(RESULTS_DIR, "bench.jsonl")
PRIMARY_THREADS = 1
PRIMARY_BATCH = 1


def read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def fmt_pct(ci: dict) -> str:
    if ci["half_width"] is None:
        return f"{ci['mean']*100:.2f} (n=1)"
    return f"{ci['mean']*100:.2f} ± {ci['half_width']*100:.2f}"


def build(args) -> dict:
    runs = [r for r in read_jsonl(RUNS) if r.get("kind") == "train"]
    bench = [b for b in read_jsonl(BENCH) if b.get("kind") == "bench"]
    if not runs:
        raise SystemExit("no training runs in results/runs.jsonl")

    # ---------------- accuracy over seeds (from the fp32 torch runs) --------
    by_model = collections.defaultdict(list)
    for r in runs:
        by_model[r["model"]].append(r)

    accuracy = {}
    for model, rows in sorted(by_model.items()):
        accs = [r["test_acc"] for r in rows]
        accuracy[model] = {
            **mean_ci(accs),
            "seeds": sorted(r["seed"] for r in rows),
            "params": rows[0]["params"],
            "raw": accs,
        }

    # ---------------- pairwise significance --------------------------------
    names = [m for m in sorted(accuracy) if accuracy[m]["n"] >= 2]
    pairs = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pairs[(a, b)] = compare(accuracy[a]["raw"], accuracy[b]["raw"])
    corrected = holm_bonferroni(pairs) if pairs else {}

    # ---------------- measured deployment metrics --------------------------
    by_cfg = collections.defaultdict(list)
    for b in bench:
        by_cfg[(b["model"], b["precision"], b["threads_intra_op"],
                b["batch_size"])].append(b)

    measured = {}
    for (model, precision, threads, bs), rows in sorted(by_cfg.items()):
        e = [r["energy_joules_per_1k_inferences"] for r in rows
             if r.get("energy_joules_per_1k_inferences") is not None]
        measured[f"{model}|{precision}|t{threads}|b{bs}"] = {
            "model": model, "precision": precision, "threads": threads,
            "batch_size": bs,
            "test_acc": mean_ci([r["test_acc"] for r in rows]),
            "latency_p50_ms": mean_ci([r["latency_ms_p50"] for r in rows]),
            "latency_p95_ms": mean_ci([r["latency_ms_p95"] for r in rows]),
            "latency_p99_ms": mean_ci([r["latency_ms_p99"] for r in rows]),
            "rss_peak_mb": mean_ci([r["rss_peak_bytes"] / 1e6 for r in rows]),
            "onnx_mb": rows[0]["onnx_bytes"] / 1e6,
            "energy_j_per_1k": mean_ci(e) if e else None,
            # J -> kWh (/3.6e6) -> gCO2e. Linear in the stated grid intensity.
            "co2e_g_per_1k": (
                mean_ci([j / 3.6e6 * args.grid_intensity for j in e]) if e else None),
            "n_seeds": len(rows),
        }

    summary = {
        "accuracy_over_seeds": accuracy,
        "pairwise_significance": {f"{a} vs {b}": v for (a, b), v in corrected.items()},
        "measured": measured,
        "environment": runs[-1]["env"],
        "split_sha256": runs[-1]["split_sha256"],
        "recipe_hash": runs[-1]["recipe_hash"],
        "recipe": runs[-1]["recipe"],
        "grid_intensity_g_co2e_per_kwh": args.grid_intensity,
        "grid_intensity_source": GRID_INTENSITY_SOURCE,
        "energy_note": (
            "Energy is ESTIMATED from Apple Silicon on-die CPU package power "
            "telemetry sampled by powermetrics and integrated over each timed "
            "window. It excludes DRAM, display and PSU losses, and is not a "
            "wall-socket measurement."
        ),
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")

    write_tables(summary, args)
    write_pareto(summary, args)
    return summary


def write_tables(summary: dict, args) -> None:
    acc = summary["accuracy_over_seeds"]
    measured = summary["measured"]

    lines = ["| Model | Precision | Params | Accuracy % (mean ± 95% CI) | "
             "p95 latency (ms) | Peak RSS (MB) | Model (MB) | "
             "Energy/1k inf (J, estimated) | CO2e/1k inf (g, estimated) |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key in sorted(measured):
        m = measured[key]
        if m["threads"] != args.threads or m["batch_size"] != args.batch:
            continue
        e, c = m["energy_j_per_1k"], m["co2e_g_per_1k"]
        lines.append(
            f"| {m['model']} | {m['precision']} | "
            f"{acc.get(m['model'], {}).get('params', 0)/1e6:.2f}M | "
            f"{fmt_pct(m['test_acc'])} | {m['latency_p95_ms']['mean']:.2f} | "
            f"{m['rss_peak_mb']['mean']:.0f} | {m['onnx_mb']:.1f} | "
            f"{'not measured' if not e else format(e['mean'], '.2f')} | "
            f"{'not measured' if not c else format(c['mean'], '.4f')} |")

    header = (f"### Results (ONNX Runtime CPU EP, intra-op threads="
              f"{args.threads}, batch={args.batch})\n\n"
              f"Accuracy is the mean over {max((a['n'] for a in acc.values()), default=0)}"
              f" seeds with a Student-t 95% confidence interval. Energy figures are "
              f"ESTIMATED from on-die power telemetry, not metered at the wall. "
              f"CO2e assumes {args.grid_intensity:.0f} gCO2e/kWh "
              f"({GRID_INTENSITY_SOURCE.split(';')[0]}).\n")
    with open(os.path.join(RESULTS_DIR, "results_table.md"), "w") as fh:
        fh.write(header + "\n" + "\n".join(lines) + "\n")

    sig = summary["pairwise_significance"]
    out = ["### Which accuracy differences are statistically distinguishable?\n",
           "Welch's t-test over seeds, Holm-Bonferroni corrected across all "
           "pairwise comparisons (family-wise alpha = 0.05).\n",
           "| Comparison | Δ accuracy (pp) | p | Holm threshold | Distinguishable? |",
           "|---|---:|---:|---:|---|"]
    for name, r in sorted(sig.items(), key=lambda kv: kv[1]["p"]):
        out.append(f"| {name} | {r['diff']*100:+.2f} | {r['p']:.4f} | "
                   f"{r['holm_threshold']:.4f} | "
                   f"{'**yes**' if r['significant_holm'] else 'no'} |")
    n_sig = sum(1 for r in sig.values() if r["significant_holm"])
    out.append(f"\n{n_sig} of {len(sig)} pairwise accuracy differences are "
               f"statistically distinguishable after correction.\n")
    with open(os.path.join(RESULTS_DIR, "significance.md"), "w") as fh:
        fh.write("\n".join(out))


def write_pareto(summary: dict, args) -> None:
    pts = []
    for key, m in summary["measured"].items():
        if m["threads"] != args.threads or m["batch_size"] != args.batch:
            continue
        e = m["energy_j_per_1k"]
        if not e:
            continue
        pts.append({"label": f"{m['model']}\n{m['precision']}",
                    "model": m["model"], "precision": m["precision"],
                    "energy": e["mean"], "acc": m["test_acc"]["mean"],
                    "acc_err": m["test_acc"]["half_width"] or 0.0})
    if not pts:
        print("no energy data yet -- skipping Pareto figure", flush=True)
        return

    front = set(pareto_frontier(pts, "energy", "acc"))
    fig, ax = plt.subplots(figsize=(9, 6))
    for p in pts:
        on = p["label"] in front
        ax.errorbar(p["energy"], p["acc"] * 100, yerr=p["acc_err"] * 100,
                    fmt="o", ms=11 if on else 7,
                    color="#c1440e" if on else "#7a7a7a",
                    ecolor="#999", capsize=3, zorder=3 if on else 2)
        ax.annotate(p["label"], (p["energy"], p["acc"] * 100),
                    textcoords="offset points", xytext=(8, 5), fontsize=8)
    chain = sorted([p for p in pts if p["label"] in front], key=lambda p: p["energy"])
    if len(chain) > 1:
        ax.plot([p["energy"] for p in chain], [p["acc"] * 100 for p in chain],
                "--", color="#c1440e", lw=1.5, zorder=1, label="Pareto frontier")
        ax.legend(loc="lower right")
    ax.set_xscale("log")
    ax.set_xlabel("Energy per 1,000 inferences (J, estimated from on-die telemetry)")
    ax.set_ylabel("Test accuracy (%), mean ± 95% CI over seeds")
    ax.set_title(f"TerraScope: accuracy vs energy on CPU\n"
                 f"{summary['environment']['cpu']}, ONNX Runtime CPU EP, "
                 f"{args.threads} thread(s), batch {args.batch}", fontsize=11)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "pareto.png"), dpi=160)
    print(f"wrote {RESULTS_DIR}/pareto.png (frontier: {sorted(front)})", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--threads", type=int, default=PRIMARY_THREADS)
    ap.add_argument("--batch", type=int, default=PRIMARY_BATCH)
    ap.add_argument("--grid-intensity", type=float,
                    default=GRID_INTENSITY_G_CO2E_PER_KWH,
                    help="gCO2e/kWh used for the CO2e columns")
    args = ap.parse_args()
    s = build(args)
    print(f"models: {list(s['accuracy_over_seeds'])}")
    for m, a in s["accuracy_over_seeds"].items():
        print(f"  {m:20s} acc={fmt_pct(a)}%  n={a['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
