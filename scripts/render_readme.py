"""Inject measured results into README.md between generated markers.

Every number in the README's results section is written by this script from
results/summary.json. Nothing in those sections is hand-typed, so a README
figure cannot drift away from the run that produced it -- the failure mode
where a paper's abstract quotes a number the tables no longer support.

Prose OUTSIDE the markers is written by a human and left untouched.

Usage:
    python scripts/render_readme.py
"""

import json
import os
import re
import sys

SUMMARY = os.path.join("results", "summary.json")
README = "README.md"

BLOCK = "<!-- BEGIN:{name} -->{body}<!-- END:{name} -->"
PATTERN = "<!-- BEGIN:{name} -->.*?<!-- END:{name} -->"


def render_results_table() -> str:
    path = os.path.join("results", "results_table.md")
    return open(path).read() if os.path.exists(path) else "_not yet measured_\n"


def render_quantisation() -> str:
    path = os.path.join("results", "quantisation.md")
    return open(path).read() if os.path.exists(path) else "_not yet measured_\n"


def render_significance() -> str:
    path = os.path.join("results", "significance.md")
    return open(path).read() if os.path.exists(path) else "_not yet measured_\n"


def render_hardware(summary: dict) -> str:
    env = summary["environment"]
    ps = env.get("power_state") or {}
    lpm = ps.get("low_power_mode") or {}
    lines = [
        "All measurements in this repository come from ONE machine. Latency,",
        "memory and energy figures are properties of the model AND this hardware;",
        "they are not portable claims.",
        "",
        f"| Property | Value |",
        f"|---|---|",
        f"| CPU | {env['cpu']} |",
        f"| Cores | {env['cpu_cores_physical']} physical / {env['cpu_cores_logical']} logical |",
        f"| RAM | {int(env['ram_bytes'])/2**30:.0f} GiB |",
        f"| OS | {env['os']} |",
        f"| Python | {env['python']} |",
        f"| PyTorch | {env['torch']} |",
        f"| ONNX Runtime | {env['onnxruntime']} |",
        f"| timm | {env['timm']} |",
        f"| NumPy | {env['numpy']} |",
        f"| Measurement date (UTC) | {env['timestamp_utc']} |",
        f"| Power source during measurement | {ps.get('power_source', 'not recorded')} |",
        f"| macOS Low Power Mode | {lpm.get('AC Power', 'not recorded')} (AC) |",
        f"| Training environment (affects no reported figure) | "
        f"{(summary.get('training_environment') or {}).get('timestamp_utc', 'n/a')}, "
        f"{((summary.get('training_environment') or {}).get('power_state') or {}).get('power_source', 'n/a')} |",
        "",
        f"Split file: `{summary.get('split_csv', 'splits/eurosat_split_seed42.csv')}`  ",
        f"Split sha256: `{summary['split_sha256']}`  ",
        f"Recipe hash: `{summary['recipe_hash']}`",
        "",
        "Inference is measured through ONNX Runtime's **CPU execution provider only**.",
        "Apple's GPU (MPS) and CoreML providers are excluded deliberately, not merely",
        "left unused: the question is what CPU-only hardware achieves. Training used",
        "the GPU, which affects no reported figure -- training cost is not part of the",
        "deployment claim being made.",
        "",
    ]
    return "\n".join(lines)


def render_recipe(summary: dict) -> str:
    r = summary["recipe"]
    rows = [f"| {k} | `{v}` |" for k, v in sorted(r.items())]
    return ("Every architecture is trained under this identical recipe. There is no\n"
            "supported way to give one model a tuned recipe of its own.\n\n"
            "| Setting | Value |\n|---|---|\n" + "\n".join(rows) + "\n")


def main() -> int:
    if not os.path.exists(SUMMARY):
        sys.exit(f"missing {SUMMARY} -- run `make report` first")
    summary = json.load(open(SUMMARY))
    text = open(README).read()

    blocks = {
        "results_table": render_results_table(),
        "quantisation": render_quantisation(),
        "significance": render_significance(),
        "hardware": render_hardware(summary),
        "recipe": render_recipe(summary),
    }

    missing = []
    for name, body in blocks.items():
        pat = re.compile(PATTERN.format(name=name), re.DOTALL)
        if not pat.search(text):
            missing.append(name)
            continue
        text = pat.sub(lambda _m, n=name, b=body:
                       BLOCK.format(name=n, body="\n" + b.strip() + "\n"), text)

    if missing:
        print(f"WARNING: no marker block for: {', '.join(missing)}", flush=True)

    open(README, "w").write(text)
    print(f"rendered {len(blocks) - len(missing)} block(s) into {README}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
