"""Verify results/ regenerates from the raw measurements, across platforms.

`make report` derives summary.json and the published tables from
results/bench.jsonl and results/runs.jsonl. If a derived artefact were edited by
hand, or drifted from the data it claims to summarise, nothing else in this
repository would notice. So CI regenerates everything and compares.

A byte-for-byte comparison is the obvious way to do that and it is the wrong
one. scipy's Welch p-values differ in the last one or two units in the last
place between macOS/arm64 and Linux/x86_64 -- different libm, same arithmetic
intent:

    macOS   "p": 0.0002170793405032607
    Linux   "p": 0.0002170793405032606

Ten p-values in summary.json differ that way and nothing else does. Insisting on
identical bits would mean this check only ever passes on the maintainer's
laptop, which is the opposite of what it exists to prove.

So the invariant is split into the two claims that are actually true:

  * The PUBLISHED tables and README must match byte for byte. They are rounded
    to 2-4 decimals, so floating-point noise cannot reach them, and they are
    what a reader actually sees.
  * summary.json must match STRUCTURALLY and to floating-point tolerance: same
    keys, same non-numeric values, and every float within REL_TOL. A flipped
    `significant_holm`, a changed exclusion count, a moved accuracy figure -- all
    still fail, because those are either non-floats or differences far larger
    than tolerance.

Usage:
    python scripts/check_derived.py            # compare against HEAD
"""

import json
import math
import subprocess
import sys

# ~1e-9 relative. Six orders of magnitude above the ULP noise this exists to
# tolerate (~1e-16 relative), and many orders below any difference that would
# change a reported figure: the tables round to 4 decimals at most.
REL_TOL = 1e-9

EXACT = ("README.md", "results/results_table.md",
         "results/significance.md", "results/quantisation.md")
TOLERANT = "results/summary.json"


def committed(path: str) -> str:
    r = subprocess.run(["git", "show", f"HEAD:{path}"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"cannot read HEAD:{path} -- {r.stderr.strip()}")
    return r.stdout


def compare(a, b, path="") -> list[str]:
    """Structural comparison; floats within REL_TOL, everything else exact."""
    where = path or "<root>"
    if type(a) is not type(b) and not (isinstance(a, (int, float))
                                       and isinstance(b, (int, float))):
        return [f"{where}: type {type(a).__name__} -> {type(b).__name__}"]
    if isinstance(a, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k not in a:
                out.append(f"{where}.{k}: added")
            elif k not in b:
                out.append(f"{where}.{k}: removed")
            else:
                out += compare(a[k], b[k], f"{path}.{k}")
        return out
    if isinstance(a, list):
        if len(a) != len(b):
            return [f"{where}: length {len(a)} -> {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b))
                for d in compare(x, y, f"{path}[{i}]")]
    # bool before float: bool is a subclass of int, and a flipped
    # significant_holm must never be waved through as a numeric near-miss.
    if isinstance(a, bool) or isinstance(b, bool):
        return [] if a is b else [f"{where}: {a} -> {b}"]
    if isinstance(a, (int, float)):
        if math.isnan(a) and math.isnan(b):
            return []
        if math.isclose(a, b, rel_tol=REL_TOL, abs_tol=0.0):
            return []
        return [f"{where}: {a!r} -> {b!r} (exceeds rel_tol={REL_TOL})"]
    return [] if a == b else [f"{where}: {a!r} -> {b!r}"]


def main() -> int:
    failures = []

    for path in EXACT:
        with open(path) as fh:
            current = fh.read()
        if current != committed(path):
            failures.append(
                f"{path} differs from HEAD. It is generated -- run `make report` "
                f"and commit the result rather than editing it by hand.")

    diffs = compare(json.loads(committed(TOLERANT)),
                    json.load(open(TOLERANT)))
    if diffs:
        failures.append(f"{TOLERANT}: {len(diffs)} value(s) differ beyond "
                        f"floating-point tolerance:")
        failures += [f"    {d}" for d in diffs[:20]]
        if len(diffs) > 20:
            failures.append(f"    ... and {len(diffs) - 20} more")

    if failures:
        print("\n".join(failures))
        return 1
    print(f"derived artefacts reproduce: {len(EXACT)} file(s) byte-identical, "
          f"{TOLERANT} within rel_tol={REL_TOL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
