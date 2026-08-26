#!/usr/bin/env bash
# Continuous CPU package power sampler for the benchmark session.
#
# WHY THIS EXISTS: codecarbon reads Intel RAPL, which does not exist on Apple
# Silicon. On this hardware it falls back to a constant fraction of a hardcoded
# TDP, which makes its "energy" a linear function of runtime and therefore
# carries no information beyond latency. `powermetrics` reads the M-series
# on-die power telemetry directly, so it is the only defensible energy source
# available here -- but it requires root, which is why this is a script you run
# rather than something the harness invokes for you.
#
# Usage (run in a separate terminal BEFORE `make bench`):
#     sudo ./scripts/energy_sampler.sh
#
# Stop it with Ctrl-C once the benchmark finishes. The harness correlates its
# per-measurement timestamps against this log; it does not need the sampler to
# start or stop at any particular moment, only to span the whole session.

set -euo pipefail

OUT="${1:-results/power_log.txt}"
INTERVAL_MS="${INTERVAL_MS:-200}"

if [[ "$(id -u)" -ne 0 ]]; then
    echo "error: powermetrics requires root. Re-run as: sudo $0" >&2
    exit 1
fi

mkdir -p "$(dirname "$OUT")"
echo "sampling CPU package power every ${INTERVAL_MS}ms -> $OUT"
echo "leave this running for the whole benchmark session; Ctrl-C when done"

# --samplers cpu_power gives "CPU Power: N mW"; each sample block is preceded by
# a "*** Sampled system activity (<date>) ***" header that the parser uses to
# place samples on the wall clock.
exec powermetrics --samplers cpu_power -i "$INTERVAL_MS" -f text \
    | tee "$OUT"
