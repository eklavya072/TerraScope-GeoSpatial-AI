"""Parse `powermetrics` output and integrate power over a measurement window.

The energy figures in this benchmark come from Apple Silicon's on-die power
telemetry, read by `sudo powermetrics` (see scripts/energy_sampler.sh) and
integrated over the wall-clock window of each measured inference run. This is
still a telemetry reading rather than a wall-socket measurement -- it reports
CPU package power and excludes DRAM, display and PSU losses -- so results are
labelled "estimated" throughout. It is nonetheless a real measurement of a real
sensor, which codecarbon's TDP fallback on this platform is not.
"""

import bisect
import os
import re
from datetime import datetime

HEADER = re.compile(r"\*\*\* Sampled system activity \((.+?)\) \(([\d.]+)ms elapsed\)")
POWER = re.compile(r"^(CPU|Package) Power:\s+([\d.]+)\s*mW", re.MULTILINE)


def parse_log(path: str) -> list[tuple[float, float]]:
    """-> [(unix_timestamp, cpu_power_mW), ...] sorted by time."""
    if not os.path.exists(path):
        return []
    samples: list[tuple[float, float]] = []
    stamp: float | None = None
    with open(path, errors="replace") as fh:
        for line in fh:
            m = HEADER.search(line)
            if m:
                try:
                    stamp = datetime.strptime(
                        m.group(1).strip(), "%a %b %d %H:%M:%S %Y %z").timestamp()
                except ValueError:
                    stamp = None
                continue
            p = POWER.match(line.strip())
            if p and stamp is not None and p.group(1) == "CPU":
                samples.append((stamp, float(p.group(2))))
                stamp = None
    samples.sort()
    return samples


def energy_joules(samples: list[tuple[float, float]], t0: float, t1: float) -> dict:
    """Integrate sampled power (mW) over [t0, t1] -> joules, plus coverage.

    `coverage` is the fraction of the window actually spanned by samples. A
    window with poor coverage (sampler started late, or died) must not be
    reported as a measurement, so the caller checks it rather than silently
    integrating over a gap.
    """
    if not samples or t1 <= t0:
        return {"joules": None, "mean_power_w": None, "n_samples": 0, "coverage": 0.0}

    times = [s[0] for s in samples]
    lo = bisect.bisect_left(times, t0)
    hi = bisect.bisect_right(times, t1)
    window = samples[lo:hi]
    if len(window) < 2:
        return {"joules": None, "mean_power_w": None, "n_samples": len(window),
                "coverage": 0.0}

    joules = 0.0
    for (ta, pa), (tb, pb) in zip(window, window[1:]):
        joules += (pa + pb) / 2.0 / 1000.0 * (tb - ta)   # trapezoid, mW -> W
    span = window[-1][0] - window[0][0]
    return {
        "joules": joules,
        "mean_power_w": (joules / span) if span > 0 else None,
        "n_samples": len(window),
        "coverage": round(span / (t1 - t0), 3),
    }
