"""Parse `powermetrics` output and integrate power over a measurement window.

The energy figures in this benchmark come from Apple Silicon's on-die power
telemetry, read by `sudo powermetrics` (see scripts/energy_sampler.sh) and
integrated over the wall-clock window of each measured inference run. This is
still a telemetry reading rather than a wall-socket measurement -- it reports
CPU package power and excludes DRAM, display and PSU losses -- so results are
labelled "estimated" throughout. It is nonetheless a real measurement of a real
sensor, which codecarbon's TDP fallback on this platform is not.

Two properties of powermetrics' text output drive the design here:

1. Sample HEADERS carry a wall-clock timestamp with only ONE-SECOND resolution,
   while the sampler runs at 200ms. Five consecutive samples therefore share a
   timestamp, and integrating by timestamp difference gives a zero-width
   trapezoid -- silently yielding 0 J for any window shorter than a second or
   two. Each header also reports the interval it actually covered
   ("202.72ms elapsed"), so energy is accumulated as power x elapsed instead.
2. The log is APPENDED to for the whole session, so it must be re-read as
   measurements proceed (see PowerLog).
"""

import os
import re
from datetime import datetime

HEADER = re.compile(r"\*\*\* Sampled system activity \((.+?)\) \(([\d.]+)ms elapsed\)")
POWER = re.compile(r"^(CPU|Package) Power:\s+([\d.]+)\s*mW", re.MULTILINE)

# (timestamp_seconds, power_mW, elapsed_seconds)
Sample = tuple[float, float, float]


def _parse_lines(lines, stamp, elapsed) -> tuple[list[Sample], float | None, float]:
    out: list[Sample] = []
    for line in lines:
        m = HEADER.search(line)
        if m:
            try:
                stamp = datetime.strptime(
                    m.group(1).strip(), "%a %b %d %H:%M:%S %Y %z").timestamp()
                elapsed = float(m.group(2)) / 1000.0
            except ValueError:
                stamp = None
            continue
        p = POWER.match(line.strip())
        if p and stamp is not None and p.group(1) == "CPU":
            out.append((stamp, float(p.group(2)), elapsed))
            stamp = None
    return out, stamp, elapsed


def parse_log(path: str) -> list[Sample]:
    if not os.path.exists(path):
        return []
    with open(path, errors="replace") as fh:
        samples, _, _ = _parse_lines(fh.read().split("\n"), None, 0.0)
    samples.sort()
    return samples


def energy_joules(samples: list[Sample], t0: float, t1: float) -> dict:
    """Sum power x elapsed over samples whose timestamp lies in [t0, t1].

    `coverage` is the fraction of the window actually accounted for by sample
    intervals. Because timestamps are quantised to whole seconds, a window
    shorter than a few seconds cannot be attributed accurately -- the caller
    sizes its measurement windows so that coverage lands near 1.0, and a low
    value here is the signal that a number must not be reported as measured.
    """
    if not samples or t1 <= t0:
        return {"joules": None, "mean_power_w": None, "n_samples": 0, "coverage": 0.0}

    window = [s for s in samples if t0 <= s[0] <= t1]
    if not window:
        return {"joules": None, "mean_power_w": None, "n_samples": 0, "coverage": 0.0}

    joules = sum(p / 1000.0 * dt for _, p, dt in window)
    covered = sum(dt for _, _, dt in window)
    return {
        "joules": joules,
        "mean_power_w": (joules / covered) if covered > 0 else None,
        "n_samples": len(window),
        "coverage": round(covered / (t1 - t0), 3),
    }


class PowerLog:
    """Incrementally-read view of a live powermetrics log.

    Reading resumes from the previous byte offset and stops at the last
    complete line, so a sample being written while we read is picked up on the
    next update rather than parsed as a truncated fragment.
    """

    def __init__(self, path: str):
        self.path = path
        self._offset = 0
        self._pending = ""
        self._stamp: float | None = None
        self._elapsed = 0.0
        self.samples: list[Sample] = []
        self.update()

    def update(self) -> int:
        if not os.path.exists(self.path):
            return 0
        with open(self.path, errors="replace") as fh:
            fh.seek(self._offset)
            chunk = fh.read()
            self._offset = fh.tell()

        lines = (self._pending + chunk).split("\n")
        self._pending = lines.pop()          # incomplete tail, retried next time
        new, self._stamp, self._elapsed = _parse_lines(
            lines, self._stamp, self._elapsed)
        self.samples.extend(new)
        return len(new)

    def energy(self, t0: float, t1: float) -> dict:
        """Integrate over [t0, t1], re-reading the log first."""
        self.update()
        return energy_joules(self.samples, t0, t1)
