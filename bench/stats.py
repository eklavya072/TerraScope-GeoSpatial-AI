"""Confidence intervals and pairwise significance testing.

EuroSAT is near-saturated: every model in this zoo is expected to land in the
97-99% band, where between-architecture gaps are comparable to seed-to-seed
noise. Reporting a single seed's accuracy would therefore produce a ranking
that is an artefact of seed choice. Everything here exists to keep that from
happening.
"""

import math

import numpy as np
from scipy import stats


def mean_ci(values, confidence: float = 0.95) -> dict:
    """Mean with a Student-t confidence interval over seeds.

    The t-distribution (not the normal) is correct here because the seed count
    is small -- with n=5 the difference is substantial and using z would
    understate the interval by roughly 25%.
    """
    a = np.asarray(values, dtype=float)
    n = len(a)
    mean = float(a.mean())
    if n < 2:
        return {"mean": mean, "ci_low": None, "ci_high": None, "half_width": None,
                "std": None, "n": n}
    sem = float(stats.sem(a))
    half = float(sem * stats.t.ppf(0.5 + confidence / 2.0, n - 1))
    return {"mean": mean, "ci_low": mean - half, "ci_high": mean + half,
            "half_width": half, "std": float(a.std(ddof=1)), "n": n}


def compare(a, b, alpha: float = 0.05) -> dict:
    """Welch's t-test between two seed populations.

    Welch rather than Student: we cannot assume two architectures have equal
    seed-to-seed variance, and Welch costs nothing when they do.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    t, p = stats.ttest_ind(a, b, equal_var=False)
    diff = float(a.mean() - b.mean())
    # Cohen's d with pooled sd, for effect size alongside the p-value.
    n1, n2 = len(a), len(b)
    pooled = math.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1))
                       / max(1, n1 + n2 - 2)) if n1 + n2 > 2 else 0.0
    return {"diff": diff, "t": float(t), "p": float(p),
            "cohens_d": float(diff / pooled) if pooled > 0 else None,
            "significant_uncorrected": bool(p < alpha)}


def holm_bonferroni(pairs: dict[tuple[str, str], dict], alpha: float = 0.05) -> dict:
    """Holm-Bonferroni correction across all pairwise comparisons.

    With k models there are k(k-1)/2 comparisons; at k=5 that is 10 tests, and
    an uncorrected 5% threshold would be expected to manufacture roughly one
    spurious "significant" difference per run. Holm controls the family-wise
    error rate while being uniformly more powerful than plain Bonferroni.
    """
    items = sorted(pairs.items(), key=lambda kv: kv[1]["p"])
    m = len(items)
    out, rejected_so_far = {}, True
    for i, (key, res) in enumerate(items):
        threshold = alpha / (m - i)
        reject = bool(rejected_so_far and res["p"] < threshold)
        rejected_so_far = reject
        out[key] = {**res, "holm_threshold": threshold, "significant_holm": reject}
    return out


def pareto_frontier(points: list[dict], x_key: str, y_key: str) -> list[str]:
    """Labels on the frontier that MINIMISES x_key and MAXIMISES y_key.

    Used for energy (minimise) against accuracy (maximise): a model is on the
    frontier when no other model is both cheaper and at least as accurate.
    """
    usable = [p for p in points if p.get(x_key) is not None and p.get(y_key) is not None]
    frontier = []
    for p in usable:
        dominated = any(
            q is not p and q[x_key] <= p[x_key] and q[y_key] >= p[y_key]
            and (q[x_key] < p[x_key] or q[y_key] > p[y_key])
            for q in usable
        )
        if not dominated:
            frontier.append(p["label"])
    return frontier
