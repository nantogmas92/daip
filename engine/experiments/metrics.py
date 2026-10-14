"""Experiment logging and paired statistics.

Every run writes an append-only Parquet stream partitioned by experiment id and
seed, so that a study can be re-aggregated without re-running the simulator.
All reported means carry a percentile bootstrap interval, and every method
comparison is paired on the common random seed.
"""

from __future__ import annotations

import json
import os
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


class ExperimentLogger:
    def __init__(self, root: str, experiment_id: str):
        self.root = root
        self.experiment_id = experiment_id
        os.makedirs(root, exist_ok=True)

    def _write(self, rows: Sequence[Dict], name: str, seed: Optional[int] = None) -> str:
        import pandas as pd
        path = os.path.join(self.root, name)
        os.makedirs(path, exist_ok=True)
        fn = os.path.join(path, "%s%s.parquet" % (
            self.experiment_id, "" if seed is None else "_seed%03d" % seed))
        pd.DataFrame(list(rows)).to_parquet(fn, index=False)
        return fn

    def write_events(self, events, seed: int) -> str:
        return self._write([e.as_row(self.experiment_id, 0) for e in events],
                           "events", seed)

    def write_decisions(self, records, seed: int) -> str:
        return self._write([r.as_row() for r in records], "decisions", seed)

    def write_summary(self, rows: Sequence[Dict], name: str = "summary") -> str:
        import pandas as pd
        path = os.path.join(self.root, name)
        os.makedirs(path, exist_ok=True)
        fn = os.path.join(path, "%s.parquet" % self.experiment_id)
        df = pd.DataFrame(list(rows))
        df.to_parquet(fn, index=False)
        df.to_csv(fn.replace(".parquet", ".csv"), index=False)
        return fn

    def write_config(self, cfg: Dict) -> str:
        path = os.path.join(self.root, "config")
        os.makedirs(path, exist_ok=True)
        fn = os.path.join(path, "%s.json" % self.experiment_id)
        with open(fn, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2, default=str)
        return fn


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------


def bootstrap_ci(values: Sequence[float], n: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> Tuple[float, float, float]:
    """Percentile bootstrap CI, reported for every mean in the paper."""
    vals = [v for v in values if v == v]
    if not vals:
        return (float("nan"), float("nan"), float("nan"))
    m = sum(vals) / len(vals)
    if len(vals) < 3:
        return (m, m, m)
    rng = random.Random(seed)
    k = len(vals)
    boots = []
    for _ in range(n):
        boots.append(sum(vals[rng.randrange(k)] for _ in range(k)) / k)
    boots.sort()
    return (m, boots[int(alpha / 2 * n)], boots[min(n - 1, int((1 - alpha / 2) * n))])


def paired_diff(a: Sequence[float], b: Sequence[float], n: int = 2000,
                alpha: float = 0.05, seed: int = 0) -> Dict[str, float]:
    """Paired bootstrap on common random seeds, with a matched effect size."""
    pairs = [(x, y) for x, y in zip(a, b) if x == x and y == y]
    if not pairs:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "d": float("nan"), "n": 0, "p": float("nan")}
    diffs = [x - y for x, y in pairs]
    m, lo, hi = bootstrap_ci(diffs, n, alpha, seed)
    k = len(diffs)
    mu = sum(diffs) / k
    var = sum((d - mu) ** 2 for d in diffs) / max(1, k - 1)
    sd = var ** 0.5
    d = mu / sd if sd > 1e-12 else float("nan")      # Cohen's d_z
    return {"mean": m, "lo": lo, "hi": hi, "d": d, "n": k,
            "p": _wilcoxon_p(diffs)}


def _wilcoxon_p(diffs: Sequence[float]) -> float:
    """Two-sided Wilcoxon signed-rank p-value (normal approximation).

    Reported alongside effect sizes and intervals rather than instead of them:
    with common random numbers the paired differences are the quantity of
    interest, and a p-value alone would say little about magnitude.
    """
    vals = [d for d in diffs if abs(d) > 1e-12]
    n = len(vals)
    if n < 6:
        return float("nan")
    order = sorted(range(n), key=lambda i: abs(vals[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(abs(vals[order[j + 1]]) - abs(vals[order[i]])) < 1e-12:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    wp = sum(r for r, v in zip(ranks, vals) if v > 0)
    wm = sum(r for r, v in zip(ranks, vals) if v < 0)
    w = min(wp, wm)
    mu = n * (n + 1) / 4.0
    sd = (n * (n + 1) * (2 * n + 1) / 24.0) ** 0.5
    if sd <= 0:
        return float("nan")
    z = (w - mu + 0.5) / sd
    return max(0.0, min(1.0, 2.0 * _norm_sf(abs(z))))


def _norm_sf(x: float) -> float:
    import math
    return 0.5 * math.erfc(x / math.sqrt(2.0))
