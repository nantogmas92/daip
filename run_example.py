"""Quick start: one paired scenario of the main comparison.

    python run_example.py [--layout geometry|corridor] [--seed 1000]

Runs no repair, the dispatcher rule and DAIP (primary configuration: beam
selection, 130 planning evaluations per decision, bundle size 4) on the same
20-flight bank and prints the priority-weighted readiness delay, on-time
readiness and number of executed actions.  Layout geometry is the manuscript's
Layout A and corridor its Layout B; seeds 1000-1059 are those of the paper.
Each DAIP run takes roughly 5-20 s.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "engine"))
sys.path.insert(0, str(ROOT / "studies_main" / "baselines"))
os.environ.setdefault("PYTHONHASHSEED", "0")

import native  # noqa: E402  (registers the dispatcher rule and the other comparators)
from experiments import revision_harness as H  # noqa: E402

native.install()

METHODS = (("none", "No repair"), ("dispatch", "Dispatcher rule"), ("proposed", "DAIP"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--layout", default="geometry", choices=("geometry", "corridor"))
    ap.add_argument("--seed", type=int, default=1000)
    a = ap.parse_args()
    print("layout %s, seed %d, engine digest %s" % (a.layout, a.seed, H.source_digest()[:12]))
    print("%-16s %14s %10s %8s" % ("method", "J (min/flight)", "on-time", "actions"))
    for method, label in METHODS:
        spec = H.Spec(study="example", arm=method, seed=a.seed, layout=a.layout, method=method)
        summary, _ = H.run(spec)
        print("%-16s %14.3f %10.3f %8d" % (label, summary["delay_w_per_flight"],
                                           summary["on_time_rate"], summary["interventions"]))


if __name__ == "__main__":
    main()
