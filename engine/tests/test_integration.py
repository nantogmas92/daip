"""Closed-loop integration tests (6: candidate pruning, 10: deterministic replay).

These run the real simulator and controller on a small apron, so they take a
few seconds each rather than milliseconds.
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from airport.flights import AirportConfig                       # noqa: E402
from algorithms.algorithm2_candidates import EPS_LAMBDA         # noqa: E402
from algorithms.control import ClosedLoopController, ControllerConfig  # noqa: E402
from simulation.disturbances import build_plan                  # noqa: E402
from simulation.event_engine import TurnaroundSimulator         # noqa: E402
from simulation.replay import closed_loop_runner, replay        # noqa: E402

SMALL = {"n_flights": 10, "horizon": 110.0}


def _controller(seed: int, **ckw):
    cfg = AirportConfig(seed=seed, **SMALL)
    sim = TurnaroundSimulator(cfg, build_plan(seed, cfg.horizon, "mixed"))
    return sim, ClosedLoopController(sim, ControllerConfig(**ckw), "test")


# ---------------------------------------------------------------------------
# 6. candidate pruning: Proposition F (soundness) on live decision epochs
# ---------------------------------------------------------------------------
def test_every_candidate_has_positive_leverage():
    seen = []
    for seed in (0, 1, 2):
        sim, ctrl = _controller(seed, representation="R6", candidates="C2",
                                measure_universe=True)
        original = ctrl.generator.generate

        def recording(violations, want_universe=False, _orig=original):
            cands, stats = _orig(violations, want_universe=want_universe)
            seen.append((cands, stats))
            return cands, stats

        ctrl.generator.generate = recording
        ctrl.run()
    epochs = [c for c in seen if c[0]]
    assert epochs, "no decision epoch produced candidates; test is vacuous"
    for cands, stats in epochs:
        for u in cands:
            # soundness: positive predicted leverage on at least one target
            assert max(u.predicted_leverage.values()) > EPS_LAMBDA
        # the screened set is never larger than the enumerated universe
        if stats.universe:
            assert len(cands) <= stats.universe


def test_screening_is_a_strict_reduction():
    sizes = []
    for seed in (0, 1):
        for mode in ("C0", "C2"):
            sim, ctrl = _controller(seed, representation="R6", candidates=mode,
                                    measure_universe=True)
            s = ctrl.run()
            sizes.append((mode, s["mean_candidates"]))
    c0 = [v for m, v in sizes if m == "C0" and v == v]
    c2 = [v for m, v in sizes if m == "C2" and v == v]
    assert c0 and c2 and max(c2) < min(c0)


# ---------------------------------------------------------------------------
# 10. deterministic replay
# ---------------------------------------------------------------------------
def test_deterministic_replay():
    rep = replay(closed_loop_runner(SMALL, seed=4))
    assert rep.identical, str(rep)
    assert int(rep.first["n_events"]) > 0


def test_trace_writes_ten_streams():
    import tempfile
    import pandas as pd
    from experiments.runner import Cell
    from experiments.trace import TraceCollector
    sim, ctrl = _controller(1, representation="R6")
    ctrl.trace = TraceCollector("t")
    summary = ctrl.run()
    files = ctrl.trace.write(tempfile.mkdtemp(), Cell("test", seed=1), sim, summary)
    assert len(files) == 10
    msgs = pd.read_parquet(files["factor_messages"])
    assert len(msgs) > 0 and set(msgs["stream"]) <= {"state", "sensitivity"}
    iv = pd.read_parquet(files["interventions"])
    applied = int(iv["applied"].sum()) if len(iv) else 0
    assert applied == summary["interventions"]


def test_different_seeds_differ():
    a = replay(closed_loop_runner(SMALL, seed=4))
    b = replay(closed_loop_runner(SMALL, seed=5))
    assert a.first["events"] != b.first["events"]


if __name__ == "__main__":         # pragma: no cover - runner without pytest
    import inspect
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and inspect.isfunction(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print("PASS", name)
        except Exception as exc:          # noqa: BLE001
            failed += 1
            print("FAIL", name, "--", type(exc).__name__, exc)
    raise SystemExit(1 if failed else 0)
