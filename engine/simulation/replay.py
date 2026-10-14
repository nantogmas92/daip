"""Deterministic replay and run fingerprinting.

A run is reproducible only if the same seed produces the same event stream,
the same decisions and the same outcome in a fresh process.  Control flow
iterates over sets of entity identifiers, so ``PYTHONHASHSEED`` must be pinned;
this module does not set it, because by the time Python code runs it is too
late -- the variable must be exported before the interpreter starts.

``fingerprint`` reduces a finished simulator to a stable digest of its event
stream and operational outcome.  ``replay`` executes the same closed-loop
configuration twice and reports whether the two digests agree, which is the
check behind the reproducibility statement of the manuscript.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple


def _round(x: float, nd: int = 6) -> str:
    """Render floats canonically so the digest is platform-independent."""
    if x != x:
        return "nan"
    return ("%.*f" % (nd, x)).rstrip("0").rstrip(".")


def event_stream_digest(events) -> str:
    h = hashlib.sha256()
    for ev in events:
        parts = [
            str(ev.event_id), _round(ev.simulation_time), ev.event_type,
            ev.flight_id, ev.task_id, "|".join(ev.resource_ids),
            ev.location_id,
            "|".join("%s=%s" % (k, _round(v))
                     for k, v in sorted(ev.observed_delta.items())),
        ]
        h.update(("\x1f".join(parts) + "\n").encode("utf-8"))
    return h.hexdigest()


def fingerprint(sim, summary: Optional[Dict[str, float]] = None) -> Dict[str, str]:
    """Digest of a finished run: its event stream plus its outcome."""
    out = {"events": event_stream_digest(sim.events),
           "n_events": str(len(sim.events))}
    if summary is not None:
        keys = sorted(k for k, v in summary.items()
                      if isinstance(v, (int, float)) and not k.startswith("mean_t"))
        # wall-clock timings are excluded: they are measured, not replayed
        h = hashlib.sha256()
        for k in keys:
            h.update(("%s=%s\n" % (k, _round(float(summary[k])))).encode("utf-8"))
        out["outcome"] = h.hexdigest()
    return out


@dataclass
class ReplayReport:
    identical: bool
    first: Dict[str, str]
    second: Dict[str, str]
    hash_seed: Optional[str]

    def __str__(self) -> str:
        state = "IDENTICAL" if self.identical else "DIVERGED"
        return ("replay %s  events=%s  outcome=%s  PYTHONHASHSEED=%s"
                % (state, self.first.get("events", "")[:12],
                   self.first.get("outcome", "")[:12], self.hash_seed))


def replay(run: Callable[[], Tuple[object, Dict[str, float]]]) -> ReplayReport:
    """Execute ``run`` twice; ``run`` returns (finished simulator, summary)."""
    sim1, s1 = run()
    sim2, s2 = run()
    f1, f2 = fingerprint(sim1, s1), fingerprint(sim2, s2)
    return ReplayReport(f1 == f2, f1, f2, os.environ.get("PYTHONHASHSEED"))


def closed_loop_runner(cfg_kwargs: Dict, seed: int, family_set: str = "mixed",
                       controller_kwargs: Optional[Dict] = None):
    """Return a zero-argument callable that builds and runs one episode."""
    from airport.flights import AirportConfig
    from algorithms.control import ClosedLoopController, ControllerConfig
    from simulation.disturbances import build_plan
    from simulation.event_engine import TurnaroundSimulator

    def _run():
        cfg = AirportConfig(seed=seed, **cfg_kwargs)
        plan = build_plan(seed, cfg.horizon, family_set)
        sim = TurnaroundSimulator(cfg, plan)
        ctrl = ClosedLoopController(sim, ControllerConfig(**(controller_kwargs or {})),
                                    "replay")
        return sim, ctrl.run()

    return _run


if __name__ == "__main__":         # pragma: no cover - command-line check
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    rep = replay(closed_loop_runner({"n_flights": 10, "horizon": 120.0}, seed=3))
    print(rep)
    raise SystemExit(0 if rep.identical else 1)
