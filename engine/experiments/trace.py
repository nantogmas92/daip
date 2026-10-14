"""Full per-run logging: the ten Parquet streams of one closed-loop run.

The campaign stores one summary row per run, which is all the reported
statistics need.  Because runs replay deterministically, the complete record of
any run -- every operational event, every factor message, every candidate set,
every solver round and every executed intervention -- is regenerated on demand
by re-running it with a ``TraceCollector`` attached (``experiments/
replay_run.py``).  Tracing is off during the campaign so that it cannot perturb
the latency measurements.

Streams written, one Parquet file each:

    experiment_metadata   the cell, the seed and the software environment
    flight_events         on-block, arrival delay, stand release, readiness
    task_events           service start / completion / overrun
    gse_events            dispatch, arrival, travel delay, breakdown, congestion
    crew_events           unavailability and restoration
    factor_messages       every Algorithm 1 state- and sensitivity-stream message
    candidate_events      every Algorithm 2 candidate, with leverage and class
    solver_events         every Algorithm 3 proposal and its validation verdict
    interventions         every executed intervention and whether it applied
    metrics               the run summary (identical to the campaign row)
"""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import asdict
from typing import Any, Dict, List

FLIGHT_EVENTS = ("FLIGHT_ON_BLOCK", "ARRIVAL_DELAY", "STAND_RELEASE",
                 "STAND_RELEASE_DELAY", "AIRCRAFT_READY", "CONTROL_TICK")
TASK_EVENTS = ("TASK_START", "TASK_COMPLETE", "SERVICE_OVERRUN")
GSE_EVENTS = ("GSE_DISPATCHED", "GSE_ARRIVED", "GSE_TRAVEL_DELAY",
              "GSE_BREAKDOWN", "GSE_RESTORED", "APRON_CONGESTION")
CREW_EVENTS = ("CREW_UNAVAILABLE", "CREW_RESTORED")


class TraceCollector:
    """Receives hooks from the closed-loop controller and buffers rows."""

    def __init__(self, experiment_id: str):
        self.experiment_id = experiment_id
        self.messages: List[Dict[str, Any]] = []
        self.candidates: List[Dict[str, Any]] = []
        self.solver: List[Dict[str, Any]] = []
        self.interventions: List[Dict[str, Any]] = []

    # -- hooks called by algorithms/control.py ------------------------------
    def on_messages(self, epoch: int, t: float, messages) -> None:
        for m in messages:
            self.messages.append({
                "epoch": epoch, "simulation_time": t,
                "message_type": m.message_type,
                "source_node": m.source_node,
                "destination": m.destination_factor,
                "old_value": m.old_value, "new_value": m.new_value,
                "delta": m.delta,
                "active_set_before": "|".join(sorted(m.active_set_before)),
                "active_set_after": "|".join(sorted(m.active_set_after)),
                "causal_support": "|".join(m.causal_support),
                "stream": "sensitivity" if m.is_sensitivity else "state",
            })

    def on_candidates(self, epoch: int, t: float, cands, targets, stats) -> None:
        for u in cands:
            self.candidates.append({
                "epoch": epoch, "simulation_time": t,
                "candidate_id": u.candidate_id,
                "candidate_type": u.candidate_type,
                "resources": "|".join(u.affected_resources),
                "flights": "|".join(u.affected_flights),
                "targets": "|".join(u.target_violation),
                "max_leverage": max(u.predicted_leverage.values(), default=0.0),
                "leverage_fraction": u.leverage_fraction,
                "sufficient": u.sufficient,
                "bottleneck_class": u.bottleneck_class,
                "origin": u.origin, "cost": u.cost, "harm": u.harm,
                "universe_size": stats.universe,
                "n_violations": len(targets),
            })

    def on_solver(self, epoch: int, t: float, round_: int, sel, out) -> None:
        self.solver.append({
            "epoch": epoch, "simulation_time": t, "round": round_,
            "status": sel.status, "elapsed_ms": sel.elapsed_ms,
            "chosen": "|".join(u.candidate_id for u in sel.chosen),
            "bundle_size": len(sel.chosen),
            "accepted": bool(out.accepted) if out is not None else False,
            "validate_ms": out.elapsed_ms if out is not None else 0.0,
            "global_gain": getattr(out, "global_gain", float("nan"))
            if out is not None else float("nan"),
            "new_violations": "|".join(out.bundle.new_violations)
            if (out is not None and out.bundle is not None) else "",
            "model_error": out.bundle.model_error
            if (out is not None and out.bundle is not None) else float("nan"),
        })

    def on_intervention(self, epoch: int, t: float, u, applied: bool) -> None:
        self.interventions.append({
            "epoch": epoch, "simulation_time": t,
            "candidate_id": u.candidate_id, "candidate_type": u.candidate_type,
            "resources": "|".join(u.affected_resources),
            "theta": u.theta, "applied": bool(applied),
            "predicted_leverage": max(u.predicted_leverage.values(), default=0.0),
        })

    # -- output -------------------------------------------------------------
    def write(self, out_dir: str, cell, sim, summary: Dict[str, Any]) -> Dict[str, str]:
        import pandas as pd

        os.makedirs(out_dir, exist_ok=True)
        ev_rows = [e.as_row(self.experiment_id, 0) for e in sim.events]

        def pick(types):
            return [r for r in ev_rows if r["event_type"] in types]

        known = set(FLIGHT_EVENTS + TASK_EVENTS + GSE_EVENTS + CREW_EVENTS)
        flight = pick(FLIGHT_EVENTS) + [r for r in ev_rows
                                        if r["event_type"] not in known]
        meta = dict(asdict(cell))
        meta.update({
            "experiment_id": self.experiment_id,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "pythonhashseed": os.environ.get("PYTHONHASHSEED", ""),
            "n_events": len(ev_rows),
        })
        streams = {
            "experiment_metadata": [meta],
            "flight_events": flight,
            "task_events": pick(TASK_EVENTS),
            "gse_events": pick(GSE_EVENTS),
            "crew_events": pick(CREW_EVENTS),
            "factor_messages": self.messages,
            "candidate_events": self.candidates,
            "solver_events": self.solver,
            "interventions": self.interventions,
            "metrics": [dict((k, v) for k, v in summary.items()
                             if isinstance(v, (int, float, str, bool)))],
        }
        written = {}
        for name, rows in streams.items():
            fn = os.path.join(out_dir, name + ".parquet")
            df = pd.DataFrame(rows)
            if df.empty:                    # keep a typed, empty file
                df = pd.DataFrame({"experiment_id": pd.Series(dtype="str")})
            elif "experiment_id" not in df.columns:
                df.insert(0, "experiment_id", self.experiment_id)
            df.to_parquet(fn, index=False)
            written[name] = fn
        return written
