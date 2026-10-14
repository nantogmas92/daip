"""Flights, the turnaround scenario, and the scenario generator.

Each flight is

    F_f = (r_f, d_f, A_f, T_f, R_f, C_f)

with r_f the on-block / service-release time, d_f the target aircraft-ready
time, A_f the assigned stand, T_f the turnaround task set, R_f its resource
requirements and C_f its temporal, safety and operational constraints.

The generator produces arrival and departure *banks* rather than a uniform
stream, because clustered banks are the regime in which cross-flight resource
coupling actually appears, and one of the research questions asks precisely
when representation fidelity matters.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from . import crews as crews_mod
from . import gse as gse_mod
from .apron_graph import (ApronGraph, corridor_apron, geometry_inspired_apron,
                          stylised_apron)
from .calibration import minimum_turnaround
from .safety_constraints import SafetyProfile, get_profile, stand_conflicts
from .stands import POSITION_CAPACITY, Stand, build_stands
from .turnaround_tasks import (
    AIRCRAFT_TYPES,
    AircraftType,
    TurnaroundTask,
    build_turnaround,
    critical_path_length,
    terminal_tasks,
)

# Legacy multipliers on the precedence-only critical path (target_basis="cp").
TURNAROUND_THETA = {"loose": 1.45, "moderate": 1.22, "tight": 1.08, "extreme": 1.00}
# Multipliers on the stand-feasible minimum turnaround (target_basis="mtt").
# "moderate" is the smallest grid value at which the no-repair policy reaches
# at least 80% on-time readiness under nominal operations on layouts A and B
# (calibration seeds 900-911, disjoint from every evaluation seed).
TURNAROUND_THETA_MTT = {"loose": 1.40, "moderate": 1.25, "tight": 1.10, "extreme": 1.00}
BANK_SPREAD = {"clustered": 0.28, "moderate": 0.55, "uniform": 1.0}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class AirportConfig:
    """All exogenous experimental factors of the turnaround scenario."""

    # scale
    n_flights: int = 24                  # flights released over the horizon
    n_terminal_stands: int = 10
    n_remote_stands: int = 4
    horizon: float = 240.0               # minutes of released operation
    layout: str = "stylised"             # stylised (A) | geometry (B) | corridor (C)

    # flight stream
    bank_structure: str = "moderate"     # clustered | moderate | uniform
    n_banks: int = 3
    turnaround_tightness: str = "moderate"
    theta: Optional[float] = None        # explicit target multiplier; overrides tightness
    target_basis: str = "mtt"            # mtt: stand-feasible minimum | cp: legacy precedence path
    scenario_jitter: float = 0.10        # s.d. of nominal-duration jitter (scenario definition)
    arrival_delay_share: float = 0.30    # fraction of flights arriving late
    arrival_delay_scale: float = 9.0     # minutes
    service_policy_drop: float = 0.12    # per-service probability of omission

    # resources
    gse_utilisation: str = "medium"      # low | medium | high | extreme
    crew_utilisation: str = "medium"
    crew_heterogeneity: str = "medium"
    reserve_fraction: float = 0.10
    workload_gain: float = 0.0           # optional; zero in the primary study

    # spatial
    congestion_level: str = "medium"     # low | medium | high
    edge_capacity: float = 6.0
    congestion_beta: float = 1.6

    # stand-side rules
    safety_profile: str = "default"
    wide_share: float = 0.35

    # execution variability (physical) and information quality (controller)
    eta_noise: float = 0.05
    duration_noise: float = 0.10
    info_delay: float = 0.0              # minutes before a breakdown, crew absence or overrun is reported
    info_missing: bool = False           # overruns never reported; outage end times unknown
    allow_time_compression: bool = False  # legacy, physically unvalidated surrogates
    allow_resequencing: bool = True       # its counterfactual approximates the queue re-sort
    observation_sigma: float = 0.0       # independent multiplicative forecast error
    action_lead_time: float = 0.0        # communication delay; recheck on execution
    capacity_calendar: bool = False     # frozen-calendar sensitivity model

    seed: int = 0
    regime: str = "base"

    def scaled(self) -> "AirportConfig":
        """Keep stand supply and horizon consistent with the flight count."""
        cfg = AirportConfig(**self.__dict__)
        need = max(4, int(math.ceil(self.n_flights / 2.2)))
        if cfg.n_terminal_stands + cfg.n_remote_stands < need:
            cfg.n_terminal_stands = max(cfg.n_terminal_stands,
                                        int(round(need * 0.72)))
            cfg.n_remote_stands = max(cfg.n_remote_stands, need - cfg.n_terminal_stands)
        return cfg


@dataclass
class Flight:
    """One turnaround instance."""

    flight_id: str
    actype: str
    release: float                    # r_f : on-block / service release
    target: float                     # d_f : aircraft-ready target
    stand: str
    tasks: Dict[str, TurnaroundTask] = field(default_factory=dict)
    terminal: Tuple[str, ...] = ()
    conflicts: Tuple[Tuple[str, str, str, float], ...] = ()
    weight: float = 1.0
    min_turnaround: float = 0.0       # span on which the target is based
    scheduled_release: float = 0.0    # planned on-block, before arrival delay
    prev_on_stand: Optional[str] = None   # flight that must clear the stand first
    ready: float = float("nan")       # realised C_f^ready
    stand_released: float = float("nan")
    started: bool = False

    @property
    def size(self) -> float:
        return AIRCRAFT_TYPES[self.actype].size

    def nominal_span(self) -> float:
        return critical_path_length(self.tasks)

    def as_row(self) -> Dict[str, object]:
        return {
            "flight_id": self.flight_id,
            "actype": self.actype,
            "release": self.release,
            "target": self.target,
            "stand": self.stand,
            "n_tasks": len(self.tasks),
            "weight": self.weight,
            "ready": self.ready,
        }


# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------


class TurnaroundScenario:
    """A fully materialised, deterministically replayable apron instance."""

    def __init__(self, cfg: AirportConfig):
        self.cfg = cfg.scaled()
        c = self.cfg
        self.rng = random.Random(c.seed * 7919 + 13)
        self.safety: SafetyProfile = get_profile(c.safety_profile)

        if c.layout == "geometry":
            self.apron: ApronGraph = geometry_inspired_apron(
                c.n_terminal_stands, c.n_remote_stands, c.edge_capacity,
                c.congestion_level)
        elif c.layout == "corridor":
            self.apron = corridor_apron(
                c.n_terminal_stands, c.n_remote_stands,
                congestion_level=c.congestion_level,
                edge_capacity=c.edge_capacity, beta=c.congestion_beta)
        else:
            self.apron = stylised_apron(
                c.n_terminal_stands, c.n_remote_stands,
                congestion_level=c.congestion_level,
                edge_capacity=c.edge_capacity, beta=c.congestion_beta)

        self.stands: Dict[str, Stand] = build_stands(self.apron, c.wide_share)
        self.fleet = gse_mod.build_fleet(self.rng, c.n_flights, c.gse_utilisation,
                                         self.apron, c.reserve_fraction)
        self.crews = crews_mod.build_crews(self.rng, c.n_flights, c.crew_utilisation,
                                           self.apron, c.crew_heterogeneity,
                                           c.workload_gain)
        self.flights: Dict[str, Flight] = {}
        self.tasks: Dict[str, TurnaroundTask] = {}
        self._generate_flights()

    # ------------------------------------------------------------------
    def _sample_type(self) -> AircraftType:
        r = self.rng.random()
        acc = 0.0
        for t in AIRCRAFT_TYPES.values():
            acc += t.weight
            if r <= acc:
                return t
        return AIRCRAFT_TYPES["NARROW"]

    def _bank_times(self) -> List[float]:
        """Release times arranged into arrival banks."""
        c = self.cfg
        spread = BANK_SPREAD.get(c.bank_structure, 0.55)
        n_banks = max(1, c.n_banks)
        per = c.n_flights / float(n_banks)
        out: List[float] = []
        for k in range(c.n_flights):
            b = int(k / per)
            centre = c.horizon * (b + 0.5) / n_banks
            width = (c.horizon / n_banks) * spread * 0.5
            out.append(max(0.0, self.rng.gauss(centre, width)))
        out.sort()
        return out

    def _generate_flights(self) -> None:
        c = self.cfg
        rng = self.rng
        table = TURNAROUND_THETA if c.target_basis == "cp" else TURNAROUND_THETA_MTT
        theta = c.theta if c.theta is not None else \
            table.get(c.turnaround_tightness, table["moderate"])
        times = self._bank_times()
        stand_free: Dict[str, float] = dict((s, 0.0) for s in self.stands)
        stand_last: Dict[str, Optional[str]] = dict((s, None) for s in self.stands)

        for k, t_sched in enumerate(times):
            fid = "F%d" % k
            actype = self._sample_type()

            # service policy: omit a few non-essential services per flight
            drop = tuple(code for code in actype.services
                         if code not in ("DEBOARD", "BOARD", "PBPREP")
                         and rng.random() < c.service_policy_drop)
            # Nominal-duration jitter defines the scenario.  It is a separate
            # parameter from the execution noise so that an execution-noise
            # sweep does not also change durations and targets.
            jitter = dict((code, max(0.55, rng.gauss(1.0, c.scenario_jitter)))
                          for code in actype.services)
            tasks = build_turnaround(fid, actype, drop, jitter)
            conflicts = tuple(stand_conflicts(tasks, self.safety))

            # stand assignment: earliest-free compatible stand
            feasible = [s for s, st in self.stands.items() if st.accepts(actype.code)]
            if not feasible:
                feasible = list(self.stands)
            stand = min(feasible, key=lambda s: (max(stand_free[s], 0.0), s))

            # arrival delay shifts the on-block time; the departure target is
            # only partially adjusted, which is what compresses the turnaround
            delay = 0.0
            if rng.random() < c.arrival_delay_share:
                delay = rng.expovariate(1.0 / max(1e-6, c.arrival_delay_scale))
            release = max(t_sched + delay, stand_free[stand])

            if c.target_basis == "cp":
                span = critical_path_length(tasks)       # legacy basis
            else:
                span = minimum_turnaround(tasks, conflicts, POSITION_CAPACITY)
            target = t_sched + theta * span + 0.45 * delay
            stand_free[stand] = release + span * 1.15 + 6.0

            fl = Flight(flight_id=fid, actype=actype.code, release=release,
                        target=target, stand=stand, tasks=tasks,
                        terminal=terminal_tasks(tasks),
                        weight=actype.delay_weight,
                        scheduled_release=t_sched)
            for _t in tasks.values():
                _t.rank = release + _t.est
            fl.conflicts = conflicts
            fl.min_turnaround = span
            fl.prev_on_stand = stand_last[stand]
            stand_last[stand] = fid
            self.flights[fid] = fl
            self.tasks.update(tasks)

    # ------------------------------------------------------------------
    # capability / qualification predicates
    # ------------------------------------------------------------------
    def gse_capable(self, unit, task: TurnaroundTask) -> bool:
        if task.gse_class is None:
            return False
        return unit.resource_type == task.gse_class and unit.can_serve(task.code)

    def crew_qualified(self, crew, task: TurnaroundTask) -> bool:
        if task.crew_skill is None:
            return False
        return crew.qualified(task.code)

    def recompute_ranks(self, flight_id: str) -> None:
        """Re-derive the global scheduling key after a dynamic precedence edge.

        A resequencing intervention installs a hold a -> b.  If the ranks were
        left untouched, ``b`` could still be ordered ahead of ``a``, and the
        rank-oriented service-position and stand-side relations would then point
        the opposite way from the new precedence edge -- closing a cycle with no
        fixed point.  Recomputing the key over (predecessors + holds) keeps the
        orientation of every shared relation consistent with the precedence that
        actually holds.  The relation is acyclic by construction, because the
        simulator refuses a hold that would close a precedence cycle.
        """
        fl = self.flights.get(flight_id)
        if fl is None:
            return
        memo: Dict[str, float] = {}

        def est(tid: str, guard=0) -> float:
            if tid in memo:
                return memo[tid]
            if guard > 64:
                return 0.0
            t = fl.tasks[tid]
            memo[tid] = max([est(p, guard + 1) + fl.tasks[p].duration
                             for p in tuple(t.predecessors) + tuple(t.holds)
                             if p in fl.tasks] or [0.0])
            return memo[tid]

        for tid, t in fl.tasks.items():
            t.est = est(tid)
            t.rank = fl.release + t.est

    def rank(self, tid: str) -> Tuple[float, str]:
        """Global scheduling key of a task; orients every shared relation."""
        t = self.tasks.get(tid)
        return (t.rank, tid) if t is not None else (0.0, tid)

    def ahead(self, a: str, b: str) -> bool:
        """True when ``a`` is scheduled ahead of ``b`` in the global order."""
        return self.rank(a) < self.rank(b)

    def stand_node(self, flight_id: str) -> str:
        f = self.flights.get(flight_id)
        return self.stands[f.stand].node if f else ""

    def service_duration(self, task: TurnaroundTask, crew) -> float:
        rate = crew.rate if crew is not None else 1.0
        w = crew.workload_state if crew is not None else 0.0
        gain = crew.workload_gain if crew is not None else 0.0
        return task.duration * rate * (1.0 + gain * w) + task.overrun

    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, object]:
        return {
            "n_flights": len(self.flights),
            "n_tasks": len(self.tasks),
            "n_stands": len(self.stands),
            "n_gse": len(self.fleet),
            "n_crews": len(self.crews),
            "apron": self.apron.summary(),
            "mean_tasks_per_flight": len(self.tasks) / max(1, len(self.flights)),
            "safety_profile": self.safety.name,
        }
