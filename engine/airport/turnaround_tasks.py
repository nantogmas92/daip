"""Turnaround service catalogue and aircraft-type dependency templates.

A turnaround is represented as a precedence and resource-dependency DAG, not
as a fixed linear chain, and not every turnaround contains the same services.
Which services a flight requires depends on aircraft type, flight
characteristics, the operational scenario and the operator's service policy;
all four are inputs to ``build_turnaround`` below.

Durations are nominal service times in minutes for a stylised reference
operation.  They are experimental parameters of the synthetic scenario
generator, not measurements of any particular airport, and every value used in
the reported experiments is listed in the configuration tables of the
manuscript so that the scenarios can be regenerated exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Service positions around the aircraft.  A stand is a finite spatial resource
# containing these positions; a service vehicle occupies one while working.
# ---------------------------------------------------------------------------
POSITIONS = (
    "NOSE",        # pushback tractor, ground power unit
    "FWD_LEFT",    # forward left door: boarding bridge, catering, cleaning
    "AFT_LEFT",    # aft left door: catering
    "FWD_RIGHT",   # forward right door: potable water
    "AFT_RIGHT",   # aft right door: lavatory service
    "FWD_HOLD",    # forward hold: baggage / cargo
    "AFT_HOLD",    # aft hold: baggage / cargo
    "WING",        # fuelling point
)

# ---------------------------------------------------------------------------
# Service catalogue.  Each entry declares the GSE class and crew qualification
# it consumes, the stand-side position it occupies, and its nominal duration
# scaling with aircraft size.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ServiceSpec:
    code: str
    label: str
    gse_class: Optional[str]         # None = no mobile equipment required
    crew_skill: Optional[str]
    position: str
    base_minutes: float
    size_minutes: float              # additional minutes per size unit
    mobile: bool = True              # requires a vehicle movement to the stand


CATALOGUE: Dict[str, ServiceSpec] = {
    s.code: s for s in (
        ServiceSpec("DEBOARD",  "passenger deboarding",   None,           "gate",     "FWD_LEFT", 4.0, 5.0, False),
        ServiceSpec("BAGUNLD",  "baggage unloading",      "bag_tractor",  "baggage",  "AFT_HOLD", 5.0, 7.0),
        ServiceSpec("CARGOUNLD", "cargo unloading",       "cargo_loader", "baggage",  "FWD_HOLD", 5.0, 9.0),
        ServiceSpec("CLEAN",    "cabin cleaning",         "clean_van",    "cleaning", "FWD_LEFT", 6.0, 8.0),
        ServiceSpec("CATER_F",  "forward catering",       "catering",     "catering", "FWD_LEFT", 6.0, 6.0),
        ServiceSpec("CATER_A",  "aft catering",           "catering",     "catering", "AFT_LEFT", 6.0, 6.0),
        ServiceSpec("WATER",    "potable-water service",  "water",        "servicing", "FWD_RIGHT", 4.0, 3.0),
        ServiceSpec("LAV",      "lavatory service",       "lavatory",     "servicing", "AFT_RIGHT", 4.0, 3.0),
        ServiceSpec("FUEL",     "refuelling",             "fuel_truck",   "fuelling", "WING",     8.0, 11.0),
        ServiceSpec("GPU",      "ground power / air",     "gpu",          None,       "NOSE",     3.0, 2.0),
        ServiceSpec("BAGLD",    "baggage loading",        "bag_tractor",  "baggage",  "AFT_HOLD", 6.0, 8.0),
        ServiceSpec("CARGOLD",  "cargo loading",          "cargo_loader", "baggage",  "FWD_HOLD", 6.0, 10.0),
        ServiceSpec("BOARD",    "passenger boarding",     None,           "gate",     "FWD_LEFT", 7.0, 8.0, False),
        ServiceSpec("INSPECT",  "aircraft inspection",    "maint_van",    "maint",    "NOSE",     5.0, 3.0),
        ServiceSpec("PBPREP",   "pushback preparation",   "pushback",     "pushback", "NOSE",     3.0, 2.0),
    )
}

# ---------------------------------------------------------------------------
# Aircraft types.  ``size`` scales service durations; ``services`` selects the
# subset of the catalogue that the type and its operation require.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AircraftType:
    code: str
    size: float                       # duration scaling unit
    services: Tuple[str, ...]
    weight: float                     # sampling weight in the flight generator
    delay_weight: float               # w_f in the objective


AIRCRAFT_TYPES: Dict[str, AircraftType] = {
    "REGIONAL": AircraftType(
        "REGIONAL", 0.55,
        ("DEBOARD", "BAGUNLD", "CLEAN", "CATER_F", "LAV", "FUEL", "GPU",
         "BAGLD", "BOARD", "PBPREP"),
        weight=0.28, delay_weight=1.0),
    "NARROW": AircraftType(
        "NARROW", 1.0,
        ("DEBOARD", "BAGUNLD", "CLEAN", "CATER_F", "CATER_A", "WATER", "LAV",
         "FUEL", "GPU", "BAGLD", "BOARD", "INSPECT", "PBPREP"),
        weight=0.52, delay_weight=1.6),
    "WIDE": AircraftType(
        "WIDE", 1.8,
        ("DEBOARD", "BAGUNLD", "CARGOUNLD", "CLEAN", "CATER_F", "CATER_A",
         "WATER", "LAV", "FUEL", "GPU", "BAGLD", "CARGOLD", "BOARD",
         "INSPECT", "PBPREP"),
        weight=0.20, delay_weight=2.6),
}

# ---------------------------------------------------------------------------
# Precedence template.  ``PRECEDENCE[x]`` lists the services that must complete
# before ``x`` may start, when both are present in the flight's service set.
# Entries absent from a flight are contracted away, so the DAG is well defined
# for any service subset.
# ---------------------------------------------------------------------------
PRECEDENCE: Dict[str, Tuple[str, ...]] = {
    "DEBOARD":   (),
    "GPU":       (),
    "BAGUNLD":   (),
    "CARGOUNLD": (),
    "WATER":     (),
    "LAV":       ("DEBOARD",),
    "CLEAN":     ("DEBOARD",),
    "CATER_F":   ("DEBOARD",),
    "CATER_A":   ("DEBOARD",),
    "FUEL":      (),
    "BAGLD":     ("BAGUNLD",),
    "CARGOLD":   ("CARGOUNLD",),
    "INSPECT":   (),
    "BOARD":     ("CLEAN", "CATER_F", "CATER_A", "WATER", "LAV"),
    "PBPREP":    ("BOARD", "BAGLD", "CARGOLD", "FUEL", "INSPECT", "GPU"),
}

# Services that, when present, are the departure-enabling terminal set.  The
# aircraft-ready operator is the max over the completions of these tasks.
TERMINAL_PREFERENCE = ("PBPREP", "BOARD", "BAGLD", "FUEL")


@dataclass
class TurnaroundTask:
    """One service activity of one flight."""

    tid: str                          # "F12/CLEAN"
    flight: str
    code: str
    duration: float                   # nominal p_j
    position: str
    gse_class: Optional[str]
    crew_skill: Optional[str]
    mobile: bool
    predecessors: Tuple[str, ...] = ()
    successors: Tuple[str, ...] = ()
    holds: Tuple[str, ...] = ()      # dynamic precedence added by RESEQUENCE_TASKS
    est: float = 0.0                  # nominal earliest start, precedence only
    rank: float = 0.0                 # global scheduling key (see ``order_key``)
    # execution state
    state: str = "waiting"            # waiting | ready | running | done
    gse: Optional[str] = None
    crew: Optional[str] = None
    start: float = float("nan")
    finish: float = float("nan")
    overrun: float = 0.0
    expedite: float = 0.0

    @property
    def spec(self) -> ServiceSpec:
        return CATALOGUE[self.code]


def build_turnaround(flight_id: str, actype: AircraftType,
                     policy_drop: Sequence[str] = (),
                     duration_jitter: Optional[Dict[str, float]] = None
                     ) -> Dict[str, TurnaroundTask]:
    """Instantiate the service DAG of one flight.

    ``policy_drop`` removes services that the operational scenario or service
    policy does not require for this flight (for example a turnaround with no
    catering uplift).  Precedence edges through a dropped service are
    contracted so the remaining graph stays a connected DAG.
    """
    present = [c for c in actype.services if c not in policy_drop]
    present_set = set(present)
    jitter = duration_jitter or {}

    def contract(code: str, seen=None) -> List[str]:
        """Resolve predecessors through services dropped from this flight."""
        seen = seen or set()
        out: List[str] = []
        for p in PRECEDENCE.get(code, ()):
            if p in seen:
                continue
            seen.add(p)
            if p in present_set:
                out.append(p)
            else:
                out.extend(contract(p, seen))
        return out

    tasks: Dict[str, TurnaroundTask] = {}
    for code in present:
        spec = CATALOGUE[code]
        dur = (spec.base_minutes + spec.size_minutes * actype.size) * jitter.get(code, 1.0)
        preds = tuple(sorted(set("%s/%s" % (flight_id, p) for p in contract(code))))
        tid = "%s/%s" % (flight_id, code)
        tasks[tid] = TurnaroundTask(
            tid=tid, flight=flight_id, code=code, duration=round(dur, 3),
            position=spec.position, gse_class=spec.gse_class,
            crew_skill=spec.crew_skill, mobile=spec.mobile, predecessors=preds)

    for tid, t in tasks.items():
        t.successors = tuple(sorted(o for o, x in tasks.items() if tid in x.predecessors))
    _assign_est(tasks)
    return tasks


def _assign_est(tasks: Dict[str, TurnaroundTask]) -> None:
    """Nominal earliest start of each task under precedence alone.

    ``est`` is the basis of the global scheduling key used to orient every
    shared-resource relation in the model.  Because each service has strictly
    positive duration, a predecessor always has a strictly smaller ``est`` than
    its successor, so ordering resource queues, service-position calendars and
    stand-side separations by this key is *consistent with precedence*.  That
    consistency is what keeps the combined precedence-plus-resource constraint
    graph acyclic: a symmetric position relation between two tasks of the same
    flight would otherwise close a two-cycle with no fixed point -- a predicted
    deadlock -- and the local fixed point of Proposition C would not exist.
    """
    memo: Dict[str, float] = {}

    def est(tid: str) -> float:
        if tid in memo:
            return memo[tid]
        t = tasks[tid]
        memo[tid] = max([est(p) + tasks[p].duration
                         for p in t.predecessors if p in tasks] or [0.0])
        return memo[tid]

    for tid in tasks:
        tasks[tid].est = est(tid)


def order_key(task: TurnaroundTask, release: float = 0.0) -> Tuple[float, str]:
    """Global scheduling key: earliest nominal start, then a stable tie-break."""
    return (release + task.est, task.tid)


def terminal_tasks(tasks: Dict[str, TurnaroundTask]) -> Tuple[str, ...]:
    """T_f^terminal: tasks with no successor inside the turnaround DAG."""
    out = tuple(sorted(tid for tid, t in tasks.items() if not t.successors))
    return out if out else tuple(sorted(tasks))


def critical_path_length(tasks: Dict[str, TurnaroundTask]) -> float:
    """Longest path through the DAG; drives the nominal turnaround target."""
    memo: Dict[str, float] = {}

    def depth(tid: str) -> float:
        if tid in memo:
            return memo[tid]
        t = tasks[tid]
        base = max([depth(p) for p in t.predecessors if p in tasks] or [0.0])
        memo[tid] = base + t.duration
        return memo[tid]

    return max([depth(t) for t in tasks] or [0.0])
