"""Turnaround service execution: eligibility, stand-side conflicts, binding cause.

The functions here decide whether a service task may physically start and, if
so, *why it could not start earlier*.  That second question matters as much as
the first: the recorded binding cause is the ground truth against which the
controller's predicted binding cause is scored (RQ1 / H2), and it is what
distinguishes "this task is late" from "this task is delaying the aircraft".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from airport.safety_constraints import CONDITIONAL, INCOMPATIBLE

# binding-cause classes, in the order used for tie-breaking
CAUSE_RELEASE = "REL"      # aircraft not yet on-block
CAUSE_PRED = "PRED"        # an upstream turnaround task
CAUSE_GSE = "GSE"          # the assigned ground support unit
CAUSE_CREW = "CREW"        # the assigned qualified crew
CAUSE_POS = "POS"          # the stand-side service position
CAUSE_SAFETY = "SAFE"      # a stand-side compatibility restriction

CAUSE_CLASSES = (CAUSE_RELEASE, CAUSE_PRED, CAUSE_GSE, CAUSE_CREW,
                 CAUSE_POS, CAUSE_SAFETY)


@dataclass
class Eligibility:
    """Whether a task may start now, and the earliest time each term allows."""

    ready: bool
    earliest: float
    cause: str
    terms: Dict[str, float]

    def blocking(self) -> str:
        return self.cause


def predecessor_time(sim, task) -> float:
    t = -1e18
    for p in tuple(task.predecessors) + tuple(task.holds):
        pt = sim.scn.tasks.get(p)
        if pt is None:
            continue
        if pt.state != "done":
            return 1e18
        t = max(t, pt.finish)
    return t if t > -1e17 else -1e18


def safety_time(sim, task) -> Tuple[float, bool]:
    """Earliest start permitted by the stand-side compatibility matrix.

    Returns (earliest, hard_block).  ``hard_block`` is True while an
    incompatible peer is still running, because its completion time is not yet
    known and the task simply cannot be released.
    """
    fl = sim.scn.flights[task.flight]
    best = -1e18
    blocked = False
    for (a, b, rel, sep) in fl.conflicts:
        other = b if a == task.tid else (a if b == task.tid else None)
        if other is None:
            continue
        ot = sim.scn.tasks.get(other)
        if ot is None:
            continue
        if ot.state == "done":
            if rel == INCOMPATIBLE:
                best = max(best, ot.finish)
            elif rel == CONDITIONAL:
                best = max(best, ot.start + sep)     # start-to-start separation
        elif ot.state == "running":
            if rel == INCOMPATIBLE:
                blocked = True
            elif rel == CONDITIONAL:
                best = max(best, ot.start + sep)
    return best, blocked


def position_free(sim, task) -> bool:
    fl = sim.scn.flights[task.flight]
    st = sim.scn.stands[fl.stand]
    pos = st.positions.get(task.position)
    if pos is None:
        return True
    return len(pos.occupants) < pos.capacity or task.tid in pos.occupants


STALL_VALVE = 25.0        # minutes a queue entry may block before it is skipped
CREW_TRANSFER = 2.0       # minutes for a released crew to reach the stand


def resource_head(sim, sequence: Sequence[str], tid: str, tasks) -> bool:
    """True when ``tid`` may take its turn in a resource queue.

    Strict FIFO would make a queue entry whose *own* precedence is unsatisfied
    block everything behind it, which is neither how a ramp supervisor behaves
    nor necessary: such an entry cannot start soon, so it is skipped.  A stall
    valve additionally releases the constraint after ``STALL_VALVE`` minutes,
    which guarantees liveness even after an intervention has deliberately
    reordered a queue away from the global key order.
    """
    for other in sequence:
        if other == tid:
            return True
        ot = tasks.get(other)
        if ot is None or ot.state == "done":
            continue
        if ot.state == "running":
            return False
        # an entry whose own precedence is unmet cannot start now
        if any(tasks[p].state != "done"
               for p in tuple(ot.predecessors) + tuple(ot.holds) if p in tasks):
            continue
        if sim.t - sim.queue_block_since(tid) > STALL_VALVE:
            continue
        return False
    return True


def evaluate(sim, task) -> Eligibility:
    """Compute the max-factor terms that gate this task's start."""
    scn = sim.scn
    fl = scn.flights[task.flight]
    now = sim.t
    terms: Dict[str, float] = {}

    terms[CAUSE_RELEASE] = fl.release
    pt = predecessor_time(sim, task)
    if pt > 1e17:
        return Eligibility(False, 1e18, CAUSE_PRED, terms)
    terms[CAUSE_PRED] = pt

    unit = scn.fleet.get(task.gse) if task.gse else None
    if task.gse_class is not None:
        if unit is None or unit.status in ("moving", "serving", "down", "reserve"):
            return Eligibility(False, 1e18, CAUSE_GSE, terms)
        if unit.current_location != scn.stands[fl.stand].node:
            return Eligibility(False, 1e18, CAUSE_GSE, terms)
        arrived = sim.gse_at_stand.get(task.tid)
        if arrived is None or not resource_head(sim, unit.sequence, task.tid, scn.tasks):
            return Eligibility(False, 1e18, CAUSE_GSE, terms)
        terms[CAUSE_GSE] = max(arrived, unit.availability_time)

    crew = scn.crews.get(task.crew) if task.crew else None
    if task.crew_skill is not None:
        if crew is None or crew.status in ("unavailable", "serving"):
            return Eligibility(False, 1e18, CAUSE_CREW, terms)
        if not resource_head(sim, crew.sequence, task.tid, scn.tasks):
            return Eligibility(False, 1e18, CAUSE_CREW, terms)
        # a released crew still has to reach the stand; the predictive model
        # charges the same transfer, and the two must agree or the controller
        # will systematically over-attribute delay to crews
        terms[CAUSE_CREW] = crew.availability_time + CREW_TRANSFER

    if not position_free(sim, task):
        return Eligibility(False, 1e18, CAUSE_POS, terms)
    terms[CAUSE_POS] = sim.position_free_at.get(
        "%s:%s" % (fl.stand, task.position), -1e18)

    sft, hard = safety_time(sim, task)
    if hard:
        return Eligibility(False, 1e18, CAUSE_SAFETY, terms)
    terms[CAUSE_SAFETY] = sft

    live = dict((k, v) for k, v in terms.items() if v > -1e17)
    if not live:
        return Eligibility(True, now, CAUSE_RELEASE, terms)
    cause = max(live, key=lambda k: (live[k], k))
    earliest = live[cause]
    return Eligibility(earliest <= now + 1e-9, earliest, cause, terms)


def binding_cause(sim, task) -> str:
    """Ground-truth binding class of a task start, recorded at start time."""
    return evaluate(sim, task).cause


def root_cause(sim, tid: str) -> str:
    """Follow the realised critical chain to the physical resource at its root.

    The task that determines aircraft readiness is almost always gated by its
    own predecessors, so reporting ``PRED`` would answer a question nobody
    asked.  What matters operationally is *which physical resource* is at the
    root of the chain, so a ``PRED`` cause is resolved by stepping to the
    predecessor that actually finished last and repeating.  This is the ground
    truth against which the controller's predicted binding class is scored.
    """
    seen = set()
    while tid and tid not in seen:
        seen.add(tid)
        cause = sim.task_cause.get(tid)
        if cause is None:
            return CAUSE_RELEASE
        if cause != CAUSE_PRED:
            return cause
        t = sim.scn.tasks.get(tid)
        if t is None:
            return CAUSE_RELEASE
        preds = [p for p in tuple(t.predecessors) + tuple(t.holds)
                 if p in sim.scn.tasks
                 and sim.scn.tasks[p].state == "done"
                 and sim.scn.tasks[p].finish == sim.scn.tasks[p].finish]
        if not preds:
            return CAUSE_RELEASE
        tid = max(preds, key=lambda p: sim.scn.tasks[p].finish)
    return CAUSE_PRED


def critical_terminal(sim, flight):
    """The terminal task whose completion realised C_f^ready."""
    best, chosen = -1e18, None
    for tid in flight.terminal:
        t = sim.scn.tasks.get(tid)
        if t is None or t.state != "done" or t.finish != t.finish:
            continue
        if t.finish > best:
            best, chosen = t.finish, tid
    return chosen


def flight_binding_cause(sim, flight) -> str:
    """Root physical binding class of the flight's realised readiness."""
    tid = critical_terminal(sim, flight)
    return root_cause(sim, tid) if tid else CAUSE_RELEASE
