"""Controlled disturbance families D1-D10.

Random events alone make a weak experiment: whether a disturbance propagates
depends on whether it crosses a currently binding dependency, so the
disturbance stream must be able to target the binding and the non-binding case
*deliberately*.  Each family below therefore selects its target from the
simulator's own realised state rather than uniformly at random.

    D1  non-binding GSE route delay      expected: contained, no ready change
    D2  binding-service route delay      expected: propagates to readiness
    D3  criticality switch               expected: CRITICALITY_CHANGED
    D4  GSE breakdown                    resource removed mid-plan
    D5  crew unavailability
    D6  service-duration overrun
    D7  apron congestion on a segment
    D8  stand-release overrun
    D9  arrival delay / compressed turnaround
    D10 combined congestion + resource loss
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

FAMILIES = ("D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10")

FAMILY_LABEL = {
    "D1": "non-binding GSE route delay",
    "D2": "binding-service route delay",
    "D3": "criticality switch",
    "D4": "GSE breakdown",
    "D5": "crew unavailability",
    "D6": "service-duration overrun",
    "D7": "apron congestion",
    "D8": "stand-release overrun",
    "D9": "arrival delay",
    "D10": "combined disruption",
}

DISTURBANCE_SETS: Dict[str, Tuple[str, ...]] = {
    "none": (),
    "mixed": ("D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8"),
    "transport": ("D1", "D2", "D3", "D7"),
    "resource": ("D4", "D5", "D6"),
    "severe": ("D2", "D4", "D5", "D10"),
}
for _f in FAMILIES:
    DISTURBANCE_SETS[_f] = (_f,)


@dataclass
class DisturbanceSpec:
    time: float
    kind: str
    magnitude: float
    payload: Dict[str, object]

    def as_dict(self) -> Dict[str, object]:
        d = {"time": self.time, "kind": self.kind, "magnitude": self.magnitude}
        d.update(self.payload)
        return d


def build_plan(seed: int, horizon: float, family_set: str,
               rate: float = 1.0, lo: float = 3.0, hi: float = 12.0
               ) -> List[Dict[str, object]]:
    """Deterministic disturbance schedule, identical for every method.

    The plan is a function of the seed alone, so two configurations sharing a
    seed experience the *same* apron: paired comparison with common random
    numbers is therefore valid.
    """
    kinds = DISTURBANCE_SETS.get(family_set, DISTURBANCE_SETS["mixed"])
    if not kinds:
        return []
    rng = random.Random(seed * 7717 + 101)
    n = max(4, int(round(rate * horizon / 22.0)))
    plan: List[Dict[str, object]] = []
    for i in range(n):
        t = horizon * (i + 0.6) / (n + 1.0)
        plan.append({"time": round(t, 4), "kind": kinds[i % len(kinds)],
                     "magnitude": round(rng.uniform(lo, hi), 4)})
    return plan


# ---------------------------------------------------------------------------
# Target selection helpers, used by the event engine
# ---------------------------------------------------------------------------


def pick_enroute(sim, binding: Optional[bool], rng) -> Optional[str]:
    """Choose an in-flight GSE movement whose task is (not) currently binding.

    ``binding=True`` selects a movement whose arrival is the active cause of
    its task start, so the delay must cross the max factor.  ``binding=False``
    selects one with positive absorption margin, so a small delay must be
    contained.  ``binding=None`` selects uniformly.
    """
    cands = [tid for tid, m in sim.enroute.items() if m.arrive > sim.t + 1.0]
    if not cands:
        return None
    if binding is None:
        return rng.choice(sorted(cands))
    scored: List[Tuple[float, str]] = []
    for tid in sorted(cands):
        margin = sim.arrival_margin(tid)
        scored.append((margin, tid))
    if binding:
        tight = [t for m, t in scored if m <= 1e-6]
        return rng.choice(tight) if tight else min(scored)[1]
    loose = [t for m, t in scored if m > 3.0]
    return rng.choice(loose) if loose else max(scored)[1]


def pick_switchable(sim, rng) -> Optional[Tuple[str, float]]:
    """A non-binding movement plus the magnitude that makes it binding.

    Delaying by ``margin + eps`` is exactly the D3 criticality switch: the task
    was absorbing the disturbance until this magnitude, and becomes the new
    active cause immediately after it.
    """
    best: List[Tuple[float, str]] = []
    for tid, m in sim.enroute.items():
        if m.arrive <= sim.t + 1.0:
            continue
        margin = sim.arrival_margin(tid)
        if 0.5 < margin < 25.0:
            best.append((margin, tid))
    if not best:
        return None
    margin, tid = rng.choice(sorted(best))
    return tid, margin + max(0.75, 0.15 * margin)
