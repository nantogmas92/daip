"""Configurable stand-side compatibility and separation matrix.

Concurrency rules for aircraft servicing are operator- and airport-specific
and are governed by local procedures.  This module therefore does *not*
hard-code universal safety rules.  It provides a configurable matrix

    K_{q1,q2} in {compatible, incompatible, conditional}

over service codes, together with the temporal separation required for a
``conditional`` pair, and loads a named profile.  The default profile shipped
here is an *illustrative operator profile* used to make the experiments
reproducible; it is a modelling parameter of this study, not an assertion
about what any regulator or operator permits.  Replacing the profile with a
site-specific one changes only this file.

Two mechanisms are distinguished:

``incompatible``  the two services may not overlap at all; the later one is
                  pushed behind the earlier one's completion.
``conditional``   the two services may overlap only subject to a stated
                  precaution, modelled as a required temporal separation.
``compatible``    no stand-side restriction beyond position capacity.

Position capacity is handled separately by ``stands.py``: two services needing
the same physical service position conflict regardless of their K entry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

COMPATIBLE = "compatible"
INCOMPATIBLE = "incompatible"
CONDITIONAL = "conditional"


@dataclass
class SafetyProfile:
    """One named operator profile of stand-side concurrency rules."""

    name: str
    relations: Dict[FrozenSet[str], str] = field(default_factory=dict)
    separations: Dict[FrozenSet[str], float] = field(default_factory=dict)
    note: str = ""

    def relation(self, a: str, b: str) -> str:
        if a == b:
            return COMPATIBLE
        return self.relations.get(frozenset((a, b)), COMPATIBLE)

    def separation(self, a: str, b: str) -> float:
        return self.separations.get(frozenset((a, b)), 0.0)

    def constrained_pairs(self) -> List[Tuple[str, str, str, float]]:
        out: List[Tuple[str, str, str, float]] = []
        for key, rel in sorted(self.relations.items(), key=lambda kv: sorted(kv[0])):
            a, b = sorted(key)
            out.append((a, b, rel, self.separations.get(key, 0.0)))
        return out

    def as_rows(self) -> List[Dict[str, object]]:
        return [{"service_a": a, "service_b": b, "relation": r, "separation": s}
                for (a, b, r, s) in self.constrained_pairs()]


def _profile(name: str, pairs: Sequence[Tuple[str, str, str, float]],
             note: str) -> SafetyProfile:
    rel: Dict[FrozenSet[str], str] = {}
    sep: Dict[FrozenSet[str], float] = {}
    for (a, b, r, s) in pairs:
        key = frozenset((a, b))
        rel[key] = r
        if s:
            sep[key] = float(s)
    return SafetyProfile(name, rel, sep, note)


# ---------------------------------------------------------------------------
# Named profiles
# ---------------------------------------------------------------------------

PROFILES: Dict[str, SafetyProfile] = {}

# P0: no stand-side concurrency restriction beyond position capacity.  Used as
# the reference against which the cost of the restriction itself is measured.
PROFILES["none"] = _profile("none", (), "position capacity only")

# P1: the illustrative default profile used in the reported experiments.
PROFILES["default"] = _profile(
    "default",
    (
        # fuelling and passenger movement: modelled as permitted only with a
        # stated precaution, represented as a temporal separation
        ("FUEL", "BOARD", CONDITIONAL, 2.0),
        ("FUEL", "DEBOARD", CONDITIONAL, 2.0),
        # fuelling and other vehicles working the same wing/underside region
        ("FUEL", "CATER_A", CONDITIONAL, 1.5),
        # lavatory servicing and potable-water servicing are kept apart in
        # time in many operator procedures
        ("LAV", "WATER", INCOMPATIBLE, 0.0),
        # cabin work and catering both need the forward left door
        ("CLEAN", "CATER_F", INCOMPATIBLE, 0.0),
        # boarding cannot overlap cabin work at the same door
        ("CLEAN", "BOARD", INCOMPATIBLE, 0.0),
        ("CATER_F", "BOARD", INCOMPATIBLE, 0.0),
        # pushback preparation clears the nose area of other equipment
        ("PBPREP", "GPU", CONDITIONAL, 1.0),
        ("PBPREP", "INSPECT", INCOMPATIBLE, 0.0),
        # hold work on the same side
        ("BAGUNLD", "BAGLD", INCOMPATIBLE, 0.0),
        ("CARGOUNLD", "CARGOLD", INCOMPATIBLE, 0.0),
    ),
    "illustrative operator profile; replace with site-specific procedures")

# P2: a stricter profile used in the sensitivity analysis of Section XI.
PROFILES["strict"] = _profile(
    "strict",
    tuple(list(PROFILES["default"].constrained_pairs()) + [
        ("FUEL", "BOARD", INCOMPATIBLE, 0.0),
        ("FUEL", "CLEAN", CONDITIONAL, 2.0),
        ("FUEL", "BAGLD", CONDITIONAL, 1.0),
    ]),
    "stricter illustrative profile: no passenger movement during fuelling")


def get_profile(name: str) -> SafetyProfile:
    if name not in PROFILES:
        raise KeyError("unknown safety profile %r; available: %s"
                       % (name, sorted(PROFILES)))
    return PROFILES[name]


# ---------------------------------------------------------------------------
# Derived per-flight constraint instantiation
# ---------------------------------------------------------------------------


def stand_conflicts(tasks, profile: SafetyProfile
                    ) -> List[Tuple[str, str, str, float]]:
    """Instantiate K over the task set of one flight.

    Returns (task_a, task_b, relation, separation) for every constrained pair
    present in this flight.  Pairs already ordered by the turnaround DAG are
    dropped: a precedence edge already enforces the separation, so adding a
    second constraint would double-count it and inflate G^struct with edges
    that can never bind.
    """
    out: List[Tuple[str, str, str, float]] = []
    ids = sorted(tasks)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            ta, tb = tasks[a], tasks[b]
            rel = profile.relation(ta.code, tb.code)
            if rel == COMPATIBLE:
                continue
            if _ordered(tasks, a, b) or _ordered(tasks, b, a):
                continue
            out.append((a, b, rel, profile.separation(ta.code, tb.code)))
    return out


def _ordered(tasks, a: str, b: str) -> bool:
    """True when a precedes b through the turnaround DAG."""
    stack = [b]
    seen = set()
    while stack:
        cur = stack.pop()
        if cur == a:
            return True
        if cur in seen or cur not in tasks:
            continue
        seen.add(cur)
        stack.extend(tasks[cur].predecessors)
    return False
