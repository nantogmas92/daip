"""Apron movement: routing, traversal time, congestion occupancy.

Kept separate from the event engine so that the movement model can be
replaced (for example by a microscopic taxi-route simulator) without touching
the turnaround state machine or the decision controller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class Movement:
    """One vehicle repositioning between two apron nodes."""

    unit: str
    task: Optional[str]
    src: str
    dst: str
    legs: Tuple[str, ...]
    depart: float
    arrive: float
    productive: bool          # False when the leg performs no service (deadhead)

    @property
    def duration(self) -> float:
        return max(0.0, self.arrive - self.depart)


def plan_movement(apron, unit, dst_node: str, now: float, task: Optional[str],
                  productive: bool = True, detour: Sequence[str] = ()
                  ) -> Movement:
    """Route ``unit`` to ``dst_node`` and time the traversal under congestion."""
    src = unit.current_location
    if detour:
        legs = apron.alternate_route(src, dst_node, detour)
        if not legs:
            legs = apron.route(src, dst_node)
    else:
        legs = apron.route(src, dst_node)
    tt = apron.travel_time(legs, unit.speed, congested=True)
    return Movement(unit.resource_id, task, src, dst_node, tuple(legs),
                    now, now + tt, productive)


def edge_loads(active: Sequence[Movement], now: float) -> Dict[str, float]:
    """rho_e(t): vehicles currently occupying each service-road segment.

    A vehicle is attributed to every leg of its current route, weighted by the
    share of the route still ahead of it.  Attributing it to all legs equally
    would overstate congestion on segments already cleared; attributing it only
    to the current leg would understate the load a long movement imposes.
    """
    loads: Dict[str, float] = {}
    for m in active:
        if not m.legs or m.arrive <= now:
            continue
        span = max(1e-9, m.arrive - m.depart)
        frac = max(0.0, min(1.0, (m.arrive - now) / span))
        share = frac / len(m.legs) * len(m.legs)      # total presence = frac
        per = share / len(m.legs)
        for e in m.legs:
            loads[e] = loads.get(e, 0.0) + per
    return loads


def congestion_delay(apron, legs: Sequence[str], speed: float) -> float:
    """Traversal time attributable to congestion rather than distance."""
    return (apron.travel_time(legs, speed, congested=True)
            - apron.travel_time(legs, speed, congested=False))


def deadhead_share(movements: Sequence[Movement]) -> float:
    tot = sum(m.duration for m in movements)
    if tot <= 0.0:
        return 0.0
    return sum(m.duration for m in movements if not m.productive) / tot
