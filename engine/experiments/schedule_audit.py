"""Independent post-execution audit of a realised apron schedule.

The simulator enforces its start rules when a service begins.  This module
checks the *outcome* separately, from the final task records and the event log
only, so a defect in the start rules or in an intervention's execution would
show up here rather than being hidden by the same code that caused it.

Checked constraints (all with a 1e-6 min tolerance):

    precedence      every predecessor and dynamic hold finished before the start
    on_block        no service starts before its aircraft is on block
    stand           aircraft occupying one stand have disjoint [on-block, ready]
    unit            one equipment unit never serves two tasks at once
    crew            one crew never serves two tasks at once
    outage          no unit or crew serves during a breakdown or absence
    position        service-position occupancy never exceeds its capacity
    incompatible    incompatible services of one aircraft never overlap
    conditional     conditional pairs keep their start-to-start separation
    unfinished      every released flight reached aircraft-ready
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from airport.safety_constraints import CONDITIONAL, INCOMPATIBLE

TOL = 1e-6


def _overlaps(intervals: List[Tuple[float, float, str]]) -> int:
    bad = 0
    intervals = sorted(intervals)
    for (s0, e0, _a), (s1, e1, _b) in zip(intervals, intervals[1:]):
        if s1 < e0 - TOL:
            bad += 1
    return bad


def audit_schedule(sim) -> Dict[str, int]:
    scn = sim.scn
    out: Counter = Counter()
    started = [t for t in scn.tasks.values() if t.start == t.start]
    ends = {t.tid: (t.finish if t.state == "done" else float("inf")) for t in started}

    on_block: Dict[str, float] = {}
    outages: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    open_outage: Dict[str, float] = {}
    for ev in sim.events:
        if ev.event_type == "FLIGHT_ON_BLOCK":
            on_block.setdefault(ev.flight_id, ev.simulation_time)
        elif ev.event_type in ("GSE_BREAKDOWN", "CREW_UNAVAILABLE"):
            for r in ev.resource_ids:
                open_outage[r] = ev.simulation_time
        elif ev.event_type in ("GSE_RESTORED", "CREW_RESTORED"):
            for r in ev.resource_ids:
                if r in open_outage:
                    outages[r].append((open_outage.pop(r), ev.simulation_time))
    for r, t0 in open_outage.items():
        outages[r].append((t0, float("inf")))

    for t in started:
        for p in tuple(t.predecessors) + tuple(t.holds):
            pt = scn.tasks.get(p)
            if pt is None:
                continue
            if pt.state != "done" or pt.finish > t.start + TOL:
                out["precedence"] += 1
        ob = on_block.get(t.flight)
        if ob is None or t.start < ob - TOL:
            out["on_block"] += 1

    by_stand: Dict[str, List] = defaultdict(list)
    for fid, fl in scn.flights.items():
        if fid in on_block:
            end = fl.ready if fl.ready == fl.ready else float("inf")
            by_stand[fl.stand].append((on_block[fid], end, fid))
    out["stand"] += sum(_overlaps(v) for v in by_stand.values())

    by_unit: Dict[str, List] = defaultdict(list)
    by_crew: Dict[str, List] = defaultdict(list)
    by_pos: Dict[str, List] = defaultdict(list)
    for t in started:
        iv = (t.start, ends[t.tid], t.tid)
        if t.gse:
            by_unit[t.gse].append(iv)
        if t.crew:
            by_crew[t.crew].append(iv)
        by_pos["%s:%s" % (scn.flights[t.flight].stand, t.position)].append(iv)
    out["unit"] += sum(_overlaps(v) for v in by_unit.values())
    out["crew"] += sum(_overlaps(v) for v in by_crew.values())

    for rid, spans in list(by_unit.items()) + list(by_crew.items()):
        for (s, e, _tid) in spans:
            for (d0, d1) in outages.get(rid, ()):
                if s < d1 - TOL and e > d0 + TOL:
                    out["outage"] += 1

    for pid, spans in by_pos.items():
        stand, pos = pid.split(":", 1)
        cap = scn.stands[stand].positions[pos].capacity
        points = sorted({s for s, _e, _t in spans})
        for x in points:
            if sum(1 for s, e, _t in spans if s <= x + TOL and x < e - TOL) > cap:
                out["position"] += 1

    for fl in scn.flights.values():
        for (a, b, rel, sep) in fl.conflicts:
            ta, tb = scn.tasks.get(a), scn.tasks.get(b)
            if ta is None or tb is None or ta.start != ta.start or tb.start != tb.start:
                continue
            ea, eb = ends[a], ends[b]
            if rel == INCOMPATIBLE and ta.start < eb - TOL and tb.start < ea - TOL:
                out["incompatible"] += 1
            elif rel == CONDITIONAL:
                first, second = (ta, tb) if ta.start <= tb.start else (tb, ta)
                end_first = ends[first.tid]
                if second.start < end_first - TOL and second.start < first.start + sep - TOL:
                    out["conditional"] += 1

    out["unfinished"] = sum(1 for f in scn.flights.values() if f.ready != f.ready)
    for key in ("precedence", "on_block", "stand", "unit", "crew", "outage",
                "position", "incompatible", "conditional"):
        out.setdefault(key, 0)
    return dict(out)
