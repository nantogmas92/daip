"""Minimum stand-feasible turnaround time used to set readiness targets.

A target of the form ``scheduled on-block + theta * span`` is meaningful only
if ``span`` is a duration the turnaround can actually achieve.  The
precedence-only critical path is not: several services share a single-capacity
access zone (DEBOARD, CLEAN, CATER_F and BOARD all use the forward-left door)
and the compatibility profile forbids further overlaps, so the stand-side
minimum is on average about 1.2 times the precedence-only path.  With the
legacy basis most targets were infeasible before any travel, contention or
disturbance occurred.

``minimum_turnaround`` returns the exact minimum makespan of one flight under
precedence, service-position capacity, incompatible pairs and conditional
start separations, with unconstrained equipment and crews and zero travel.  It
is a small single-flight scheduling problem solved to optimality by CP-SAT.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, Sequence, Tuple

from .safety_constraints import CONDITIONAL, INCOMPATIBLE

RES = 100      # integer resolution: 0.01 min


def minimum_turnaround(tasks, conflicts: Sequence[Tuple[str, str, str, float]],
                       capacity: Dict[str, int]) -> float:
    key = (tuple(sorted((t.tid.split("/", 1)[1], round(t.duration, 3),
                         t.position,
                         tuple(sorted(p.split("/", 1)[1] for p in t.predecessors)))
                        for t in tasks.values())),
           tuple(sorted((a.split("/", 1)[1], b.split("/", 1)[1], r, s)
                        for (a, b, r, s) in conflicts)),
           tuple(sorted(capacity.items())))
    return _solve(key)


@lru_cache(maxsize=4096)
def _solve(key) -> float:
    from ortools.sat.python import cp_model

    rows, conflicts, capacity = key
    capacity = dict(capacity)
    horizon = sum(int(round(d * RES)) for _, d, _, _ in rows) + 1
    m = cp_model.CpModel()
    start, end, iv = {}, {}, {}
    for code, dur, pos, preds in rows:
        d = max(1, int(round(dur * RES)))
        s = m.NewIntVar(0, horizon, "s_" + code)
        e = m.NewIntVar(0, horizon, "e_" + code)
        start[code], end[code] = s, e
        iv[code] = m.NewIntervalVar(s, d, e, "i_" + code)
    for code, _dur, _pos, preds in rows:
        for p in preds:
            if p in end:
                m.Add(start[code] >= end[p])
    by_pos: Dict[str, list] = {}
    for code, _dur, pos, _preds in rows:
        by_pos.setdefault(pos, []).append(code)
    for pos, members in by_pos.items():
        cap = int(capacity.get(pos, 1))
        if len(members) > cap:
            if cap == 1:
                m.AddNoOverlap([iv[c] for c in members])
            else:
                m.AddCumulative([iv[c] for c in members], [1] * len(members), cap)
    for a, b, rel, sep in conflicts:
        if a not in iv or b not in iv:
            continue
        if rel == INCOMPATIBLE:
            m.AddNoOverlap([iv[a], iv[b]])
        elif rel == CONDITIONAL:
            first = m.NewBoolVar("o_%s_%s" % (a, b))
            gap = int(round(sep * RES))
            m.Add(start[b] >= start[a] + gap).OnlyEnforceIf(first)
            m.Add(start[a] >= start[b] + gap).OnlyEnforceIf(first.Not())
    mk = m.NewIntVar(0, horizon, "makespan")
    m.AddMaxEquality(mk, list(end.values()))
    m.Minimize(mk)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = 10.0
    solver.parameters.random_seed = 0
    status = solver.Solve(m)
    if status != cp_model.OPTIMAL:
        raise RuntimeError("minimum turnaround not proven optimal")
    return solver.Value(mk) / RES
