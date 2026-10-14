"""Optional MILP reference backend (OR-Tools linear solver, CBC).

Used only to verify the CP-SAT formulation on small instances: the two must
select bundles of equal objective value.  It is not used for any reported
scalability result, because CP-SAT is the fixed backend of the main study.

Pairwise interaction products y_u y_v are linearised in the usual way with
auxiliary binaries q_uv and the three McCormick constraints.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple


def solve(cands, G, violations, leverage, budget: int, time_limit: float
          ) -> Tuple[List[int], str, float]:
    try:
        from ortools.linear_solver import pywraplp
    except ImportError:                        # pragma: no cover
        return [], "unavailable", 0.0

    s = pywraplp.Solver.CreateSolver("CBC")
    if s is None:                              # pragma: no cover
        return [], "unavailable", 0.0
    s.SetTimeLimit(int(time_limit * 1000))

    idx = dict((u.candidate_id, i) for i, u in enumerate(cands))
    y = [s.BoolVar("y%d" % i) for i in range(len(cands))]
    q: Dict[Tuple[str, str], object] = {}
    for (a, b), rel in G.edges.items():
        if a not in idx or b not in idx:
            continue
        ia, ib = idx[a], idx[b]
        if rel == "exclusive":
            s.Add(y[ia] + y[ib] <= 1)
        elif (a, b) in G.kappa:
            qv = s.BoolVar("q%d_%d" % (ia, ib))
            s.Add(qv <= y[ia])
            s.Add(qv <= y[ib])
            s.Add(qv >= y[ia] + y[ib] - 1)
            q[(a, b)] = qv

    byres: Dict[str, List[int]] = {}
    for u in cands:
        for r in u.resource_requirements:
            if r.startswith(("gse:", "crew:", "task:")):
                byres.setdefault(r, []).append(idx[u.candidate_id])
    for r, members in byres.items():
        if len(members) > 1:
            s.Add(sum(y[i] for i in members) <= 1)
    s.Add(sum(y) <= budget)

    obj = s.Objective()
    for z in violations:
        sv = s.NumVar(0.0, max(0.0, z.residual), "s_" + z.zid.replace(":", "_"))
        expr = sum(leverage(u, z.zid) * y[idx[u.candidate_id]] for u in cands)
        for (a, b), qv in q.items():
            expr += G.kappa.get((a, b), {}).get(z.zid, 0.0) * qv
        s.Add(sv >= z.residual - expr)
        obj.SetCoefficient(sv, 1000.0 * z.weight)
    for i, u in enumerate(cands):
        obj.SetCoefficient(y[i], u.cost + 1000.0 * max(0.0, u.harm))
    obj.SetMinimization()

    st = s.Solve()
    if st not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return [], "infeasible", 0.0
    sel = [i for i in range(len(cands)) if y[i].solution_value() > 0.5]
    status = "optimal" if st == pywraplp.Solver.OPTIMAL else "feasible"
    return sel, status, s.Objective().Value()
