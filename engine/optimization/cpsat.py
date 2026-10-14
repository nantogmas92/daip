"""OR-Tools CP-SAT backend for Algorithm 3.

CP-SAT is the primary fixed centralised selection backend of this study.  It
is held constant across every representation ablation, so that the reported
differences are attributable to the representation rather than to the
optimiser.

The lexicographic objective lexmin(J1, J2, J3) is realised by exact
scalarisation: the weight on J1 exceeds any attainable value of (10 J2 + J3),
so no trade of residual violation against intervention cost is possible.  A
two-stage solve is also provided and is used to verify the scalarisation on
small instances.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

SCALE = 10          # integer scaling of minutes for CP-SAT
WEIGHT_SCALE = 10   # preserves the study's 1.0, 1.6 and 2.6 flight priorities
LEX_BIG = 100000    # lexicographic separation weight
LEV_CAP = 1e6       # sanity bound on a single leverage coefficient


def solve(cands, G, violations, leverage, budget: int, time_limit: float,
          nogoods: Sequence[Tuple[str, ...]] = (),
          two_stage: bool = False) -> Tuple[List[int], str, float]:
    """Return (selected indices, status, objective)."""
    from ortools.sat.python import cp_model

    idx = dict((u.candidate_id, i) for i, u in enumerate(cands))
    zs = list(violations)

    cost_scale = min(budget, len(cands)) + 1
    lower_bound = cost_scale * sum(sorted((max(0, int(round(u.cost * SCALE)))
                                         for u in cands), reverse=True)[:budget]) + budget
    lex_scale = lower_bound + 1

    def build(stage1_bound: Optional[int], scalar=False):
        m = cp_model.CpModel()
        y = [m.NewBoolVar("y%d" % i) for i in range(len(cands))]
        q: Dict[Tuple[str, str], object] = {}
        for (a, b), rel in G.edges.items():
            if a not in idx or b not in idx:
                continue
            ia, ib = idx[a], idx[b]
            if rel == "exclusive":
                m.Add(y[ia] + y[ib] <= 1)
            elif (a, b) in G.kappa and any(abs(v) > 0.25
                                           for v in G.kappa[(a, b)].values()):
                qv = m.NewBoolVar("q_%d_%d" % (ia, ib))
                m.AddMultiplicationEquality(qv, [y[ia], y[ib]])
                q[(a, b)] = qv

        # joint resource exclusivity: one unit, crew or task hosts one action
        byres: Dict[str, List[int]] = {}
        for u in cands:
            for r in u.resource_requirements:
                if r.startswith(("gse:", "crew:", "task:", "position:")):
                    byres.setdefault(r, []).append(idx[u.candidate_id])
        for r, members in byres.items():
            if len(members) > 1 and r.startswith(("gse:", "crew:", "task:")):
                m.Add(sum(y[i] for i in members) <= 1)
        m.Add(sum(y) <= budget)
        for ng in nogoods:
            # Rejection of B does not imply rejection of a complementary superset.
            if any(c not in idx for c in ng):
                continue
            sel = {idx[c] for c in ng}
            m.Add(sum(1-y[i] if i in sel else y[i] for i in range(len(cands))) >= 1)

        j1_terms = []
        for z in zs:
            r_int = int(round(z.residual * SCALE))
            s = m.NewIntVar(0, max(0, r_int), "s_%s" % z.zid.replace(":", "_"))
            expr = [int(round(leverage(u, z.zid) * SCALE)) * y[idx[u.candidate_id]]
                    for u in cands if leverage(u, z.zid) > 0]
            for (a, b), qv in q.items():
                k = G.kappa.get((a, b), {}).get(z.zid, 0.0)
                if k == k and 0.25 < abs(k) <= LEV_CAP:
                    expr.append(int(round(k * SCALE)) * qv)
            m.Add(s >= r_int - sum(expr) if expr else s >= r_int)
            j1_terms.append(int(round(z.weight * WEIGHT_SCALE)) * s)
        harm = [int(round(u.harm * SCALE * WEIGHT_SCALE)) * y[idx[u.candidate_id]] for u in cands
                if u.harm == u.harm and 0.0 < u.harm <= LEV_CAP]
        j1 = (sum(j1_terms) if j1_terms else 0) + (sum(harm) if harm else 0)
        j2 = sum(int(round(u.cost * SCALE)) * y[idx[u.candidate_id]] for u in cands)
        j3 = sum(y)
        if scalar:
            m.Minimize(lex_scale * j1 + cost_scale * j2 + j3)
        elif stage1_bound is None:
            m.Minimize(j1)
        else:
            m.Add(j1 <= stage1_bound)
            m.Minimize(j2 * cost_scale + j3)
        return m, y

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = 1

    if two_stage:
        m1, y1 = build(None)
        st1 = solver.Solve(m1)
        if st1 not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return [], "infeasible", 0.0
        bound = int(round(solver.ObjectiveValue()))
        m2, y2 = build(bound)
        s2 = cp_model.CpSolver()
        s2.parameters.max_time_in_seconds = time_limit
        s2.parameters.num_search_workers = 1
        st2 = s2.Solve(m2)
        if st2 in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            sel = [i for i in range(len(cands)) if s2.Value(y2[i])]
            return sel, ("optimal" if st1 == cp_model.OPTIMAL and st2 == cp_model.OPTIMAL else "feasible"), \
                s2.ObjectiveValue()
        sel = [i for i in range(len(cands)) if solver.Value(y1[i])]
        return sel, "stage1", float(bound)

    m, y = build(None, scalar=True)
    st = solver.Solve(m)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return [], "infeasible", 0.0
    sel = [i for i in range(len(cands)) if solver.Value(y[i])]
    return sel, ("optimal" if st == cp_model.OPTIMAL else "feasible"), \
        solver.ObjectiveValue()
