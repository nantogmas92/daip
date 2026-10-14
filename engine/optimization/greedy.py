"""Greedy marginal-leverage backend for Algorithm 3.

A lightweight backend that selects the intervention with the greatest positive
marginal value per unit of operational disruption,

    Score(u) = sum_z w_z Lambda_z(u) / (c_u + eps),

subject to the exclusivity edges of the candidate interaction graph.  Residual
violations are decremented after each pick so that leverage estimates are not
combined blindly.

Its research role is to separate the value of the *representation* from the
value of optimiser sophistication: if greedy over Algorithm 2 candidates
already captures most of the benefit, then the reduction, not the solver, is
doing the work.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Set, Tuple


def solve(cands, G, violations, leverage, budget: int
          ) -> Tuple[List[int], str, float]:
    remaining = dict((z.zid, z.residual) for z in violations)
    weight = dict((z.zid, z.weight) for z in violations)
    excl: Set[Tuple[str, str]] = set()
    for (a, b), rel in G.edges.items():
        if rel == "exclusive":
            excl.add((a, b))
            excl.add((b, a))

    chosen: List[int] = []
    chosen_ids: List[str] = []
    pool = list(range(len(cands)))
    while pool and len(chosen) < budget and any(v > 1e-9 for v in remaining.values()):
        best, best_score = None, 0.0
        for i in pool:
            u = cands[i]
            if any((u.candidate_id, c) in excl for c in chosen_ids):
                continue
            gain = sum(weight.get(z, 1.0)
                       * min(remaining.get(z, 0.0), leverage(u, z))
                       for z in u.predicted_leverage)
            gain -= max(0.0, u.harm)
            score = gain / (u.cost + 1e-6)
            if score > best_score:
                best, best_score = i, score
        if best is None or best_score <= 1e-9:
            break
        u = cands[best]
        chosen.append(best)
        chosen_ids.append(u.candidate_id)
        pool.remove(best)
        for z in u.predicted_leverage:
            remaining[z] = max(0.0, remaining.get(z, 0.0) - leverage(u, z))
    obj = sum(weight.get(z, 1.0) * r for z, r in remaining.items())
    return chosen, "greedy", obj
