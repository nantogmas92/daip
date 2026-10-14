"""Beam and top-k selection backends for Algorithm 3.

Both operate on the same priced candidate representation as the greedy and
CP-SAT backends: per-candidate leverage Lambda_z(u), the interaction graph's
exclusivity edges, the priced pairwise interaction terms kappa_z(u, v) and the
harm estimate.

``solve_topk`` is the ablation floor: it repeats greedy's static score without
the residual decrement, so it shows what the residual-aware incremental pass
buys.

``solve`` is a beam search over feasible bundles.  Its objective is the
continuous counterpart of the CP-SAT lexicographic model -- weighted residual
reduction with per-target saturation, kappa synergy/antagonism terms of priced
pairs, and harm -- subject to the same exclusivity and joint resource
constraints, with rejected bundles (no-goods) excluded as exact sets.  Unlike
CP-SAT it needs no solver, is deterministic, and runs in microseconds; unlike
greedy it prices *interactions*, which is the information the candidate
interaction graph exists to provide.  With beam width 1 and no kappa terms it
degenerates to greedy's search without the cost normalisation.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Set, Tuple

BEAM_WIDTH = 8          # states kept per depth
KAPPA_ACTIVE = 0.25     # priced pairs below this |kappa| carry no term


def _exclusivity(G, ids) -> Set[Tuple[str, str]]:
    excl: Set[Tuple[str, str]] = set()
    for (a, b), rel in G.edges.items():
        if rel == "exclusive" and a in ids and b in ids:
            excl.add((a, b))
            excl.add((b, a))
    return excl


def _joint_resource_groups(cands, ids) -> List[Set[int]]:
    """Candidates sharing a gse:/crew:/task: requirement are alternatives."""
    byres: Dict[str, Set[int]] = {}
    for i, u in enumerate(cands):
        if u.candidate_id not in ids:
            continue
        for r in u.resource_requirements:
            if r.startswith(("gse:", "crew:", "task:")):
                byres.setdefault(r, set()).add(i)
    return [m for m in byres.values() if len(m) > 1]


def _kappa(G, ia: str, ib: str) -> Dict[str, float]:
    k = G.kappa.get((ia, ib))
    if k is None:
        k = G.kappa.get((ib, ia), {})
    return k


def _score(cands, G, violations, leverage, chosen: Tuple[int, ...],
           resid: Dict[str, float]) -> Tuple[float, float]:
    """(j1, cost): weighted saturated predicted residual + harm, then cost."""
    ids = set(cands[i].candidate_id for i in chosen)
    j1 = 0.0
    for z in violations:
        red = 0.0
        for i in chosen:
            u = cands[i]
            if z.zid in u.predicted_leverage:
                red += leverage(u, z.zid)
        for a in chosen:
            for b in chosen:
                if a >= b:
                    continue
                ka, kb = cands[a].candidate_id, cands[b].candidate_id
                k = _kappa(G, ka, kb).get(z.zid, 0.0)
                if k == k and abs(k) > KAPPA_ACTIVE:
                    red += k
        j1 += z.weight * max(0.0, resid[z.zid] - min(resid[z.zid], red))
    for i in chosen:
        u = cands[i]
        if u.harm == u.harm:
            j1 += max(0.0, u.harm)
    cost = sum(cands[i].cost for i in chosen)
    return j1, cost


def solve(cands, G, violations, leverage, budget: int,
          nogoods: Sequence[Tuple[str, ...]] = ()) -> Tuple[List[int], str, float]:
    """Beam search over feasible bundles; returns (indices, status, objective)."""
    ids = set(u.candidate_id for u in cands)
    excl = _exclusivity(G, ids)
    groups = _joint_resource_groups(cands, ids)
    resid = dict((z.zid, z.residual) for z in violations)
    nogood_sets = set(frozenset(ng) for ng in nogoods)

    def feasible(state: Tuple[int, ...], i: int) -> bool:
        uid = cands[i].candidate_id
        for j in state:
            if (cands[j].candidate_id, uid) in excl:
                return False
        for g in groups:
            if i in g and any(j in g for j in state):
                return False
        return True

    beams: List[Tuple[Tuple[float, float, Tuple[int, ...]], Tuple[int, ...]]] = []
    states: List[Tuple[int, ...]] = [()]
    for _depth in range(budget):
        scored = []
        for st in states:
            for i in range(len(cands)):
                if i in st or not feasible(st, i):
                    continue
                nxt = tuple(sorted(st + (i,)))
                if frozenset(cands[j].candidate_id for j in nxt) in nogood_sets:
                    continue
                j1, cost = _score(cands, G, violations, leverage, nxt, resid)
                scored.append(((j1, cost, nxt), nxt))
        if not scored:
            break
        scored.sort(key=lambda t: (t[0][0], t[0][1], t[0][2]))
        beams = scored[:BEAM_WIDTH]
        states = [nxt for _k, nxt in beams]
    best = min(beams, key=lambda t: (t[0][0], t[0][1], t[0][2])) if beams else None
    if best is None:
        return [], "beam", 0.0
    (j1, _c, _st), chosen = best
    return list(chosen), "beam", j1


def solve_topk(cands, G, violations, leverage, budget: int
               ) -> Tuple[List[int], str, float]:
    """Static score, no residual decrement: the selection-ablation floor."""
    remaining = dict((z.zid, z.residual) for z in violations)
    weight = dict((z.zid, z.weight) for z in violations)
    ids = set(u.candidate_id for u in cands)
    excl = _exclusivity(G, ids)
    scored = []
    for i, u in enumerate(cands):
        gain = sum(weight.get(z, 1.0) * min(remaining.get(z, 0.0), leverage(u, z))
                   for z in u.predicted_leverage)
        gain -= max(0.0, u.harm) if u.harm == u.harm else 0.0
        scored.append((gain / (u.cost + 1e-6), i))
    scored.sort(key=lambda t: (-t[0], t[1]))
    chosen: List[int] = []
    for _s, i in scored:
        if len(chosen) >= budget:
            break
        uid = cands[i].candidate_id
        if any((uid, cands[j].candidate_id) in excl for j in chosen):
            continue
        chosen.append(i)
    obj = sum(weight.get(z, 1.0) * r for z, r in remaining.items())
    return chosen, "topk", obj
