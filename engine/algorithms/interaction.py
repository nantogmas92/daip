"""The candidate interaction graph G_t^cand.

Proposition H (exact additivity under independent causal support) keeps this
graph sparse: two candidates whose causal closures are disjoint, and which do
not alter each other's binding structure, are exactly additive, so no
interaction term is needed.  We therefore materialise an edge only when the
two candidates share a physical resource or a factor support, and we evaluate
a joint counterfactual (to obtain kappa_z) only for those pairs.

    kappa_z(u, v) = Lambda_z(u, v) - Lambda_z(u) - Lambda_z(v)

with kappa > 0 synergistic and kappa < 0 antagonistic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Set, Tuple

from core.messages import InterventionCandidate

from .algorithm1_propagation import PropagationEngine
from .counterfactual import apply_bundle
from .violations import Violation

EXCLUSIVE_PREFIXES = ("task:", "gse:", "crew:", "position:")


@dataclass
class InteractionGraph:
    nodes: List[InterventionCandidate] = field(default_factory=list)
    edges: Dict[Tuple[str, str], str] = field(default_factory=dict)
    kappa: Dict[Tuple[str, str], Dict[str, float]] = field(default_factory=dict)
    elapsed_ms: float = 0.0
    joint_evals: int = 0

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    def relation_counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for r in self.edges.values():
            out[r] = out.get(r, 0) + 1
        return out

    def exclusive_pairs(self) -> List[Tuple[str, str]]:
        return [k for k, v in self.edges.items() if v == "exclusive"]


class InteractionBuilder:
    def __init__(self, model, engine: PropagationEngine, max_joint: int = 32):
        self.model = model
        self.engine = engine
        self.max_joint = max_joint

    # ------------------------------------------------------------------
    @staticmethod
    def _exclusive(u: InterventionCandidate, v: InterventionCandidate) -> bool:
        """Alternative decisions for the same physical commitment."""
        gse_types = ("REASSIGN_GSE", "ACTIVATE_RESERVE_GSE", "SWAP_GSE_SEQUENCE",
                     "REROUTE_GSE")
        crew_types = ("REASSIGN_CREW", "SWAP_CREW_SEQUENCE")
        ua, va = u.affected_resources, v.affected_resources
        if not ua or not va:
            return False
        # two ways of resolving the same task's equipment
        if u.candidate_type in gse_types and v.candidate_type in gse_types:
            if ua[0] == va[0]:
                return True
        if u.candidate_type in crew_types and v.candidate_type in crew_types:
            if ua[0] == va[0]:
                return True
        # the same unit cannot be handed two different tasks at once
        if u.candidate_type in ("REASSIGN_GSE", "ACTIVATE_RESERVE_GSE") \
                and v.candidate_type in ("REASSIGN_GSE", "ACTIVATE_RESERVE_GSE"):
            if len(ua) > 1 and len(va) > 1 and ua[1] == va[1]:
                return True
        if u.candidate_type == "REASSIGN_CREW" and v.candidate_type == "REASSIGN_CREW":
            if len(ua) > 1 and len(va) > 1 and ua[1] == va[1]:
                return True
        # a position change and an admission advance for the same task
        if set((u.candidate_type, v.candidate_type)) == \
                set(("CHANGE_SERVICE_POSITION", "ADVANCE_POSITION_ADMISSION")):
            if ua[0] == va[0]:
                return True
        # opposite resequencing of the same pair
        if u.candidate_type == v.candidate_type == "RESEQUENCE_TASKS":
            if set(ua[:2]) == set(va[:2]):
                return True
        return False

    # ------------------------------------------------------------------
    def build(self, cands: Sequence[InterventionCandidate],
              violations: Sequence[Violation]) -> InteractionGraph:
        t0 = time.perf_counter()
        G = InteractionGraph(nodes=list(cands))
        n = len(cands)
        shared: List[Tuple[int, int]] = []
        for i in range(n):
            for j in range(i + 1, n):
                u, v = cands[i], cands[j]
                if self._exclusive(u, v):
                    G.edges[(u.candidate_id, v.candidate_id)] = "exclusive"
                    continue
                ru, rv = set(u.resource_requirements), set(v.resource_requirements)
                if ru & rv:
                    shared.append((i, j))
        shared.sort(key=lambda ij: -(
            max(cands[ij[0]].predicted_leverage.values(), default=0.0)
            + max(cands[ij[1]].predicted_leverage.values(), default=0.0)))
        for (i, j) in shared[: self.max_joint]:
            u, v = cands[i], cands[j]
            kap = self._kappa(u, v, violations)
            G.joint_evals += 1
            G.kappa[(u.candidate_id, v.candidate_id)] = kap
            hi = max(kap.values(), default=0.0)
            lo = min(kap.values(), default=0.0)
            if hi > 0.25:
                rel = "synergistic"
            elif lo < -0.25:
                rel = "antagonistic"
            else:
                rel = "compatible"
            G.edges[(u.candidate_id, v.candidate_id)] = rel
        G.elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return G

    # ------------------------------------------------------------------
    def _kappa(self, u: InterventionCandidate, v: InterventionCandidate,
               violations: Sequence[Violation]) -> Dict[str, float]:
        model = self.model
        g = model.g
        eng = PropagationEngine(model, mode="A1")
        before = dict((z.zid, g.var[z.zid].value)
                      for z in violations if z.zid in g.var)
        kap: Dict[str, float] = {}
        g.start_journal()
        try:
            seeds = apply_bundle(model, [
                (u.candidate_type, u.affected_resources, u.theta),
                (v.candidate_type, v.affected_resources, v.theta)])
            if not seeds:
                return {}
            if g.ensure_scc().n_cyclic:
                return {}
            eng.propagate(seeds, collect_messages=False)
            for z in violations:
                var = g.var.get(z.zid)
                if var is None or z.zid not in before:
                    continue
                joint = z.direction * (var.value - before[z.zid])
                kap[z.zid] = joint - u.leverage_on(z.zid) - v.leverage_on(z.zid)
        finally:
            g.rollback()
        return kap
