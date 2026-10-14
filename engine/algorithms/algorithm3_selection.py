"""Algorithm 3: intervention selection over the candidate interaction graph.

Algorithm 3 selects a feasible intervention *bundle* from the reduced
candidate representation:

    U^cand + G^cand + Xi_t  ->  U_t^star

The key design requirement is solver independence.  CP-SAT, MILP and greedy
selection operate on the same candidate representation and are evaluated under
the same post-selection propagation semantics, so that the effect of the
representation can be separated from the effect of the optimiser.

A solver proposal is never executed directly.  The bundle prediction is built
from individual and selected pairwise counterfactual effects, but for several
simultaneous interventions the joint effect is nonlinear, because the bundle
can change active max constraints, resource sequencing, apron congestion,
position occupancy, task criticality and action sensitivities.  In general

    sum_u Delta-Xi_u  !=  Delta-Xi_{u_1..u_k},

so every proposed bundle is validated by Algorithm 1 on a journalled
counterfactual state before it may become U^star.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from core.messages import InterventionCandidate

from .interaction import InteractionGraph
from .violations import Violation

LEV_CAP = 1e6
DEFAULT_BUDGET = 4       # B_I: maximum simultaneous interventions


@dataclass
class SelectionResult:
    chosen: List[InterventionCandidate] = field(default_factory=list)
    predicted: Dict[str, float] = field(default_factory=dict)
    status: str = "none"
    elapsed_ms: float = 0.0
    objective: float = 0.0
    rounds: int = 1


class InterventionSelector:
    """Solver-neutral bundle selection."""

    def __init__(self, backend: str = "cpsat", budget: int = DEFAULT_BUDGET,
                 time_limit: float = 0.75, robust_beta: float = 0.0,
                 two_stage: bool = False):
        self.backend = backend
        self.budget = budget
        self.time_limit = time_limit
        self.beta = robust_beta
        self.two_stage = two_stage

    # ------------------------------------------------------------------
    def _lev(self, u: InterventionCandidate, z: str) -> float:
        """Conservative leverage Lambda_z(u) - beta sigma_zu."""
        lam = u.leverage_on(z)
        if self.beta > 0.0:
            lam -= self.beta * u.uncertainty.get(z, 0.0)
        if not (lam == lam) or abs(lam) > LEV_CAP:      # NaN / runaway guard
            return 0.0
        return max(0.0, lam)

    # ------------------------------------------------------------------
    def select(self, cands: Sequence[InterventionCandidate],
               G: InteractionGraph, violations: Sequence[Violation],
               nogoods: Sequence[Tuple[str, ...]] = ()) -> SelectionResult:
        t0 = time.perf_counter()
        if not cands:
            return SelectionResult(status="empty",
                                   elapsed_ms=(time.perf_counter() - t0) * 1000.0)
        cl = list(cands)
        if self.backend == "greedy":
            from optimization.greedy import solve as gsolve
            sel, status, obj = gsolve(cl, G, violations, self._lev, self.budget)
        elif self.backend == "beam":
            from optimization.beam import solve as bsolve
            sel, status, obj = bsolve(cl, G, violations, self._lev, self.budget,
                                      nogoods)
        elif self.backend == "topk":
            from optimization.beam import solve_topk as tsolve
            sel, status, obj = tsolve(cl, G, violations, self._lev, self.budget)
        elif self.backend == "milp":
            from optimization.milp_optional import solve as msolve
            sel, status, obj = msolve(cl, G, violations, self._lev, self.budget,
                                      self.time_limit)
            if status == "unavailable":
                from optimization.cpsat import solve as csolve
                sel, status, obj = csolve(cl, G, violations, self._lev,
                                          self.budget, self.time_limit, nogoods,
                                          self.two_stage)
        else:
            from optimization.cpsat import solve as csolve
            sel, status, obj = csolve(cl, G, violations, self._lev, self.budget,
                                      self.time_limit, nogoods, self.two_stage)
        chosen = [cl[i] for i in sel]
        res = SelectionResult(chosen=chosen,
                              predicted=self._predict(chosen, G, violations),
                              status=status, objective=obj)
        res.elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return res

    # ------------------------------------------------------------------
    def _predict(self, sel: Sequence[InterventionCandidate],
                 G: InteractionGraph,
                 violations: Sequence[Violation]) -> Dict[str, float]:
        """The reduced model's predicted improvement, before validation."""
        ids = set(u.candidate_id for u in sel)
        out: Dict[str, float] = {}
        for z in violations:
            v = sum(self._lev(u, z.zid) for u in sel)
            for (a, b), kap in G.kappa.items():
                if a in ids and b in ids:
                    v += kap.get(z.zid, 0.0)
            out[z.zid] = v
        return out
