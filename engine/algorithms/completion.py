"""Anytime completion of a recovery decision.

Screening prices single actions and the selector coordinates them through
their pairwise interactions.  Two situations still leave recoverable delay:

* no selected bundle survives joint validation (or the query budget runs out
  first), although an acceptable bundle was already evaluated exactly during
  the decision; and
* the best repair contains actions that become useful only once other actions
  have been applied, which single-action screening cannot see.

``complete`` therefore (1) falls back to the best acceptable bundle evaluated so
far when the selector returns nothing, and (2) deepens the bundle: it applies
the bundle tentatively, rebuilds the search region of the controller's own mode
for the targets that are still violated on the updated plan, and adds the
region action whose joint evaluation is acceptable, lowers the weighted
residual by more than ``eps_gain`` and has the lowest score, for at most
``rounds`` additions within the bundle-size limit.  Every evaluation is charged
to the decision's planning oracle, so completion only spends budget that
screening, coordination and validation left unused.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from .algorithm1_propagation import PropagationEngine
from .counterfactual import apply_bundle
from .literature_recovery import BudgetExceeded, compatible, raw_candidate
from .violations import ViolationDetector

ROUNDS = 3


def region_after(ctrl, bundle) -> Optional[list]:
    """Search region for the targets still violated once ``bundle`` is applied."""
    model, g = ctrl.model, ctrl.model.g
    g.start_journal()
    try:
        seeds = apply_bundle(model, [(a.candidate_type, a.entities, a.theta) for a in bundle])
        if g.ensure_scc().n_cyclic:
            return None
        if seeds:
            PropagationEngine(model, mode=ctrl.engine.mode).propagate(seeds, collect_messages=False)
        left = ViolationDetector(model).detect(model.sim.t)
        return ctrl.generator.proposals(left) if left else []
    finally:
        g.rollback()


def deepen(ctrl, oracle, base, base_val) -> Optional[Tuple[list, Any]]:
    """Best acceptable one-action extension of ``base``, or None."""
    if len(base) >= ctrl.spec.budget:
        return None
    region = region_after(ctrl, base)
    if not region:
        return None
    gen = ctrl.generator
    have = {a.signature for a in base}
    best = None
    for k, (kind, ents, theta) in enumerate(region):
        if not gen.feasible(kind, ents, theta):
            continue
        act = raw_candidate(gen, (kind, tuple(ents), theta), 10000 + k)
        if act.signature in have:
            continue
        if ctrl.spec.urgency and not ctrl._urgency_filter([act], ctrl._completion_targets):
            continue
        bundle = list(base) + [act]
        if not compatible(bundle, ctrl.spec.budget):
            continue
        val = oracle.evaluate(bundle)          # may raise BudgetExceeded
        if not math.isfinite(val.delay) or not oracle.acceptable(val):
            continue
        if val.delay < base_val.delay - ctrl.spec.eps_gain and (best is None
                                                              or val.score < best[1].score):
            best = (bundle, val)
    return best


def complete(ctrl, targets, oracle, chosen) -> Tuple[List, Dict[str, Any]]:
    """Fallback and deepening after the selector's decision ``chosen``."""
    info: Dict[str, Any] = dict(completion="selected" if chosen else "none")
    cur, cur_val = list(chosen), None
    if cur:
        cur_val = oracle.cache.get(tuple(a.signature for a in cur))
        if cur_val is None or not oracle.acceptable(cur_val):
            cur = []
    best = list(oracle.best[0])
    if ctrl.spec.urgency and len(ctrl._urgency_filter(best, targets)) < len(best):
        best = []                              # the urgency rule restricts the fallback too
    if not cur and best:
        cur, cur_val = best, oracle.best[1]
        info["completion"] = "fallback"
    ctrl._completion_targets = targets
    try:
        for _ in range(ROUNDS if cur else 0):
            ext = deepen(ctrl, oracle, cur, cur_val)
            if ext is None:
                break
            cur, cur_val = ext
            info["completion"] = "deepened"
    except BudgetExceeded:
        info["budget_exhausted"] = True
    return cur, info
