"""Counterfactual bundle validation.

No intervention reaches the execution controller until the *joint* effect of
the whole bundle has been propagated by Algorithm 1 on a journalled copy of
the affected SCC closure.  By Proposition G the resulting state equals what
full global recomputation under the same bundle would produce, so a bundle
that repairs one aircraft by pushing three others past their targets is
rejected here rather than discovered on the apron.

Two modes support the ablation:

    V0  execute the reduced model's prediction directly, with no validation;
    V1  full validation as described above.

The acceptance test is deliberately global.  It is not enough that the target
violation shrinks: the priority-weighted residual over *every* tracked flight
must improve by more than MIN_GLOBAL_GAIN, and no previously satisfied flight
may become violated.  A flight that is already late may become later if the
aggregate still improves; such worsened violations are recorded, not refused.
The joint state must also stay acyclic.  These are checks on the *predicted*
planning slack.  Physical feasibility (equipment capability and status, crew
qualification and availability, position capacity, stand-side compatibility)
is enforced by the simulator's execution and start rules, and the realised
schedule is audited independently after each run.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from core.messages import InterventionCandidate, ValidatedInterventionBundle

from .algorithm1_propagation import PropagationEngine
from .counterfactual import apply_bundle
from .violations import Violation

MODEL_TOL = 1.5            # epsilon_model, minutes
MIN_GLOBAL_GAIN = 0.5      # minimum weighted repair required to act (minutes)


@dataclass
class ValidationOutcome:
    bundle: Optional[ValidatedInterventionBundle]
    elapsed_ms: float
    touched: int
    accepted: bool
    realised: Dict[str, float]
    global_gain: float = 0.0
    reason: str = ""                 # accepted | cyclic | new_violation | low_gain
    worsened: Tuple[str, ...] = ()   # already-late targets predicted to get later


class CounterfactualValidator:
    def __init__(self, model, mode: str = "V1"):
        self.model = model
        self.mode = mode

    # ------------------------------------------------------------------
    def validate(self, sel: Sequence[InterventionCandidate],
                 predicted: Dict[str, float],
                 violations: Sequence[Violation],
                 bundle_id: str = "b0",
                 epoch: int = 0) -> ValidationOutcome:
        t0 = time.perf_counter()
        if not sel:
            return ValidationOutcome(None, 0.0, 0, False, {})

        if self.mode == "V0":
            vb = ValidatedInterventionBundle(
                bundle_id=bundle_id,
                candidate_ids=tuple(u.candidate_id for u in sel),
                predicted_target_effects=dict(predicted),
                validated_target_effects=dict(predicted),
                resource_feasibility=True, safety_feasibility=True,
                affected_flights=self._flights(sel),
                affected_resources=self._resources(sel),
                validation_epoch=epoch)
            return ValidationOutcome(vb, (time.perf_counter() - t0) * 1000.0,
                                     0, True, dict(predicted))

        model = self.model
        g = model.g
        eng = PropagationEngine(model, mode="A1")
        before = dict((vid, v.value) for vid, v in g.var.items() if v.kind == "L")
        realised: Dict[str, float] = {}
        new_viol: List[str] = []
        worse: List[str] = []
        touched = 0
        gain = 0.0

        g.start_journal()
        try:
            seeds = apply_bundle(model, [
                (u.candidate_type, u.affected_resources, u.theta) for u in sel])
            if g.ensure_scc().n_cyclic:
                return ValidationOutcome(None, (time.perf_counter()-t0)*1000., 0,
                                         False, {}, 0., "cyclic")
            res = eng.propagate(seeds, collect_messages=False) if seeds else None
            touched = res.touched if res else 0
            for z in violations:
                v = g.var.get(z.zid)
                if v is not None and z.zid in before:
                    realised[z.zid] = z.direction * (v.value - before[z.zid])
            scn = model.sim.scn
            pre = post = 0.0
            for vid, old in before.items():
                v = g.var.get(vid)
                if v is None:
                    continue
                ent = vid.split(":", 1)[1]
                fl = scn.flights.get(ent)
                if fl is None:
                    tk = scn.tasks.get(ent)
                    fl = scn.flights.get(tk.flight) if tk else None
                w = fl.weight if fl is not None else 1.0
                pre += w * max(0.0, -old)
                post += w * max(0.0, -v.value)
                if old >= 0.0 and v.value < -1e-6:
                    new_viol.append(vid)
                elif old < 0.0 and v.value < old - 1e-6:
                    worse.append(vid)
            gain = pre - post
        finally:
            g.rollback()

        err = 0.0
        for z, p in predicted.items():
            err = max(err, abs(p - realised.get(z, 0.0)))
        accepted = not new_viol and gain > MIN_GLOBAL_GAIN
        reason = ("accepted" if accepted else
                  "new_violation" if new_viol else "low_gain")

        vb = ValidatedInterventionBundle(
            bundle_id=bundle_id,
            candidate_ids=tuple(u.candidate_id for u in sel),
            predicted_target_effects=dict(predicted),
            validated_target_effects=realised,
            resource_feasibility=True,       # enforced at execution, audited post hoc
            safety_feasibility=True,
            affected_flights=self._flights(sel),
            affected_resources=self._resources(sel),
            new_violations=tuple(new_viol),
            validation_epoch=epoch,
            model_error=err)
        return ValidationOutcome(vb, (time.perf_counter() - t0) * 1000.0,
                                 touched, accepted, realised, gain, reason,
                                 tuple(worse))

    @staticmethod
    def _flights(sel) -> Tuple[str, ...]:
        out = set()
        for u in sel:
            out.update(u.affected_flights)
        return tuple(sorted(out))

    @staticmethod
    def _resources(sel) -> Tuple[str, ...]:
        out = set()
        for u in sel:
            out.update(u.resource_requirements)
        return tuple(sorted(out))
