"""Algorithm 1: incremental causal constraint propagation.

The engine receives a physical or operational event -- a GSE travel delay, a
resource breakdown, a service-duration overrun, a crew unavailability, a stand
conflict, apron congestion or an arrival delay -- and updates *only* the
causally affected component of the factor graph.

Four propagation modes are implemented so that the containment mechanism can
be ablated independently of representation fidelity:

    A0  full global recomputation of every factor after every event;
    AS  structural-closure invalidation: every structural descendant of the
        seed is re-evaluated (the usual dirty-subtree scheme);
    A1  incremental propagation with value-equality containment only, with no
        sensitivity stream, so criticality switches are silently missed;
    A2  full Algorithm 1: value, active set and first-order influence must all
        be unchanged before propagation is allowed to terminate.

Only A2 satisfies the termination rule of Proposition C.  Checking values
alone is insufficient: an active-set change may preserve a numerical output
while changing which action can influence that output.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from core.factors import CapacityFactor, Factor
from core.graph import FactorGraph
from core.messages import FactorUpdateMessage
from core.sensitivities import (
    factor_sensitivity,
    recompute_all,
    sens_differs,
)

TOL = 1e-7
SCC_TOL = 1e-4          # local fixed-point convergence tolerance
MAX_SCC_ITERS = 6


@dataclass
class PropagationResult:
    """Delta-Xi_t plus the instrumentation used for PF, CS and T_prop."""

    touched: int = 0                 # distinct factors recomputed -> PF
    evaluations: int = 0             # evaluations including SCC re-sweeps
    messages: List[FactorUpdateMessage] = field(default_factory=list)
    depth: int = 0
    sccs_resolved: int = 0
    changed_decision: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    criticality_switches: int = 0
    contained_at: int = 0            # factors evaluated whose output was stable
    elapsed_ms: float = 0.0
    touched_ids: Set[str] = field(default_factory=set)

    def message_counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for m in self.messages:
            out[m.message_type] = out.get(m.message_type, 0) + 1
        return out


class PropagationEngine:
    """Event-driven factor propagation over the SCC condensation DAG."""

    def __init__(self, model, mode: str = "A2"):
        self.model = model
        self.mode = mode
        self.epoch = 0
        self._collect = True

    # ------------------------------------------------------------------
    @property
    def g(self) -> FactorGraph:
        return self.model.g

    # ------------------------------------------------------------------
    def initial_solve(self) -> PropagationResult:
        t0 = time.perf_counter()
        g = self.g
        g.ensure_scc()
        ev = g.full_recompute()
        recompute_all(g)
        res = PropagationResult(touched=len(g.fac), evaluations=ev)
        res.touched_ids = set(g.fac)
        res.elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return res

    # ------------------------------------------------------------------
    def propagate(self, seed_factors: Sequence[str],
                  collect_messages: bool = True) -> PropagationResult:
        """INCREMENTAL-CAUSAL-CONSTRAINT-PROPAGATION(e_t, G_t)."""
        t0 = time.perf_counter()
        g = self.g
        g.ensure_scc()
        self.epoch += 1
        res = PropagationResult()

        if self.mode == "A0":
            res.evaluations = g.full_recompute()
            res.touched_ids = set(g.fac)
            res.touched = len(res.touched_ids)
            recompute_all(g)
            now = getattr(self.model, "now", 0.0)
            for v in g.var.values():          # every variable was re-derived
                v.stamp = now
            for vid, v in g.var.items():
                if v.decision_facing:
                    res.changed_decision[vid] = (v.value, v.value)
            res.sccs_resolved = g.ensure_scc().n_components
            res.elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return res

        if self.mode == "AS":
            return self._structural_closure(seed_factors, res, t0)

        structural = False
        with_sens = self.mode == "A2"
        self._collect = collect_messages

        # priority queue keyed by the topological index of the producing SCC,
        # so every upstream change reaches a component before it is evaluated
        heap: List[Tuple[int, int, str]] = []
        seen: Set[str] = set()
        depth_of: Dict[str, int] = {}
        counter = 0
        for fid in seed_factors:
            f = g.fac.get(fid)
            if f is None:
                continue
            heapq.heappush(heap, (g.topo_index(f.output), counter, fid))
            counter += 1
            seen.add(fid)
            depth_of[fid] = 0

        guard = 0
        # The structural-closure mode AS re-queues a factor once per incoming
        # structural edge, so its batch count is bounded by the number of
        # structural edges rather than by the number of factors.  A cap sized
        # for the causal modes would truncate AS mid-sweep and silently make it
        # disagree with global recomputation.
        cap = 400 * max(1, len(g.fac))
        while heap:
            guard += 1
            if guard > cap:
                break
            k_idx = heap[0][0]
            batch: List[str] = []
            while heap and heap[0][0] == k_idx:
                _, _, fid = heapq.heappop(heap)
                batch.append(fid)
                seen.discard(fid)
            scc_id = g.scc_of(g.fac[batch[0]].output) if batch else None
            cyclic = bool(scc_id is not None and g.is_cyclic(scc_id))
            res.sccs_resolved += 1

            if cyclic:
                emitted = self._solve_component(scc_id, res, structural,
                                                with_sens, depth_of, batch)
            else:
                emitted = []
                for fid in batch:
                    emitted += self._evaluate_one(fid, res, structural,
                                                  with_sens, depth_of)

            for fid in emitted:
                if fid in seen:
                    continue
                f = g.fac.get(fid)
                if f is None:
                    continue
                heapq.heappush(heap, (g.topo_index(f.output), counter, fid))
                counter += 1
                seen.add(fid)

        res.depth = max(depth_of.values()) if depth_of else 0
        res.elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return res

    # ------------------------------------------------------------------
    def _structural_closure(self, seed_factors: Sequence[str],
                            res: PropagationResult, t0: float
                            ) -> PropagationResult:
        """AS: the usual dirty-subtree scheme.

        Every structural descendant of the seed is invalidated and re-evaluated
        once, in condensation order, with cyclic components iterated to their
        local fixed point.  No containment test is applied: a factor is
        recomputed because it is *reachable*, not because anything about it
        actually changed.  This is the natural baseline against which causal
        containment is measured, and being exact by construction it isolates
        the cost of over-invalidation from any question of correctness.
        """
        g = self.g
        # forward structural closure of the seeds
        closure: Set[str] = set()
        stack = [fid for fid in seed_factors if fid in g.fac]
        while stack:
            fid = stack.pop()
            if fid in closure:
                continue
            closure.add(fid)
            out = g.var.get(g.fac[fid].output)
            if out is None:
                continue
            for cfid in out.consumers:
                if cfid not in closure:
                    stack.append(cfid)

        order = sorted(closure, key=lambda fid: g.topo_index(g.fac[fid].output))
        depth_of: Dict[str, int] = dict((fid, 0) for fid in order)
        done: Set[int] = set()
        for fid in order:
            f = g.fac.get(fid)
            if f is None:
                continue
            cid = g.scc_of(f.output)
            if g.is_cyclic(cid):
                if cid in done:
                    continue
                done.add(cid)
                comp = [k for k in order if g.scc_of(g.fac[k].output) == cid]
                for _ in range(MAX_SCC_ITERS):
                    changed = False
                    for k in comp:
                        out = g.var.get(g.fac[k].output)
                        before = out.value if out is not None else 0.0
                        self._evaluate_one(k, res, True, False, depth_of,
                                           internal=True)
                        after = out.value if out is not None else 0.0
                        if abs(after - before) > SCC_TOL:
                            changed = True
                    if not changed:
                        break
            else:
                self._evaluate_one(fid, res, True, False, depth_of)
        res.sccs_resolved = len(set(g.scc_of(g.fac[k].output) for k in closure))
        res.depth = max(depth_of.values()) if depth_of else 0
        res.elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return res

    # ------------------------------------------------------------------
    def _solve_component(self, scc_id: int, res: PropagationResult,
                         structural: bool, with_sens: bool,
                         depth_of: Dict[str, int],
                         batch: Sequence[str]) -> List[str]:
        """x_K* = Psi_K(x_K*, x_dK): local fixed point before boundary export."""
        g = self.g
        comp_f = [fid for fid, f in g.fac.items() if g.scc_of(f.output) == scc_id]
        downstream: Set[str] = set()
        base_depth = max((depth_of.get(fid, 0) for fid in batch), default=0)
        for _ in range(MAX_SCC_ITERS):
            changed = False
            for fid in comp_f:
                depth_of.setdefault(fid, base_depth)
                out = g.var.get(g.fac[fid].output)
                before = out.value if out is not None else 0.0
                em = self._evaluate_one(fid, res, True, with_sens, depth_of,
                                        internal=True)
                after = out.value if out is not None else 0.0
                if em:
                    downstream.update(em)
                if abs(after - before) > SCC_TOL:
                    changed = True
            if not changed:
                break
        # only boundary outputs leave the component
        return [fid for fid in downstream
                if fid in g.fac and g.scc_of(g.fac[fid].output) != scc_id]

    # ------------------------------------------------------------------
    def _evaluate_one(self, fid: str, res: PropagationResult, structural: bool,
                      with_sens: bool, depth_of: Dict[str, int],
                      internal: bool = False) -> List[str]:
        g = self.g
        f = g.fac.get(fid)
        if f is None:
            return []
        out = g.var.get(f.output)
        if out is None:
            return []
        res.evaluations += 1
        if fid not in res.touched_ids:
            res.touched_ids.add(fid)
            res.touched += 1
        if g._journal_on:
            g.journal_var(out.vid)
            g.journal_fac(fid)

        old_val, old_act, old_sens = out.value, f.active, out.sens

        val, act = f.evaluate(g)
        out.value = val
        out.active_set = act
        f.active = act
        out.version += 1
        out.stamp = getattr(self.model, "now", 0.0)   # simulation-time stamp

        changed_val = abs(val - old_val) > TOL
        changed_act = act != old_act
        changed_sens = False
        # A factor is evaluated only because something upstream changed, so in
        # A2 its sensitivity row is always recomputed.  Gating this on a value
        # or active-set change would miss the case that motivates the second
        # stream in the first place: an upstream *sensitivity* change that
        # leaves this factor's output numerically identical.  The row would
        # then go stale and S_x would disagree with global recomputation even
        # though every state value matched.
        if with_sens:
            # The truncation convention must depend on whether the *output*
            # lies in a cyclic component, not on whether this call happened to
            # arrive through the component solver.  Keying it on the call site
            # would make a variable's sensitivity row depend on the order in
            # which it was reached, and the incremental result would then
            # disagree with the global reference on exactly those variables.
            in_cycle = g.is_cyclic(g.scc_of(out.vid))
            new_sens = factor_sensitivity(g, f, out, in_cycle=in_cycle)
            changed_sens = sens_differs(old_sens, new_sens)
            out.sens = new_sens

        if changed_act and old_act:
            res.criticality_switches += 1
        if changed_val and out.decision_facing:
            res.changed_decision[out.vid] = (old_val, val)

        if self._collect and (changed_val or changed_act or changed_sens):
            self._emit(res, f, out, old_val, val, old_act, act,
                       changed_val, changed_act, changed_sens)

        propagate = structural or changed_val or changed_act or changed_sens
        if not propagate:
            res.contained_at += 1
            return []

        d = depth_of.get(fid, 0) + 1
        nxt: List[str] = []
        for cfid in out.consumers:
            if cfid == fid:
                continue
            if cfid not in depth_of or depth_of[cfid] > d:
                depth_of[cfid] = d
            nxt.append(cfid)
        return nxt

    # ------------------------------------------------------------------
    def _emit(self, res, f, out, old_val, val, old_act, act,
              changed_val, changed_act, changed_sens) -> None:
        def msg(mtype: str) -> FactorUpdateMessage:
            return FactorUpdateMessage(
                epoch=self.epoch, message_type=mtype, source_node=f.fid,
                destination_factor=out.vid, old_value=old_val, new_value=val,
                delta=val - old_val, active_set_before=old_act,
                active_set_after=act,
                causal_support=tuple(sorted(act))[:4],
                timestamp=getattr(self.model, "now", 0.0))

        if changed_val:
            if out.kind in ("eta", "Rv", "Rh", "Rp", "S"):
                res.messages.append(msg("ETA_CHANGED"))
            elif out.kind == "C":
                res.messages.append(msg("TASK_COMPLETION_CHANGED"))
            elif out.kind == "ready":
                res.messages.append(msg("AIRCRAFT_READY_TIME_CHANGED"))
            elif out.kind == "L":
                res.messages.append(msg("SLACK_CHANGED"))
                if (old_val >= 0.0) != (val >= 0.0):
                    res.messages.append(msg("FEASIBILITY_CHANGED"))
        if changed_act:
            res.messages.append(msg("CRITICALITY_CHANGED"))
        if isinstance(f, CapacityFactor):
            res.messages.append(msg("RESERVATION_CHANGED"))
            if f.binding:
                res.messages.append(msg("RESOURCE_BINDING_CHANGED"))
        if changed_sens:
            res.messages.append(msg("ACTION_SENSITIVITY_CHANGED"))

    # ------------------------------------------------------------------
    def sensitivities_of(self, target: str) -> Dict[str, float]:
        """S_z: the sparse nonzero action sensitivities of a target."""
        v = self.g.var.get(target)
        return dict(v.sens) if v is not None else {}
