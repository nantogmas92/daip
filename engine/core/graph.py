"""Factor-graph runtime: the three graphs, the SCC condensation, the sandbox.

The runtime maintains the three structures that the manuscript keeps
explicitly distinct:

    G^struct   every potential dependency (static within an epoch);
    G^bind_t   only the currently binding dependencies (argmax / saturated);
    G^inf_t    the weighted action-influence graph (nonzero derivatives).

``structural_edges`` exposes the first, ``binding_edges`` the second and
``influence_edges`` the third.  The invariant G^inf subset-of G^struct is
checked by ``verify_influence_subset``: an influence edge without a structural
edge would let a change bypass propagation and silently break Proposition C.

The counterfactual sandbox is a *journal*, not a copy.  Recording an undo log
costs work proportional to the causal closure actually touched, which is what
makes localized counterfactual evaluation inside Algorithms 2 and 3
affordable; deep-copying the graph would cost work proportional to the size of
the whole apron.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

from .factors import Factor
from .scc import Condensation, tarjan_scc
from .variables import Variable


class FactorGraph:
    """Container for variables, factors and the structural condensation."""

    def __init__(self, representation=None):
        self.rep = representation
        self.var: Dict[str, Variable] = {}
        self.fac: Dict[str, Factor] = {}
        self.edge_length: Dict[str, float] = {}
        self.cond: Optional[Condensation] = None
        self._dirty = True
        # set by rewire/add_factor when an added edge may have closed a cycle;
        # cleared whenever the condensation is rebuilt
        self._cycle_risk = False
        # factors whose output changed cyclic status at the last decomposition
        self.cycle_changed: Set[str] = set()
        # journal used for O(|R(u)|) counterfactual sandboxes
        self._journal_on = False
        self._jvar: Dict[str, tuple] = {}
        self._jfac: Dict[str, dict] = {}
        self._jcons: Dict[str, list] = {}
        self._jderived: Optional[Tuple[bool, Optional[Condensation]]] = None

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def add_var(self, vid: str, kind: str, value: float = 0.0, entity: str = "",
                decision_facing: bool = False, layer: str = "task") -> Variable:
        v = Variable(vid, kind, value, entity=entity,
                     decision_facing=decision_facing, layer=layer)
        self.var[vid] = v
        return v

    def add_factor(self, f: Factor) -> Factor:
        self.fac[f.fid] = f
        if f.output in self.var:
            self.var[f.output].producer = f.fid
        for vid in f.inputs:
            if vid in self.var:
                self.var[vid].consumers.append(f.fid)
        self._dirty = True
        self._cycle_risk = True
        return f

    def remove_factor(self, fid: str) -> None:
        f = self.fac.pop(fid, None)
        if f is None:
            return
        for vid in f.inputs:
            v = self.var.get(vid)
            if v is not None and fid in v.consumers:
                v.consumers.remove(fid)
        self._dirty = True

    def rewire(self, f: Factor, inputs: Sequence[str]) -> bool:
        """Replace a factor's input list, keeping consumer lists consistent."""
        if list(f.inputs) == list(inputs):
            return False
        self.journal_fac(f.fid)
        for vid in list(f.inputs) + list(inputs):
            self.journal_consumers(vid)
        old = set(f.inputs)
        added = [vid for vid in inputs if vid not in old]
        for vid in f.inputs:
            v = self.var.get(vid)
            if v is not None and f.fid in v.consumers:
                v.consumers.remove(f.fid)
        f.inputs = list(inputs)
        for vid in inputs:
            v = self.var.get(vid)
            if v is not None:
                v.consumers.append(f.fid)
        self._dirty = True
        # Removing edges can never close a cycle; an added edge u -> f.output
        # closes one exactly when f.output already reaches u.  Each rewire is
        # checked against the graph *including* the edges added by earlier
        # rewires of the same edit, so sequential checks compose exactly.
        if added and self._may_close_cycle(f.output, added):
            self._cycle_risk = True
        return True

    def _may_close_cycle(self, output: str, added: Sequence[str]) -> bool:
        """Whether ``output`` reaches any of ``added`` (then edge u->output cycles)."""
        added_set = set(added)
        if output in added_set:
            return True
        seen = {output}
        stack = [output]
        while stack:
            v = self.var.get(stack.pop())
            if v is None:
                continue
            for fid in v.consumers:
                f = self.fac.get(fid)
                if f is None:
                    continue
                w = f.output
                if w in added_set:
                    return True
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        return False

    # ------------------------------------------------------------------
    # the three graphs
    # ------------------------------------------------------------------
    def structural_edges(self) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        for f in self.fac.values():
            for vid in f.inputs:
                if vid in self.var and f.output in self.var:
                    out.append((vid, f.output))
        return out

    def binding_edges(self) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        for f in self.fac.values():
            for vid in f.active:
                if vid in self.var and f.output in self.var:
                    out.append((vid, f.output))
        return out

    def influence_edges(self) -> List[Tuple[str, str, float]]:
        out: List[Tuple[str, str, float]] = []
        for f in self.fac.values():
            for vid, w in f.jacobian(self).items():
                if vid in self.var and f.output in self.var and abs(w) > 1e-12:
                    out.append((vid, f.output, w))
        return out

    def verify_influence_subset(self) -> List[Tuple[str, str]]:
        """Return influence edges that lack a structural counterpart."""
        struct = set(self.structural_edges())
        bad: List[Tuple[str, str]] = []
        for (a, b, _w) in self.influence_edges():
            if (a, b) not in struct:
                bad.append((a, b))
        return bad

    def graph_sizes(self) -> Dict[str, int]:
        return {
            "n_var": len(self.var),
            "n_fac": len(self.fac),
            "e_struct": len(self.structural_edges()),
            "e_bind": len(self.binding_edges()),
            "e_inf": len(self.influence_edges()),
        }

    # ------------------------------------------------------------------
    # SCC decomposition on G^struct
    # ------------------------------------------------------------------
    def compute_scc(self) -> Condensation:
        """Re-derive the condensation of G^struct.

        A structural edit -- a resequencing hold, a resource reassignment, a
        newly released flight -- can move a variable into or out of a cyclic
        component.  The sensitivity convention differs between the two cases
        (contributions from inside a component are truncated), so a variable
        whose cyclic status flips must have its row re-derived even when
        nothing else about it changed.  ``cycle_changed`` reports exactly those
        factors so that the caller can seed them.
        """
        prev = {}
        if self.cond is not None:
            for vid in self.var:
                cid = self.cond.component_of(vid)
                prev[vid] = self.cond.is_cyclic(cid) if cid >= 0 else False
        adj: Dict[str, List[str]] = dict((v, []) for v in self.var)
        for f in self.fac.values():
            if f.output not in self.var:
                continue
            for vid in f.inputs:
                if vid in self.var:
                    adj[vid].append(f.output)
        comps = tarjan_scc(adj)
        self.cond = Condensation(comps, adj)
        for f in self.fac.values():
            f.scc = self.cond.component_of(f.output)
        self.cycle_changed = set()
        for f in self.fac.values():
            vid = f.output
            if vid not in self.var:
                continue
            now = self.cond.is_cyclic(self.cond.component_of(vid))
            if vid in prev and prev[vid] != now:
                self.cycle_changed.add(f.fid)
        self._dirty = False
        self._cycle_risk = False
        return self.cond

    def ensure_scc(self) -> Condensation:
        if self._dirty or self.cond is None:
            # Inside a counterfactual sandbox over an acyclic baseline, an edit
            # that added no cycle-closing edge keeps the graph acyclic, so every
            # component is still a singleton and is_cyclic stays False; the only
            # stale part is the topological order, which propagate corrects by
            # re-queueing consumers on every value change.  Rebuilding the
            # condensation from scratch would cost O(V+E) Tarjan per counter-
            # factual query and dominate the decision latency, so it is skipped
            # here and reserved for the live loop and for edits that may have
            # closed a cycle.
            if (self._journal_on and self.cond is not None
                    and self.cond.n_cyclic == 0 and not self._cycle_risk):
                self._dirty = False
                return self.cond
            return self.compute_scc()
        return self.cond

    def topo_index(self, vid: str) -> int:
        return self.ensure_scc().topo_index(vid)

    def scc_of(self, vid: str) -> int:
        return self.ensure_scc().component_of(vid)

    def is_cyclic(self, cid: int) -> bool:
        return self.ensure_scc().is_cyclic(cid)

    # ------------------------------------------------------------------
    # counterfactual sandbox (journal + rollback)
    # ------------------------------------------------------------------
    def start_journal(self) -> None:
        self._journal_on = True
        self._jvar = {}
        self._jfac = {}
        self._jcons = {}
        # A counterfactual must leave no trace whatsoever on the live model.
        # That includes the *derived* structures: a synthetic rewire marks the
        # condensation stale, and if that flag survived the rollback the next
        # decomposition would reclassify variables under a topology that was
        # never actually adopted -- changing the sensitivity truncation
        # convention for variables the counterfactual never touched.
        self._jderived = (self._dirty, self.cond, set(self.cycle_changed),
                          {k:f.scc for k,f in self.fac.items()},
                          self._cycle_risk)

    def journal_var(self, vid: str) -> None:
        if self._journal_on and vid not in self._jvar:
            v = self.var.get(vid)
            if v is not None:
                self._jvar[vid] = v.snapshot()

    def journal_fac(self, fid: str) -> None:
        if self._journal_on and fid not in self._jfac:
            f = self.fac.get(fid)
            if f is not None:
                self._jfac[fid] = dict(f.__dict__)

    def journal_consumers(self, vid: str) -> None:
        if self._journal_on and vid not in self._jcons:
            v = self.var.get(vid)
            if v is not None:
                self._jcons[vid] = list(v.consumers)

    def rollback(self) -> None:
        for vid, snap in self._jvar.items():
            v = self.var.get(vid)
            if v is not None:
                v.restore(snap)
        for fid, d in self._jfac.items():
            f = self.fac.get(fid)
            if f is not None:
                f.__dict__.update(d)
        for vid, cons in self._jcons.items():
            v = self.var.get(vid)
            if v is not None:
                v.consumers = cons
        if self._jderived is not None:
            self._dirty, self.cond, self.cycle_changed, old_scc, \
                self._cycle_risk = self._jderived
            for fid, cid in old_scc.items():
                self.fac[fid].scc = cid
            self._jderived = None
        self._journal_on = False
        self._jvar = {}
        self._jfac = {}
        self._jcons = {}

    @property
    def journal_size(self) -> int:
        return len(self._jvar) + len(self._jfac)

    # ------------------------------------------------------------------
    # evaluation
    # ------------------------------------------------------------------
    def full_recompute(self, sweeps: int = 200) -> int:
        """A0 reference: evaluate every factor in condensation order.

        Sweeps repeat until no value or active set changes, so a cyclic
        component is driven to its fixed point rather than stopped after a fixed
        number of passes.  An earlier version stopped after three sweeps, which
        is exact for an acyclic graph -- one topological sweep suffices and the
        second confirms it -- but leaves a cyclic component short of its fixed
        point, making the reference less accurate than the incremental solver
        it is meant to audit.  For acyclic graphs the cost is unchanged.
        """
        self.ensure_scc()
        order = sorted(self.fac.values(), key=lambda f: self.topo_index(f.output))
        evaluations = 0
        for _ in range(sweeps):
            changed = False
            for f in order:
                val, act = f.evaluate(self)
                evaluations += 1
                out = self.var.get(f.output)
                if out is None:
                    continue
                if abs(val - out.value) > 1e-9 or act != f.active:
                    changed = True
                out.value = val
                out.active_set = act
                f.active = act
            if not changed:
                break
        return evaluations

    # ------------------------------------------------------------------
    def flight_vars(self, fid: str) -> Dict[str, float]:
        return dict((k, v.value) for k, v in self.var.items() if v.entity == fid)

    def decision_state(self) -> Dict[str, float]:
        """Xi_t: the decision-facing projection of the representation."""
        return dict((k, v.value) for k, v in self.var.items() if v.decision_facing)
