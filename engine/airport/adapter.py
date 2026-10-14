"""Airport adapter: physical apron state -> generic factor graph, at fidelity rho.

This module is the *only* place where airport semantics meet the causal
engine.  ``core/`` and ``algorithms/`` contain no aircraft, stand or GSE
concept; they see typed variables and local operators.  The adapter converts
flight, GSE, crew and stand events into generic factor-graph state changes, so
the same engine can be re-targeted to another cyber-physical domain by
replacing this file.

Representation ladder
---------------------
R0  independent service view: task release, duration, assigned resource,
    resource-to-task distance.  No turnaround DAG, no aircraft-ready node.
R1  turnaround-DAG aware: flight membership, precedence, time windows, the
    aircraft-ready operator, the departure target and the flight weight.
R2  heterogeneous-resource aware: GSE capability and crew qualification, each
    resource's own calendar, resource sequencing (as an exogenous floor).
R3  aircraft-stand aware: stand occupancy, service-position capacity, the
    configurable stand-side compatibility matrix.
R4  apron-spatial aware: service-road routing, congestion-sensitive ETA.
R5  cross-flight-coupling aware: a resource's readiness is *linked* to the
    predicted completion of the task it serves on the previous aircraft, and
    a flight's release is linked to the stand predecessor's readiness.
R6  full causal representation: R5 plus G^bind, G^inf and the sparse
    sensitivities S_x, enabling causal containment (Algorithm 1) and causal
    intervention screening (Algorithm 2).

Components that rho does not contain are replaced by constant surrogates
rather than being silently correct.  An R2 model, for example, has no
service-position term at all, so its predicted task start is systematically
optimistic whenever a position is the true binding cause.  That deliberate
omission is the source of the representation error E_rep reported in the
experiments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from core.factors import (
    AffineFactor,
    CapacityFactor,
    CongestionFactor,
    ConstFactor,
    IncompatibilityFactor,
    MaxFactor,
    ServiceFactor,
    TravelFactor,
    WorkloadFactor,
)
from core.graph import FactorGraph
from core.variables import NEG_INF

from .safety_constraints import CONDITIONAL, INCOMPATIBLE
from .stands import position_id
from .turnaround_tasks import critical_path_length
from .observations import observed_overrun, resource_release

CREW_TRANSFER = 2.0


# ---------------------------------------------------------------------------
# Representation configuration rho
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Representation:
    """rho = (rho_D, rho_R, rho_S, rho_E, rho_X, rho_I)."""

    name: str
    dag: bool = False            # rho_D: turnaround DAG + aircraft-ready node
    resources: bool = False      # rho_R: GSE capability, crew qualification
    stand: bool = False          # rho_S: positions, occupancy, compatibility
    spatial: bool = False        # rho_E: apron routing + congestion ETA
    crossflight: bool = False    # rho_X: resource and stand handover links
    influence: bool = False      # rho_I: G^bind, G^inf, S_x
    workload: bool = False       # optional crew-workload feedback

    @property
    def level(self) -> int:
        return int(self.name[1:])


LADDER: Dict[str, Representation] = {
    "R0": Representation("R0"),
    "R1": Representation("R1", dag=True),
    "R2": Representation("R2", dag=True, resources=True),
    "R3": Representation("R3", dag=True, resources=True, stand=True),
    "R4": Representation("R4", dag=True, resources=True, stand=True, spatial=True),
    "R5": Representation("R5", dag=True, resources=True, stand=True, spatial=True,
                         crossflight=True),
    "R6": Representation("R6", dag=True, resources=True, stand=True, spatial=True,
                         crossflight=True, influence=True),
}


# ---------------------------------------------------------------------------
# Predictive model
# ---------------------------------------------------------------------------


class TurnaroundModel:
    """The controller's factor-graph view of the apron under rho."""

    def __init__(self, sim, rep: Representation):
        self.sim = sim
        self.rep = rep
        self.g = FactorGraph(rep)
        self.g.edge_length = dict((e, edge.length)
                                  for e, edge in sim.scn.apron.edges.items())
        self.tracked_flights: Set[str] = set()
        self.tracked_tasks: Set[str] = set()
        self.now = 0.0
        self._build_static()

    # ------------------------------------------------------------------
    # static layer: apron congestion multipliers
    # ------------------------------------------------------------------
    def _build_static(self) -> None:
        apron = self.sim.scn.apron
        for eid, edge in apron.edges.items():
            vid = "mu:" + eid
            self.g.add_var(vid, "mu", 1.0, entity=eid, layer="spatial")
            if self.rep.spatial:
                f = CongestionFactor("fmu:" + eid, [], vid, entity=eid,
                                     layer="spatial", alpha=apron.alpha,
                                     beta=apron.beta, cap=edge.capacity)
            else:
                f = ConstFactor("fmu:" + eid, [], vid, entity=eid, layer="spatial")
                f.params["value"] = 1.0
            self.g.add_factor(f)

    # ------------------------------------------------------------------
    # dynamic structure
    # ------------------------------------------------------------------
    def sync(self, now: float) -> List[str]:
        """Add newly released flights, drop completed ones.

        Returns the ids of newly created factors, which must be evaluated once
        before incremental propagation can rely on their cached state.
        """
        self.now = now
        before = set(self.g.fac)
        edits = 0
        active = self.sim.active
        for fid in sorted(self.tracked_flights):
            if fid not in active:
                self._remove_flight(fid)
                edits += 1
        for fid in sorted(active):
            if fid not in self.tracked_flights:
                self._add_flight(fid, now)
                edits += 1
        if edits:
            self.g.compute_scc()
        return [f for f in self.g.fac if f not in before]

    def _remove_flight(self, fid: str) -> None:
        drop_f = [k for k, f in self.g.fac.items() if f.entity == fid]
        for k in drop_f:
            self.g.remove_factor(k)
        drop_v = [k for k, v in self.g.var.items() if v.entity == fid]
        for k in drop_v:
            self.g.var.pop(k, None)
        for tid in sorted(self.tracked_tasks):
            if tid.startswith(fid + "/"):
                self.tracked_tasks.discard(tid)
        self.tracked_flights.discard(fid)
        self.g._dirty = True

    # ------------------------------------------------------------------
    def _add_flight(self, fid: str, now: float) -> None:
        scn = self.sim.scn
        rep = self.rep
        g = self.g
        fl = scn.flights[fid]

        live = [t for t in sorted(fl.tasks) if fl.tasks[t].state != "done"]
        if not live:
            return
        self.tracked_flights.add(fid)

        if not rep.dag:
            self._add_flight_r0(fl, live, now)
            return

        for tid in self._topo(fl, live):
            self._add_task(fl, fl.tasks[tid], now)
            self.tracked_tasks.add(tid)

        # aircraft-ready operator over the departure-enabling terminal set
        term = [("C:" + t) for t in fl.terminal if ("C:" + t) in g.var]
        if not term:
            term = [("C:" + t) for t in live if ("C:" + t) in g.var]
        rid = "ready:" + fid
        g.add_var(rid, "ready", now, entity=fid, layer="flight",
                  decision_facing=True)
        g.add_factor(MaxFactor("fready:" + fid, term, rid, entity=fid,
                               layer="flight"))
        lid = "L:" + fid
        g.add_var(lid, "L", 0.0, entity=fid, layer="flight", decision_facing=True)
        g.add_factor(AffineFactor("fL:" + fid, [rid], lid, entity=fid,
                                  layer="flight", c0=fl.target, coeff={rid: -1.0}))

    @staticmethod
    def _topo(fl, live: Sequence[str]) -> List[str]:
        """Topological order so a predecessor's C variable exists first."""
        live_set = set(live)
        order: List[str] = []
        seen: Set[str] = set()

        def visit(x: str) -> None:
            if x in seen or x not in live_set:
                return
            seen.add(x)
            t = fl.tasks[x]
            for p in tuple(t.predecessors) + tuple(t.holds):
                visit(p)
            order.append(x)

        for tid in sorted(live):
            visit(tid)
        return order

    # ------------------------------------------------------------------
    def _add_flight_r0(self, fl, live: Sequence[str], now: float) -> None:
        """R0: independent services with per-task local due dates.

        There is no aircraft-ready variable and no coupling between services,
        so the controller can only observe that individual tasks are late.
        Its local due dates are derived by charging each task the remaining
        nominal span of its own turnaround, which is the best a task-centric
        model can do without a dependency graph.
        """
        g = self.g
        scn = self.sim.scn
        span = critical_path_length(fl.tasks)
        for tid in live:
            task = fl.tasks[tid]
            self.tracked_tasks.add(tid)
            eta = "eta:" + tid
            g.add_var(eta, "eta", now, entity=fl.flight_id)
            legs, speed, base = self._route_terms(fl, task, now)
            g.add_factor(TravelFactor("feta:" + tid, [], eta,
                                      entity=fl.flight_id, legs=legs, base=base,
                                      speed=speed, congestion=False))
            cvar = "C:" + tid
            g.add_var(cvar, "C", now, entity=fl.flight_id)
            g.add_factor(ServiceFactor("fC:" + tid, [eta], cvar,
                                       entity=fl.flight_id, nominal=task.duration))
            lid = "L:" + tid
            g.add_var(lid, "L", 0.0, entity=fl.flight_id, decision_facing=True)
            due = fl.target - max(0.0, span - self._tail(fl, tid))
            g.add_factor(AffineFactor("fL:" + tid, [cvar], lid,
                                      entity=fl.flight_id, c0=due,
                                      coeff={cvar: -1.0}))

    @staticmethod
    def _tail(fl, tid: str) -> float:
        """Nominal span from this task to the end of its turnaround."""
        memo: Dict[str, float] = {}

        def depth(x: str) -> float:
            if x in memo:
                return memo[x]
            t = fl.tasks[x]
            memo[x] = t.duration + max(
                [depth(s) for s in t.successors if s in fl.tasks] or [0.0])
            return memo[x]

        return depth(tid)

    # ------------------------------------------------------------------
    def _route_terms(self, fl, task, now: float) -> Tuple[Tuple[str, ...], float, float]:
        """Legs, speed and exogenous release of the serving unit's movement."""
        scn = self.sim.scn
        unit = scn.fleet.get(task.gse) if task.gse else None
        if unit is None:
            return (), 1.0, max(now, fl.release)
        dst = scn.stands[fl.stand].node
        legs = scn.apron.route(unit.current_location, dst) if self.rep.spatial \
            else scn.apron.route(unit.current_location, dst)
        base = max(now,self.resource_release(unit))
        mv = self.sim.enroute.get(task.tid)
        if mv is not None:
            base = mv.depart
            legs = mv.legs
        arrived = self.sim.gse_at_stand.get(task.tid)
        if arrived is not None:
            return (), unit.speed, arrived
        return tuple(legs), unit.speed / self._forecast_multiplier(task.tid, "travel"), base

    def resource_release(self, resource):
        return resource_release(self.sim,resource,self._forecast_multiplier,observed=True)

    def _forecast_multiplier(self, tid, kind):
        """Task-keyed information error; never changes execution randomness."""
        import hashlib
        import random
        sigma = self.sim.cfg.observation_sigma
        if not sigma:
            return 1.0
        key = f"{self.sim.cfg.seed}|observation|{kind}|{tid}"
        seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big')
        return max(.25, random.Random(seed).gauss(1., sigma))

    def _mu_inputs(self, legs: Sequence[str]) -> List[str]:
        """Congestion multipliers the factor actually integrates over.

        These must be exactly the legs the factor traverses: registering a
        different leg set would leave G^struct missing an edge, and a
        congestion update on the unregistered segment would then never reach
        the factor, breaking the equivalence of Proposition C.
        """
        if not self.rep.spatial:
            return []
        return [("mu:" + e) for e in legs if ("mu:" + e) in self.g.var]

    # ------------------------------------------------------------------
    def _add_task(self, fl, task, now: float) -> None:
        g = self.g
        rep = self.rep
        fid = fl.flight_id
        tid = task.tid

        # --- precedence-only earliest start ---------------------------
        apre = "S0:" + tid
        g.add_var(apre, "S", now, entity=fid)
        pred = [("C:" + p) for p in tuple(task.predecessors) + tuple(task.holds)]
        floor = fl.release
        f_apre = MaxFactor("fS0:" + tid, [p for p in pred if p in g.var],
                           apre, entity=fid, floor=floor, floor_label="__release__")
        g.add_factor(f_apre)

        start_inputs = [apre]

        # --- GSE readiness (ETA at the stand) -------------------------
        if rep.resources and task.gse_class is not None:
            eta = "eta:" + tid
            g.add_var(eta, "eta", now, entity=fid, layer="resource")
            legs, speed, base = self._route_terms(fl, task, now)
            inputs = self._mu_inputs(legs)
            if rep.crossflight:
                prev = self._resource_predecessor(task, "gse")
                if prev:
                    inputs.append("C:" + prev)
            g.add_factor(TravelFactor(
                "feta:" + tid, inputs, eta, entity=fid, layer="resource",
                legs=legs, base=base, speed=speed,
                congestion=rep.spatial))
            start_inputs.append(eta)

        # --- crew readiness -------------------------------------------
        if rep.resources and task.crew_skill is not None:
            rh = "Rh:" + tid
            g.add_var(rh, "Rh", now, entity=fid, layer="resource")
            inputs = []
            if rep.crossflight:
                prev = self._resource_predecessor(task, "crew")
                if prev:
                    inputs.append("C:" + prev)
            crew = self.sim.scn.crews.get(task.crew) if task.crew else None
            cfloor = max(now,self.resource_release(crew) + CREW_TRANSFER) if crew else now
            g.add_factor(MaxFactor("fRh:" + tid, inputs, rh, entity=fid,
                                   layer="resource", floor=cfloor,
                                   floor_label="__crewfree__"))
            start_inputs.append(rh)

        # --- service-position capacity --------------------------------
        if rep.stand:
            rp = "Rp:" + tid
            g.add_var(rp, "Rp", now, entity=fid, layer="resource")
            cf = CapacityFactor("fRp:" + tid, [apre], rp, entity=fid,
                                layer="resource", capacity=self._pos_capacity(fl, task),
                                resource=position_id(fl.stand, task.position))
            cf.peer_starts = self._position_peers(fl, task, now)
            cf.calendar_mode = self.sim.cfg.capacity_calendar
            cf.peer_start_vars = self._peer_start_vars(cf.peer_starts)
            cf.duration = task.duration
            g.add_factor(cf)
            for pv in self.capacity_inputs(tid, cf)[1:]:
                if pv in g.var:
                    cf.inputs.append(pv)
                    g.var[pv].consumers.append(cf.fid)
            start_inputs.append(rp)

            # --- configurable stand-side compatibility ----------------
            sep_inputs, sep = self._conflict_inputs(fl, task)
            if sep_inputs:
                sfv = "Sf:" + tid
                g.add_var(sfv, "S", NEG_INF, entity=fid)
                g.add_factor(IncompatibilityFactor("fSf:" + tid, sep_inputs, sfv,
                                                   entity=fid, sep=sep))
                start_inputs.append(sfv)

        # --- start and completion -------------------------------------
        sv = "S:" + tid
        g.add_var(sv, "S", now, entity=fid)
        g.add_factor(MaxFactor("fS:" + tid, start_inputs, sv, entity=fid))

        cv = "C:" + tid
        g.add_var(cv, "C", now, entity=fid)
        wvar = None
        crew = self.sim.scn.crews.get(task.crew) if task.crew else None
        rate = crew.rate if (crew is not None and rep.resources) else 1.0
        if rep.workload and crew is not None and crew.workload_gain > 0.0:
            wvar = "W:" + crew.crew_id
            if wvar not in g.var:
                g.add_var(wvar, "W", crew.workload_state, entity="CREW",
                          layer="resource")
                g.add_factor(WorkloadFactor("fW:" + crew.crew_id, [], wvar,
                                            entity="CREW", layer="resource",
                                            base=crew.workload_state,
                                            kappa=crew.workload_gain))
        inputs = [sv] + ([wvar] if wvar else [])
        # gamma must be the crew's own gain: the physical duration is
        # p0 * rate * (1 + gain * W) (Scenario.service_duration), and a model
        # coefficient that differs from it would be a silent prediction bias.
        # W enters as the *observed* workload before this service starts, which
        # is what the physics uses, so the workload chain is feed-forward along
        # the crew sequence and closes no cycle.
        g.add_factor(ServiceFactor(
            "fC:" + tid, inputs, cv, entity=fid, nominal=task.duration,
            rate=rate, gamma=(crew.workload_gain if wvar else 0.0),
            workload_var=wvar))

    # ------------------------------------------------------------------
    def _pos_capacity(self, fl, task) -> int:
        st = self.sim.scn.stands[fl.stand]
        p = st.positions.get(task.position)
        return p.capacity if p is not None else 1

    def _position_peers(self, fl, task, now: float) -> Dict[str, float]:
        """Predicted occupancy calendar of this stand-side service position.

        Peers are the tasks scheduled *ahead* of this one, in the global
        scheduling order, that need the same service position on the same
        stand and are not yet done.  Their completion variables are genuine
        structural inputs, which is what makes the position a shared
        finite-capacity resource in G^struct.

        Orienting the relation by the global key matters: a symmetric relation
        between two same-flight tasks at one position would close a two-cycle
        in which each waits for the other, and that component has no fixed
        point.  The engine would then either diverge or report a predicted
        deadlock, and the local fixed point required by Proposition C would not
        exist.  Ordering the capacity relation by the same key as the resource
        queues keeps the two mutually acyclic.
        """
        scn = self.sim.scn
        out: Dict[str, float] = {}
        pid = position_id(fl.stand, task.position)
        for other in sorted(self.tracked_tasks | set(fl.tasks)):
            if other == task.tid:
                continue
            ot = scn.tasks.get(other)
            if ot is None or ot.state == "done":
                continue
            # same orientation rule as the compatibility matrix: predictive
            # edges only from peers ahead in the global order, plus observed
            # (already started) peers whose completion is pinned
            if not scn.ahead(other, task.tid) and ot.state != "running":
                continue
            ofl = scn.flights.get(ot.flight)
            if ofl is None or position_id(ofl.stand, ot.position) != pid:
                continue
            cv = "C:" + other
            if cv in self.g.var:
                out[cv] = max(now, ot.start if ot.start == ot.start else now)
        return out

    def _peer_start_vars(self, peers: Dict[str, float]) -> Dict[str, str]:
        """Predicted start variable of every unstarted peer (calendar mode)."""
        if not self.sim.cfg.capacity_calendar:
            return {}
        scn = self.sim.scn
        out: Dict[str, str] = {}
        for cv in peers:
            tid = cv[2:]
            tk = scn.tasks.get(tid)
            if tk is None or tk.state in ("running", "done"):
                continue
            if ("S:" + tid) in self.g.var:
                out[cv] = "S:" + tid
        return out

    def capacity_inputs(self, tid: str, cf) -> List[str]:
        """Structural inputs of a capacity factor: request, peer ends, peer starts."""
        extra = sorted(v for v in cf.peer_start_vars.values() if v in self.g.var)
        return ["S0:" + tid] + sorted(cf.peer_starts) + extra

    def _conflict_inputs(self, fl, task) -> Tuple[List[str], float]:
        """Instantiate the configurable K matrix for this task."""
        inputs: List[str] = []
        sep = 0.0
        scn = self.sim.scn
        for (a, b, rel, s) in fl.conflicts:
            other = b if a == task.tid else (a if b == task.tid else None)
            if other is None:
                continue
            ot = scn.tasks.get(other)
            if ot is None:
                continue
            # A peer *ahead* in the global order contributes a predictive edge.
            # A peer behind in the order contributes only once it has actually
            # started: its completion is then pinned to an observed value, so
            # the edge carries no prediction and cannot close a cycle.  Without
            # this second case the model would be blind to exactly the
            # stand-side conflicts the physical system does enforce.
            observed = ot.state in ("running", "done")
            if not scn.ahead(other, task.tid) and not observed:
                continue
            if ot.state == "done" and ot.finish == ot.finish                     and ot.finish <= self.sim.t - 1e-9 and rel == "incompatible":
                continue        # already cleared; no longer constrains
            cv = "C:" + other
            if cv in self.g.var:
                inputs.append(cv)
                if rel == CONDITIONAL:
                    sep = max(sep, s - ot.duration)
        return inputs, sep

    def _resource_predecessor(self, task, kind: str) -> Optional[str]:
        """The task this unit or crew serves immediately before ``task``.

        This single link is the cross-flight coupling mechanism: it is present
        only from R5 upward, and it is what lets a delay on one aircraft reach
        another aircraft that experienced no disturbance of its own.
        """
        scn = self.sim.scn
        if kind == "gse":
            res = scn.fleet.get(task.gse) if task.gse else None
        else:
            res = scn.crews.get(task.crew) if task.crew else None
        if res is None or task.tid not in res.sequence:
            return None
        idx = res.sequence.index(task.tid)
        for j in range(idx - 1, -1, -1):
            prev = res.sequence[j]
            pt = scn.tasks.get(prev)
            if pt is None or pt.state == "done":
                continue
            if ("C:" + prev) in self.g.var:
                return prev
            return None
        return None

    # ------------------------------------------------------------------
    # parameter refresh from the physical state
    # ------------------------------------------------------------------
    def refresh(self, now: float) -> List[str]:
        """Re-bind factor parameters to the observed apron state.

        Returns the ids of factors whose *parameters* changed.  These are the
        seeds handed to Algorithm 1, so propagation begins exactly where the
        physical world contradicted the cached model.
        """
        self.now = now
        scn = self.sim.scn
        rep = self.rep
        g = self.g
        dirty: List[str] = []

        if rep.spatial:
            loads = self.sim.edge_load_view()
            for eid in scn.apron.edges:
                f = g.fac.get("fmu:" + eid)
                if isinstance(f, CongestionFactor):
                    new = loads.get(eid, 0.0)
                    cap = scn.apron.edges[eid].capacity
                    if abs(new - f.observed_load) > 1e-9 or abs(cap - f.cap) > 1e-9:
                        f.observed_load = new
                        f.cap = max(1e-6, cap)
                        dirty.append(f.fid)

        for tid in sorted(self.tracked_tasks):
            task = scn.tasks.get(tid)
            if task is None:
                continue
            fl = scn.flights[task.flight]
            dirty += self._refresh_task(fl, task, now)

        # A rewire above may have changed the condensation.  Any variable whose
        # cyclic status flipped needs its sensitivity row re-derived under the
        # new convention, so those factors are seeded alongside the ordinary
        # parameter changes.
        if g._dirty:
            g.compute_scc()
        if g.cycle_changed:
            dirty += sorted(g.cycle_changed)

        return list(dict.fromkeys(dirty))

    def _refresh_task(self, fl, task, now: float) -> List[str]:
        g = self.g
        rep = self.rep
        scn = self.sim.scn
        tid = task.tid
        dirty: List[str] = []

        # precedence floor and dynamic holds
        f0 = g.fac.get("fS0:" + tid)
        if isinstance(f0, MaxFactor):
            want = [("C:" + p) for p in tuple(task.predecessors) + tuple(task.holds)
                    if ("C:" + p) in g.var]
            if g.rewire(f0, want):
                dirty.append(f0.fid)
            base_floor = fl.release
            if rep.crossflight and fl.prev_on_stand:
                pv = "ready:" + fl.prev_on_stand
                if pv in g.var:
                    base_floor = max(base_floor, g.var[pv].value)
            if task.state in ("running", "done"):
                base_floor = task.start if task.start == task.start else base_floor
            if abs(f0.floor - base_floor) > 1e-9:
                f0.floor = base_floor
                dirty.append(f0.fid)

        # GSE arrival
        fe = g.fac.get("feta:" + tid)
        if isinstance(fe, TravelFactor):
            legs, speed, base = self._route_terms(fl, task, now)
            want = self._mu_inputs(legs)
            if rep.crossflight:
                prev = self._resource_predecessor(task, "gse")
                if prev and ("C:" + prev) in g.var:
                    want.append("C:" + prev)
            changed = g.rewire(fe, want)
            if (abs(fe.base - base) > 1e-9 or abs(fe.speed - speed) > 1e-12
                    or fe.legs != tuple(legs)):
                fe.base, fe.speed, fe.legs = base, speed, tuple(legs)
                changed = True
            if changed:
                dirty.append(fe.fid)

        # crew readiness
        fh = g.fac.get("fRh:" + tid)
        if isinstance(fh, MaxFactor):
            crew = scn.crews.get(task.crew) if task.crew else None
            want = []
            if rep.crossflight:
                prev = self._resource_predecessor(task, "crew")
                if prev and ("C:" + prev) in g.var:
                    want.append("C:" + prev)
            changed = g.rewire(fh, want)
            if crew is not None:
                cf = max(now,self.resource_release(crew) + CREW_TRANSFER)
                if abs(fh.floor - cf) > 1e-9:
                    fh.floor = cf
                    changed = True
            if changed:
                dirty.append(fh.fid)

        # service-position calendar
        fp = g.fac.get("fRp:" + tid)
        if isinstance(fp, CapacityFactor):
            duration = task.duration * (scn.crews[task.crew].rate if task.crew in scn.crews else 1.)
            if abs(fp.duration - duration) > 1e-9:
                fp.duration = duration
                dirty.append(fp.fid)
            peers = self._position_peers(fl, task, now)
            cap = self._pos_capacity(fl, task)
            svars = self._peer_start_vars(peers)
            if fp.peer_starts != peers or fp.capacity != cap or fp.peer_start_vars != svars:
                fp.peer_starts = peers
                fp.capacity = cap
                fp.peer_start_vars = svars
                dirty.append(fp.fid)
            if g.rewire(fp, self.capacity_inputs(tid, fp)):
                dirty.append(fp.fid)

        # stand-side compatibility
        fs = g.fac.get("fSf:" + tid)
        if isinstance(fs, IncompatibilityFactor):
            want, sep = self._conflict_inputs(fl, task)
            changed = g.rewire(fs, want)
            if abs(fs.sep - sep) > 1e-9:
                fs.sep = sep
                changed = True
            if changed:
                dirty.append(fs.fid)

        # service duration
        fc = g.fac.get("fC:" + tid)
        if isinstance(fc, ServiceFactor):
            crew = scn.crews.get(task.crew) if task.crew else None
            rate = crew.rate if (crew is not None and rep.resources) else 1.0
            rate *= self._forecast_multiplier(tid, "service") if task.state != "done" else 1.0
            over = observed_overrun(self.sim, task) - task.expedite
            if abs(fc.rate - rate) > 1e-12 or abs(fc.overrun - over) > 1e-9:
                fc.rate, fc.overrun = rate, over
                dirty.append(fc.fid)

        # A task that has already started is *observed*, not predicted.  Its
        # start is pinned to the realised value and its readiness terms are
        # disconnected: leaving them wired would let a stale resource estimate
        # push the start of a service that demonstrably began on time, and that
        # error would then be carried into C_f^ready by the max operator.
        if task.state in ("running", "done"):
            fstart = g.fac.get("fS:" + tid)
            if isinstance(fstart, MaxFactor):
                if g.rewire(fstart, []):
                    dirty.append(fstart.fid)
                if abs(fstart.floor - task.start) > 1e-9:
                    fstart.floor = task.start
                    dirty.append(fstart.fid)
            if task.state == "done" and isinstance(fc, ServiceFactor):
                realised = task.finish - task.start
                if abs(fc.nominal * fc.rate + fc.overrun - realised) > 1e-6:
                    fc.overrun = realised - fc.nominal * fc.rate
                    dirty.append(fc.fid)
        return dirty

    # ------------------------------------------------------------------
    def decision_state(self) -> Dict[str, float]:
        return self.g.decision_state()

    def task_binding_class(self, tid: str) -> Tuple[str, Optional[str]]:
        """Predicted binding class of one task start, plus its binding predecessor.

        Reads the argmax of the start operator

            S_j = max(S0_j, eta_j, Rh_j, Rp_j, Sf_j)

        and reports which physical term is currently active.  When the active
        term is the precedence-only earliest start, the binding predecessor is
        returned as well so that the caller can resolve the chain recursively,
        exactly as the simulator's ground truth does.
        """
        g = self.g
        f = g.fac.get("fS:" + tid)
        if f is None or not f.active:
            return ("REL", None)
        live = sorted(v for v in f.active if v in g.var)
        if not live:
            return ("REL", None)
        # the largest active term decides; ties resolve to the richer cause
        best = max(live, key=lambda v: (g.var[v].value, v))
        if best.startswith("eta:"):
            return ("GSE", None)
        if best.startswith("Rh:"):
            return ("CREW", None)
        if best.startswith("Rp:"):
            return ("POS", None)
        if best.startswith("Sf:"):
            return ("SAFE", None)
        if best.startswith("S0:"):
            f0 = g.fac.get("fS0:" + tid)
            if f0 is not None:
                preds = sorted(v for v in f0.active if v.startswith("C:"))
                if preds:
                    return ("PRED", max(preds, key=lambda v: g.var[v].value)[2:])
            return ("REL", None)
        return ("REL", None)

    def binding_class_of(self, fid: str) -> str:
        """Predicted root binding class of a flight's readiness.

        Walks the currently binding chain from the aircraft-ready operator down
        to the first *physical* term, resolving precedence links recursively so
        that the answer names a resource class rather than merely "an upstream
        task".  This is the prediction scored against the simulator's ground
        truth, and it is the quantity an apron controller actually needs: not
        which service is late, but which resource is binding.
        """
        g = self.g
        rv = g.var.get("ready:" + fid)
        if rv is None or rv.producer is None:
            return "REL"
        fready = g.fac.get(rv.producer)
        if fready is None or not fready.active:
            return "REL"
        live = [v for v in fready.active if v.startswith("C:")]
        if not live:
            return "REL"
        tid = max(live, key=lambda v: g.var[v].value)[2:]
        seen: Set[str] = set()
        while tid and tid not in seen:
            seen.add(tid)
            cls, prev = self.task_binding_class(tid)
            if cls != "PRED":
                return cls
            if prev is None:
                return "REL"
            tid = prev
        return "PRED"
