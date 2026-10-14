"""Algorithm 2: network-aware intervention candidate generation.

When a turnaround violation is detected, the problem is not yet to select an
optimal intervention.  It is to construct a *small* set of interventions that
retain nonzero causal leverage on the violation.  Algorithm 2 therefore

    1. eliminates actions with no causal connection to the violation,
    2. identifies the physical resource responsible for it,
    3. instantiates templates only in the causally relevant graph region,
    4. quantifies expected leverage without deciding what to execute.

Three generation modes are provided so that the causal-screening mechanism can
be ablated independently of representation fidelity:

    C0  the full feasible action universe U_t, with each action's leverage
        established by counterfactual propagation (no screening at all);
    C1  structural-neighbourhood pruning only: templates are instantiated
        inside Anc_{G^struct}(z), but action sensitivity is never consulted;
    C2  full Algorithm 2: gradient screening over S_z (Phase A) followed by
        template instantiation inside the *influence* region (Phase B).

|U_t| from C0 is what the candidate reduction ratio CRR is measured against,
and C0's positive-leverage subset defines U^useful for the recall measurement.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from core.factors import (
    CapacityFactor,
    IncompatibilityFactor,
    MaxFactor,
    ServiceFactor,
    TravelFactor,
)
from core.messages import CLASS_OF_TYPE, InterventionCandidate
from core.sensitivities import influence_ancestors, structural_expansion

from .algorithm1_propagation import PropagationEngine
from .counterfactual import apply_synthetic
from .violations import Violation
from airport.observations import observed_status

EPS_LAMBDA = 0.25          # epsilon_Lambda: minimum admissible causal leverage
KAPPA = 1                  # structural expansion radius
# Ablation switch for experiments/verify_ring.py: when False, C2 instantiates
# templates on the binding frontier only, without the one-hop structural ring.
STRUCTURAL_RING = True
# Bound on the backward causal path length.  One service level costs roughly
# six hops (ready -> C -> S -> S0 -> C_pred -> S_pred -> eta_pred), so a bound
# of ten truncates the walk inside the *second* level: a violation whose
# binding cause is an upstream service would then expose no template at all,
# and Algorithm 2 would silently return an empty candidate set precisely when
# the delay is propagating rather than local.
MAX_CAUSAL_DEPTH = 40
MAX_REROUTE = 6.0          # minutes; cap on a single expedite

# Intervention cost c_u: operational disruption, not computation.
COST = {
    "REROUTE_GSE": 1.1,
    "REASSIGN_GSE": 1.9,
    "ACTIVATE_RESERVE_GSE": 2.6,
    "SWAP_GSE_SEQUENCE": 1.5,
    "REASSIGN_CREW": 2.0,
    "SWAP_CREW_SEQUENCE": 1.4,
    "CHANGE_SERVICE_POSITION": 2.2,
    "ADVANCE_POSITION_ADMISSION": 1.0,
    "RESEQUENCE_TASKS": 1.2,
}


@dataclass
class GenerationStats:
    truncated: bool = False
    universe: int = 0
    generated: int = 0
    retained: int = 0
    counterfactuals: int = 0
    cf_work: int = 0
    elapsed_ms: float = 0.0
    phase_a: int = 0
    phase_b: int = 0
    region_size: int = 0


class CandidateGenerator:
    """Algorithm 2 with pluggable screening depth."""

    def __init__(self, model, engine: PropagationEngine, mode: str = "C2",
                 budget: int = 900, eval_budget: int = 160):
        self.model = model
        self.engine = engine
        self.mode = mode
        self.budget = budget              # cap on universe enumeration
        self.eval_budget = eval_budget    # cap on counterfactual evaluations
        self.truncated = False

    # ------------------------------------------------------------------
    # feasibility  Feas(u, X_t)
    # ------------------------------------------------------------------
    def feasible(self, itype: str, ents: Sequence[str], theta: float = 0.0) -> bool:
        sim = self.model.sim
        scn = sim.scn
        if not sim.cfg.allow_time_compression and itype in ("REROUTE_GSE", "ADVANCE_POSITION_ADMISSION"):
            return False
        if not getattr(sim.cfg, "allow_resequencing", True) and itype == "RESEQUENCE_TASKS":
            return False
        try:
            if itype == "REROUTE_GSE":
                mv = sim.enroute.get(ents[0])
                return mv is not None and mv.arrive - sim.t > 1.0

            if itype in ("REASSIGN_GSE", "ACTIVATE_RESERVE_GSE"):
                task, unit = scn.tasks.get(ents[0]), scn.fleet.get(ents[1])
                if ents[0] in sim.enroute:
                    return False  # do not cancel a vehicle part-way through a road leg
                if task is None or unit is None:
                    return False
                if task.state in ("running", "done") or observed_status(sim, unit) == "down":
                    return False
                if not scn.gse_capable(unit, task) or unit.resource_id == task.gse:
                    return False
                return unit.reserve if itype == "ACTIVATE_RESERVE_GSE" else not unit.reserve

            if itype == "SWAP_GSE_SEQUENCE":
                unit = scn.fleet.get(ents[2])
                ta, tb = scn.tasks.get(ents[0]), scn.tasks.get(ents[1])
                if unit is None or ta is None or tb is None or ents[0] == ents[1]:
                    return False
                if ta.state in ("running", "done") or tb.state in ("running", "done"):
                    return False
                return ents[0] in unit.sequence and ents[1] in unit.sequence

            if itype == "REASSIGN_CREW":
                task, crew = scn.tasks.get(ents[0]), scn.crews.get(ents[1])
                if task is None or crew is None or task.state in ("running", "done"):
                    return False
                return (scn.crew_qualified(crew, task)
                        and observed_status(sim, crew) != "unavailable"
                        and crew.crew_id != task.crew)

            if itype == "SWAP_CREW_SEQUENCE":
                crew = scn.crews.get(ents[2])
                ta, tb = scn.tasks.get(ents[0]), scn.tasks.get(ents[1])
                if crew is None or ta is None or tb is None or ents[0] == ents[1]:
                    return False
                if ta.state in ("running", "done") or tb.state in ("running", "done"):
                    return False
                return ents[0] in crew.sequence and ents[1] in crew.sequence

            if itype == "CHANGE_SERVICE_POSITION":
                from airport.stands import alternative_positions
                task = scn.tasks.get(ents[0])
                if task is None or task.state in ("running", "done"):
                    return False
                fl = scn.flights[task.flight]
                # only the access zones from which this service can physically
                # be worked; a catering truck cannot serve from the fuelling
                # point, so enumerating every position would inflate |U_t|
                # with actions that are not merely useless but infeasible
                return (ents[1] in alternative_positions(task.code)
                        and ents[1] in scn.stands[fl.stand].positions
                        and ents[1] != task.position)

            if itype == "ADVANCE_POSITION_ADMISSION":
                task = scn.tasks.get(ents[0])
                if task is None or task.state in ("running", "done"):
                    return False
                fl = scn.flights[task.flight]
                p = scn.stands[fl.stand].positions.get(task.position)
                return p is not None and len(p.occupants) >= p.capacity

            if itype == "RESEQUENCE_TASKS":
                ta, tb = scn.tasks.get(ents[0]), scn.tasks.get(ents[1])
                if ta is None or tb is None or ta.flight != tb.flight:
                    return False
                if ta.state in ("running", "done") or tb.state in ("running", "done"):
                    return False
                if ents[0] in tb.holds or sim._reaches(ents[0], ents[1]):
                    return False
                fl = scn.flights[ta.flight]
                return any(set((c[0], c[1])) == set((ents[0], ents[1]))
                           or ta.position == tb.position for c in fl.conflicts) \
                    or ta.position == tb.position
        except (KeyError, IndexError):
            return False
        return False

    def domain(self, itype: str, ents: Sequence[str]) -> Tuple[float, float]:
        """Feasible parameter domain D_u for a continuous intervention."""
        sim = self.model.sim
        if itype == "REROUTE_GSE":
            mv = sim.enroute.get(ents[0])
            if mv is None:
                return (0.0, 0.0)
            rem = max(0.0, mv.arrive - sim.t)
            return (-min(MAX_REROUTE, 0.55 * rem), 0.0)
        return (0.0, 0.0)

    # ------------------------------------------------------------------
    # full feasible action universe U_t
    # ------------------------------------------------------------------
    def enumerate_universe(self) -> List[Tuple[str, Tuple[str, ...], float]]:
        sim = self.model.sim
        scn = sim.scn
        U: List[Tuple[str, Tuple[str, ...], float]] = []
        for tid in sorted(self.model.tracked_tasks):
            task = scn.tasks.get(tid)
            if task is None or task.state in ("running", "done"):
                continue
            if tid in sim.enroute:
                lo, _ = self.domain("REROUTE_GSE", (tid,))
                if lo < -1e-6:
                    U.append(("REROUTE_GSE", (tid,), -lo))
            if task.gse_class is not None:
                for unit in scn.fleet.values():
                    if not scn.gse_capable(unit, task) or unit.resource_id == task.gse:
                        continue
                    if observed_status(sim, unit) == "down":
                        continue
                    kind = "ACTIVATE_RESERVE_GSE" if unit.reserve else "REASSIGN_GSE"
                    U.append((kind, (tid, unit.resource_id), 0.0))
                unit = scn.fleet.get(task.gse)
                if unit is not None:
                    for other in unit.sequence:
                        if other != tid:
                            U.append(("SWAP_GSE_SEQUENCE",
                                      (tid, other, unit.resource_id), 0.0))
            if task.crew_skill is not None:
                for crew in scn.crews.values():
                    if not scn.crew_qualified(crew, task) or crew.crew_id == task.crew:
                        continue
                    U.append(("REASSIGN_CREW", (tid, crew.crew_id), 0.0))
                crew = scn.crews.get(task.crew)
                if crew is not None:
                    for other in crew.sequence:
                        if other != tid:
                            U.append(("SWAP_CREW_SEQUENCE",
                                      (tid, other, crew.crew_id), 0.0))
            fl = scn.flights[task.flight]
            from airport.stands import alternative_positions
            for pos in alternative_positions(task.code):
                if pos in scn.stands[fl.stand].positions and pos != task.position:
                    U.append(("CHANGE_SERVICE_POSITION", (tid, pos), 0.0))
            U.append(("ADVANCE_POSITION_ADMISSION", (tid,), 0.0))
            for other in sorted(fl.tasks):
                if other != tid:
                    U.append(("RESEQUENCE_TASKS", (tid, other), 0.0))
            if len(U) > self.budget:
                break
        return U[: self.budget]

    # ------------------------------------------------------------------
    # causal backward region  B_t(z)
    # ------------------------------------------------------------------
    def backward_region(self, zid: str, use_influence: bool) -> Set[str]:
        return influence_ancestors(self.model.g, zid, use_influence,
                                   MAX_CAUSAL_DEPTH)

    def binding_frontier(self, region: Set[str]) -> Set[str]:
        """G^bind restricted to the region.

        Instantiating templates only here is what separates Algorithm 2 from
        structural enumeration: a service that is not its flight's straggler,
        or a position that is not saturated, exposes no template even though it
        is structurally adjacent to the violation.
        """
        g = self.model.g
        out: Set[str] = set()
        for vid in region:
            v = g.var.get(vid)
            if v is None:
                continue
            for fid in v.consumers:
                f = g.fac.get(fid)
                if f is not None and vid in f.active:
                    out.add(vid)
                    break
        return out

    # ------------------------------------------------------------------
    # templates
    # ------------------------------------------------------------------
    def instantiate(self, seeds: Set[str]) -> List[Tuple[str, Tuple[str, ...], float]]:
        """Instantiate the template registry at a given set of seed variables."""
        g = self.model.g
        sim = self.model.sim
        scn = sim.scn
        out: List[Tuple[str, Tuple[str, ...], float]] = []
        producers: Set[str] = set()
        for vid in sorted(seeds):
            v = g.var.get(vid)
            if v is not None and v.producer:
                producers.add(v.producer)

        for fid in sorted(producers):
            f = g.fac.get(fid)
            if f is None:
                continue
            tid = fid.split(":", 1)[1] if ":" in fid else ""
            task = scn.tasks.get(tid)
            if task is None:
                continue
            fl = scn.flights.get(task.flight)
            if fl is None:
                continue

            if isinstance(f, TravelFactor):
                # the binding cause is the vehicle's arrival
                if tid in sim.enroute:
                    lo, _ = self.domain("REROUTE_GSE", (tid,))
                    if lo < -1e-6:
                        out.append(("REROUTE_GSE", (tid,), -lo))
                for unit in sorted(scn.fleet.values(), key=lambda u: u.resource_id):
                    if not scn.gse_capable(unit, task) or unit.resource_id == task.gse:
                        continue
                    if observed_status(sim, unit) == "down":
                        continue
                    kind = "ACTIVATE_RESERVE_GSE" if unit.reserve else "REASSIGN_GSE"
                    out.append((kind, (tid, unit.resource_id), 0.0))
                unit = scn.fleet.get(task.gse)
                if unit is not None:
                    for other in unit.sequence:
                        if other != tid and other in scn.tasks:
                            out.append(("SWAP_GSE_SEQUENCE",
                                        (tid, other, unit.resource_id), 0.0))

            elif fid.startswith("fRh:"):
                for crew in sorted(scn.crews.values(), key=lambda c: c.crew_id):
                    if not scn.crew_qualified(crew, task) or crew.crew_id == task.crew:
                        continue
                    out.append(("REASSIGN_CREW", (tid, crew.crew_id), 0.0))
                crew = scn.crews.get(task.crew)
                if crew is not None:
                    for other in crew.sequence:
                        if other != tid and other in scn.tasks:
                            out.append(("SWAP_CREW_SEQUENCE",
                                        (tid, other, crew.crew_id), 0.0))

            elif isinstance(f, CapacityFactor):
                out.append(("ADVANCE_POSITION_ADMISSION", (tid,), 0.0))
                from airport.stands import alternative_positions
                for pos in alternative_positions(task.code):
                    if pos in scn.stands[fl.stand].positions and pos != task.position:
                        out.append(("CHANGE_SERVICE_POSITION", (tid, pos), 0.0))

            elif isinstance(f, IncompatibilityFactor):
                for vid in f.active:
                    if vid.startswith("C:"):
                        other = vid[2:]
                        out.append(("RESEQUENCE_TASKS", (tid, other), 0.0))

            elif isinstance(f, ServiceFactor) or isinstance(f, MaxFactor):
                # a binding upstream task: act on the predecessor's own resource
                for p in tuple(task.predecessors):
                    pt = scn.tasks.get(p)
                    if pt is None or pt.state in ("running", "done"):
                        continue
                    if p in sim.enroute:
                        lo, _ = self.domain("REROUTE_GSE", (p,))
                        if lo < -1e-6:
                            out.append(("REROUTE_GSE", (p,), -lo))
        return list(dict.fromkeys(out))

    # ------------------------------------------------------------------
    # leverage evaluation
    # ------------------------------------------------------------------
    def counterfactual_leverage(self, itype: str, ents: Sequence[str], theta: float,
                                targets: Sequence[Violation]
                                ) -> Tuple[Dict[str, float], Dict[str, float], int]:
        """Lambda_z^Delta(u) via localized counterfactual propagation."""
        model = self.model
        g = model.g
        # value-only containment is exact for a leverage measurement, which
        # reads state deltas rather than the maintained sensitivity stream
        eng = PropagationEngine(model, mode="A1")
        eng.epoch = self.engine.epoch
        before = dict((z.zid, g.var[z.zid].value) for z in targets if z.zid in g.var)
        g.start_journal()
        try:
            seeds = apply_synthetic(model, itype, ents, theta)
            if not seeds:
                return {}, {}, 0
            if g.ensure_scc().n_cyclic:
                return {}, {}, 0
            res = eng.propagate(seeds, collect_messages=False)
            lev: Dict[str, float] = {}
            for z in targets:
                v = g.var.get(z.zid)
                if v is None or z.zid not in before:
                    continue
                lev[z.zid] = z.direction * (v.value - before[z.zid])
            side: Dict[str, float] = {}
            for vid, (old, new) in res.changed_decision.items():
                if vid not in lev and vid.startswith("L:"):
                    side[vid] = new - old
            work = res.touched
        finally:
            g.rollback()
        return lev, side, work

    # ------------------------------------------------------------------
    # search region
    # ------------------------------------------------------------------
    def proposals(self, violations: Sequence[Violation],
                  stats: Optional[GenerationStats] = None,
                  universe: Optional[List[Tuple[str, Tuple[str, ...], float]]] = None
                  ) -> List[Tuple[str, Tuple[str, ...], float]]:
        """Template instances in the search region of the mode, in pricing order."""
        if self.mode == "C0":
            return list(universe if universe is not None else self.enumerate_universe())
        proposals: List[Tuple[str, Tuple[str, ...], float]] = []
        use_inf = (self.mode == "C2" and self.model.rep.influence)
        for z in violations:
            core = self.backward_region(z.zid, use_influence=use_inf)
            if not use_inf:
                # C1: no influence information at all, so every structural
                # ancestor is a candidate site.
                seeds = structural_expansion(self.model.g, core, KAPPA)
            else:
                # C2: the influence region tells us where causal leverage
                # currently flows, and its binding frontier is where a
                # *continuous* action can act.  Structural interventions,
                # however, can create a causal path that is absent from the
                # current Jacobian -- reassigning a unit, swapping a service
                # order -- so the frontier is widened by one structural hop
                # and templates are instantiated there as well.  Discarding
                # that expansion would prune exactly the actions whose whole
                # purpose is to change the active structure.
                frontier = self.binding_frontier(core)
                ring = (structural_expansion(self.model.g, core, KAPPA) - core
                        if STRUCTURAL_RING else set())
                seeds = frontier | ring
            if stats is not None:
                stats.region_size = max(stats.region_size, len(seeds))
            proposals += self.instantiate(seeds)
        return list(dict.fromkeys(proposals))

    # ------------------------------------------------------------------
    # main entry
    # ------------------------------------------------------------------
    def generate(self, violations: Sequence[Violation],
                 want_universe: bool = False
                 ) -> Tuple[List[InterventionCandidate], GenerationStats]:
        t0 = time.perf_counter()
        stats = GenerationStats()
        cands: Dict[Tuple[str, ...], InterventionCandidate] = {}
        if not violations:
            stats.elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return [], stats

        universe = (self.enumerate_universe()
                    if (self.mode == "C0" or want_universe) else [])
        stats.universe = len(universe)
        proposals = self.proposals(violations, stats, universe)

        # ---- Phase A: current-sensitivity screening (C2 only) ------------
        if self.mode == "C2" and self.model.rep.influence:
            for z in violations:
                S_z = self.engine.sensitivities_of(z.zid)
                for handle, s in sorted(S_z.items()):
                    if abs(s) < 1e-9 or not handle.startswith("eta:"):
                        continue
                    tid = handle.split(":", 1)[1]
                    if not self.feasible("REROUTE_GSE", (tid,)):
                        continue
                    lo, hi = self.domain("REROUTE_GSE", (tid,))
                    lam = max(z.direction * s * lo, z.direction * s * hi)
                    if lam <= EPS_LAMBDA:
                        continue
                    stats.phase_a += 1
                    key = ("REROUTE_GSE", tid)
                    task = self.model.sim.scn.tasks.get(tid)
                    prev = cands.get(key)
                    lev = dict(prev.predicted_leverage) if prev else {}
                    lev[z.zid] = max(lev.get(z.zid, 0.0), lam)
                    cands[key] = InterventionCandidate(
                        candidate_id="u%d" % len(cands),
                        candidate_type="REROUTE_GSE",
                        affected_flights=(task.flight,) if task else (),
                        affected_tasks=(tid,),
                        affected_resources=(tid,),
                        target_violation=tuple(sorted(lev)),
                        predicted_leverage=lev,
                        causal_path=(handle, z.zid),
                        side_effects={},
                        resource_requirements=self._resources("REROUTE_GSE", (tid,), task),
                        feasibility=True,
                        uncertainty=dict((k, 0.25 * v) for k, v in lev.items()),
                        validity_epoch=self.engine.epoch,
                        theta=-lo, cost=COST["REROUTE_GSE"],
                        bottleneck_class="gse",
                        sufficient=lam >= z.residual,
                        leverage_fraction=min(1.0, lam / max(1e-9, z.residual)),
                        origin="gradient")

        # ---- Phase B: structural / discrete interventions ----------------
        seen: Set[Tuple[str, ...]] = set()
        self.truncated = False
        zres = dict((z.zid, z.residual) for z in violations)
        for (itype, ents, theta) in proposals:
            if stats.counterfactuals >= self.eval_budget:
                # A real-time controller cannot spend unbounded time
                # establishing leverage coefficients.  The budget applies
                # uniformly to every mode; C0 exhausts it far more often than
                # C2, and the exhaustion rate is itself reported.
                self.truncated = True
                break
            key = (itype,) + tuple(ents)
            if key in cands or key in seen:
                continue
            seen.add(key)
            stats.generated += 1
            if not self.feasible(itype, ents, theta):
                continue
            lev, side, work = self.counterfactual_leverage(itype, ents, theta,
                                                           violations)
            stats.counterfactuals += 1
            stats.cf_work += work
            stats.phase_b += 1
            if self.mode == "C0":
                # No causal screening at all: the full feasible action universe
                # is handed to the optimiser, including actions of zero or
                # negative leverage.  Establishing each coefficient still costs
                # one counterfactual propagation -- exactly the cost Algorithm 2
                # avoids.
                tv = tuple(sorted(lev))
                if not tv:
                    continue
            else:
                best = max(lev.values()) if lev else 0.0
                if best <= EPS_LAMBDA:
                    continue
                tv = tuple(sorted(z for z in lev if lev[z] > EPS_LAMBDA))
                if not tv:
                    continue
            top = max(tv, key=lambda z: lev[z])
            task = self.model.sim.scn.tasks.get(ents[0])
            cands[key] = InterventionCandidate(
                candidate_id="u%d" % len(cands),
                candidate_type=itype,
                affected_flights=self._flights(ents),
                affected_tasks=tuple(e for e in ents if e in self.model.sim.scn.tasks),
                affected_resources=tuple(ents),
                target_violation=tv,
                predicted_leverage=dict((z, lev[z]) for z in tv),
                causal_path=(itype,) + tuple(ents),
                side_effects=side,
                resource_requirements=self._resources(itype, ents, task),
                feasibility=True,
                uncertainty=dict((z, 0.2 * abs(lev[z])) for z in tv),
                validity_epoch=self.engine.epoch,
                theta=theta, cost=COST.get(itype, 1.0),
                bottleneck_class=CLASS_OF_TYPE.get(itype, "mixed"),
                sufficient=lev[top] >= zres.get(top, 0.0),
                leverage_fraction=min(1.0, lev[top] / max(1e-9, zres.get(top, 1.0))),
                origin="structural",
                harm=self._harm(side))

        out = list(cands.values())
        stats.retained = len(out)
        stats.truncated = self.truncated
        if not stats.universe:
            stats.universe = max(len(out), stats.generated)
        stats.elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return out, stats

    # ------------------------------------------------------------------
    def _flights(self, ents: Sequence[str]) -> Tuple[str, ...]:
        scn = self.model.sim.scn
        out = set()
        for e in ents:
            t = scn.tasks.get(e)
            if t is not None:
                out.add(t.flight)
        return tuple(sorted(out))

    def _harm(self, side: Dict[str, float]) -> float:
        """Priority-weighted damage a candidate inflicts on other flights.

        Algorithm 2 records Delta-Xi^side without scalarising it; this scalar is
        supplied *in addition* so that Algorithm 3 can price the trade-off
        without re-deriving it.
        """
        scn = self.model.sim.scn
        tot = 0.0
        for vid, d in side.items():
            if d >= -1e-9:
                continue
            fid = vid.split(":", 1)[1]
            fl = scn.flights.get(fid)
            tot += (fl.weight if fl else 1.0) * (-d)
        return tot

    @staticmethod
    def _resources(itype: str, ents: Sequence[str], task) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if ents:
            out["task:" + ents[0]] = 1.0
        if task is not None:
            out["flight:" + task.flight] = 1.0
        if itype in ("REASSIGN_GSE", "ACTIVATE_RESERVE_GSE") and len(ents) > 1:
            out["gse:" + ents[1]] = 1.0
        elif itype == "SWAP_GSE_SEQUENCE" and len(ents) > 2:
            out["gse:" + ents[2]] = 1.0
            out["task:" + ents[1]] = 1.0
        elif itype == "REASSIGN_CREW" and len(ents) > 1:
            out["crew:" + ents[1]] = 1.0
        elif itype == "SWAP_CREW_SEQUENCE" and len(ents) > 2:
            out["crew:" + ents[2]] = 1.0
            out["task:" + ents[1]] = 1.0
        elif itype == "CHANGE_SERVICE_POSITION" and len(ents) > 1:
            out["position:" + ents[1]] = 1.0
        elif itype == "RESEQUENCE_TASKS" and len(ents) > 1:
            out["task:" + ents[1]] = 1.0
        return out
