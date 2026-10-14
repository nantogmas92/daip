"""The closed loop.

    X_t --R_rho--> G_t^rho --A1--> Xi_t^rho --A2--> U_t^cand --A3--> U^proposal
        --A1^cf--> U_t^star --F--> X_{t+1}

Every stage is timed separately so that the end-to-end decision latency

    T_dec = T_A1 + T_A2 + T_A3 + T_validation

can be decomposed.  The controller owns no physical state: it observes events
from the simulator and returns validated interventions through the simulator's
single ``execute`` entry point.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from airport.adapter import LADDER, Representation, TurnaroundModel
from core.messages import DecisionRecord

from .algorithm1_propagation import PropagationEngine
from .algorithm2_candidates import EPS_LAMBDA, CandidateGenerator
from .algorithm3_selection import InterventionSelector
from .interaction import InteractionBuilder
from .validation import CounterfactualValidator
from .violations import ViolationDetector

J_MAX = 2                      # Algorithm 3 refinement rounds
BOTTLENECK_CLASS = {"GSE": "gse", "CREW": "crew", "POS": "stand",
                    "SAFE": "scheduling", "PRED": "scheduling", "REL": "scheduling"}


@dataclass
class ControllerConfig:
    representation: str = "R6"
    propagation: str = "A2"        # A0 | AS | A1 | A2
    candidates: str = "C2"         # C0 | C1 | C2
    validation: str = "V1"         # V0 | V1
    backend: str = "cpsat"         # cpsat | milp | greedy
    budget: int = 4
    solver_time: float = 0.6
    intervene: bool = True
    measure_universe: bool = False   # enumerate U_t every epoch (for CRR)
    recall_every: int = 0            # 0 = never; else every n-th decision epoch
    robust_beta: float = 0.0
    max_targets: int = 3             # violations forwarded to Algorithm 2
    eval_budget: int = 160           # counterfactual evaluations per epoch
    replan_period: float = 6.0       # forced re-plan interval (minutes)
    urgency_only: bool = False       # B2 baseline: task-urgency heuristic


class ClosedLoopController:
    def __init__(self, sim, cfg: ControllerConfig, experiment_id: str = "exp"):
        self.sim = sim
        self.cfg = cfg
        self.experiment_id = experiment_id
        # optional full-stream logging (experiments/trace.py); None in the
        # campaign so that tracing cannot perturb latency measurements
        self.trace = None
        rep: Representation = LADDER[cfg.representation]
        self.model = TurnaroundModel(sim, rep)
        self.engine = PropagationEngine(self.model, mode=cfg.propagation)
        self.detector = ViolationDetector(self.model)
        self.generator = CandidateGenerator(self.model, self.engine,
                                            mode=cfg.candidates,
                                            eval_budget=cfg.eval_budget)
        self.interaction = InteractionBuilder(self.model, self.engine)
        self.selector = InterventionSelector(cfg.backend, cfg.budget,
                                             cfg.solver_time, cfg.robust_beta)
        self.validator = CounterfactualValidator(self.model, cfg.validation)
        self.records: List[DecisionRecord] = []
        self.epoch = 0
        self.predicted_ready: Dict[str, float] = {}
        self.predicted_binding: Dict[str, str] = {}
        # (time, class) trace so that the model's prediction can be compared
        # with the simulator's ground truth *at the same instant*
        self.binding_trace: Dict[str, List[Tuple[float, str]]] = {}
        self.ready_trace: Dict[str, List[Tuple[float, float]]] = {}
        # Per-task predicted binding class, frozen when the task starts.  The
        # simulator freezes its ground-truth cause at exactly the same instant,
        # so the two are compared like with like.
        self.task_binding: Dict[str, Tuple[str, Optional[str]]] = {}
        self.pred_critical: Dict[str, str] = {}
        self.pending_recovery: List[Tuple[float, float]] = []
        self._last_signature: Optional[tuple] = None
        self._last_replan = -1e9

        self.model.sync(sim.t)
        self.model.refresh(sim.t)
        self.engine.initial_solve()

    # ------------------------------------------------------------------
    def run(self) -> Dict[str, float]:
        sim = self.sim
        guard = 0
        while not sim.finished() and guard < 400000:
            guard += 1
            ev = sim.step()
            if ev is None:
                continue
            self.epoch += 1
            self._decide(ev)
        return self.summary()

    # ------------------------------------------------------------------
    def _decide(self, ev) -> None:
        started = time.perf_counter()
        before = len(self.records)
        try:
            self._decide_impl(ev)
        finally:
            if len(self.records) > before:
                self.records[-1].t_total_ms = (time.perf_counter()-started)*1000.

    def _decide_impl(self, ev) -> None:
        sim = self.sim
        cfg = self.cfg
        rec = DecisionRecord(
            experiment_id=self.experiment_id, seed=sim.cfg.seed, epoch=self.epoch,
            simulation_time=sim.t, representation=cfg.representation,
            propagation_mode=cfg.propagation, candidate_mode=cfg.candidates,
            validation_mode=cfg.validation, backend=cfg.backend,
            regime=sim.cfg.regime, n_flights_active=len(sim.active),
            n_stands=len(sim.scn.stands), n_gse=len(sim.scn.fleet),
            disturbance_class=(sim.last_disturbance or "none")
            if ev.event_type in ("GSE_TRAVEL_DELAY", "GSE_BREAKDOWN",
                                 "CREW_UNAVAILABLE", "SERVICE_OVERRUN",
                                 "APRON_CONGESTION", "STAND_RELEASE_DELAY",
                                 "ARRIVAL_DELAY") else "none")

        # ---- representation maintenance + Algorithm 1 -------------------
        t0 = time.perf_counter()
        new_f = self.model.sync(sim.t)
        dirty = self.model.refresh(sim.t)
        seeds = list(dict.fromkeys(list(new_f) + list(dirty)))
        res = self.engine.propagate(seeds) if seeds else None
        rec.t_prop_ms = (time.perf_counter() - t0) * 1000.0
        rec.factors_total = len(self.model.g.fac)
        rec.factors_touched = res.touched if res else 0
        rec.messages = len(res.messages) if res else 0
        rec.criticality_switches = res.criticality_switches if res else 0
        rec.propagation_depth = res.depth if res else 0
        rec.sccs_resolved = res.sccs_resolved if res else 0
        if self.trace is not None and res is not None:
            self.trace.on_messages(self.epoch, sim.t, res.messages)

        self._snapshot_predictions()

        # ---- violation detection ----------------------------------------
        violations = self.detector.detect(sim.t)
        rec.n_violations = len(violations)
        rec.residual_pre = sum(z.weight * z.residual for z in violations)
        if rec.disturbance_class != "none" and rec.residual_pre > 0:
            self.pending_recovery.append((sim.t, rec.residual_pre))
        self._check_recovery(rec.residual_pre)

        if not violations or not cfg.intervene:
            self.records.append(rec)
            return

        # ---- replanning gate ---------------------------------------------
        # Algorithms 2-3 run only when the decision-relevant structure changed
        # -- the set of violated targets together with the binding cause class
        # of each -- or when the forced re-plan period expires.  Re-deriving an
        # identical candidate set at every task completion is both wasteful and
        # unlike an event-driven apron controller.
        sig = tuple(sorted((z.zid, self._binding_class(z.entity))
                           for z in violations))
        if (sig == self._last_signature
                and sim.t - self._last_replan < cfg.replan_period):
            self.records.append(rec)
            return
        self._last_signature = sig
        self._last_replan = sim.t

        targets = violations[: cfg.max_targets]

        # ---- Algorithm 2 --------------------------------------------------
        want_u = cfg.measure_universe or (
            cfg.recall_every and self.epoch % cfg.recall_every == 0)
        cands, stats = self.generator.generate(targets, want_universe=want_u)
        rec.t_cand_ms = stats.elapsed_ms
        rec.universe_size = stats.universe
        rec.candidate_size = len(cands)
        rec.budget_exhausted = bool(stats.truncated)
        if self.trace is not None:
            self.trace.on_candidates(self.epoch, sim.t, cands, targets, stats)
        for u in cands:
            if u.bottleneck_class == "gse":
                rec.n_gse_cand += 1
            elif u.bottleneck_class == "crew":
                rec.n_crew_cand += 1
            elif u.bottleneck_class == "stand":
                rec.n_stand_cand += 1
            else:
                rec.n_sched_cand += 1

        if cfg.recall_every and self.epoch % cfg.recall_every == 0:
            self._measure_recall(rec, cands, targets)

        if cfg.urgency_only:
            cands = self._urgency_filter(cands, targets)
            rec.candidate_size = len(cands)

        if not cands:
            self.records.append(rec)
            return

        # ---- Algorithm 3 + counterfactual validation loop ------------------
        G = self.interaction.build(cands, targets)
        rec.t_interaction_ms = G.elapsed_ms
        rec.interaction_edges = G.n_edges
        nogoods: List[Tuple[str, ...]] = []
        accepted = None
        t_sel = t_val = 0.0
        for j in range(J_MAX):
            sel = self.selector.select(cands, G, targets, nogoods)
            t_sel += sel.elapsed_ms
            rec.solver_status = sel.status
            if not sel.chosen:
                break
            out = self.validator.validate(sel.chosen, sel.predicted, targets,
                                          bundle_id="b%d_%d" % (self.epoch, j),
                                          epoch=self.epoch)
            t_val += out.elapsed_ms
            if self.trace is not None:
                self.trace.on_solver(self.epoch, sim.t, j, sel, out)
            if out.accepted or self.cfg.validation == "V0":
                accepted = out
                break
            if out.bundle is not None:
                nogoods.append(out.bundle.candidate_ids)
                rec.new_violations += len(out.bundle.new_violations)
        rec.t_select_ms = t_sel
        rec.t_validate_ms = t_val

        if accepted is None or accepted.bundle is None:
            self.records.append(rec)
            return

        bundle = accepted.bundle
        rec.bundle_size = len(bundle.candidate_ids)
        rec.bundle_model_error = bundle.model_error
        keep = set(bundle.candidate_ids)
        chosen = [u for u in cands if u.candidate_id in keep]

        # ---- execution ----------------------------------------------------
        applied = 0
        for u in chosen:
            ok = self.sim.execute(u.candidate_type, u.affected_resources, u.theta)
            if ok:
                applied += 1
            if self.trace is not None:
                self.trace.on_intervention(self.epoch, sim.t, u, ok)
        if applied:
            d2 = self.model.refresh(sim.t)
            if d2:
                self.engine.propagate(d2)
        post = self.detector.detect(sim.t)
        rec.residual_post = sum(z.weight * z.residual for z in post)
        rec.repair_fraction = (1.0 - rec.residual_post / rec.residual_pre
                               if rec.residual_pre > 0 else 0.0)
        pre_ids = set(z.zid for z in violations)
        fresh = [z for z in post if z.zid not in pre_ids]
        rec.new_violations += len(fresh)
        self.sim.metrics.new_violations += len(fresh)
        self.records.append(rec)

    # ------------------------------------------------------------------
    def _urgency_filter(self, cands, targets):
        """B2: act on the most delayed flight without causal information.

        Keeps only interventions that touch the lowest-slack flight, which is
        what a task-urgency heuristic does: it identifies *who* is late, never
        *what is binding*.
        """
        if not targets:
            return []
        worst = max(targets, key=lambda z: z.weight * z.residual).entity
        return [u for u in cands if worst in u.affected_flights]

    def _binding_class(self, fid: str) -> str:
        if not self.model.rep.dag:
            return "-"
        return self.model.binding_class_of(fid)

    def _snapshot_predictions(self) -> None:
        """Record the E_rep observations: predicted readiness and binding class.

        Both are *traced* rather than merely overwritten, because the
        simulator's ground-truth binding cause is established at the instant
        the readiness-determining task starts.  Scoring the model against it
        requires the prediction that stood at that same instant; comparing a
        post-hoc prediction against it would flatter the model.
        """
        g = self.model.g
        now = self.sim.t
        scn = self.sim.scn
        if self.model.rep.dag:
            for fid in self.model.tracked_flights:
                rv = g.var.get("ready:" + fid)
                if rv is None:
                    continue
                self.predicted_ready[fid] = rv.value
                self.ready_trace.setdefault(fid, []).append((now, rv.value))
                # A task that has started is pinned to its observed values, so
                # the model is no longer *predicting* its binding cause.  The
                # attribution trace therefore stops once every departure-
                # enabling service has begun: the last entry is the prediction
                # that stood while the answer was still open.
                fl = scn.flights.get(fid)
                if fl is None:
                    continue
                fready = g.fac.get("fready:" + fid)
                if fready is not None and fready.active:
                    live = [v for v in fready.active if v.startswith("C:")]
                    if live:
                        self.pred_critical[fid] = max(
                            live, key=lambda v: g.var[v].value)[2:]
                if any(fl.tasks[t].state not in ("running", "done")
                       for t in fl.terminal if t in fl.tasks):
                    cls = self._predicted_root_cause(fid)
                    self.predicted_binding[fid] = cls
                    self.binding_trace.setdefault(fid, []).append((now, cls))

            # freeze the per-task attribution of every task that has not started
            for tid in self.model.tracked_tasks:
                tk = scn.tasks.get(tid)
                if tk is None or tk.state in ("running", "done"):
                    continue
                self.task_binding[tid] = self.model.task_binding_class(tid)
        else:
            # R0 has no aircraft-ready node and no dependency graph: the best a
            # task-centric model can offer is the latest predicted completion
            # among the flight's own services.
            latest: Dict[str, float] = {}
            for tid in self.model.tracked_tasks:
                cv = g.var.get("C:" + tid)
                if cv is None:
                    continue
                fid = tid.split("/", 1)[0]
                latest[fid] = max(latest.get(fid, -1e18), cv.value)
            for fid, val in latest.items():
                self.predicted_ready[fid] = val
                self.ready_trace.setdefault(fid, []).append((now, val))

    def _predicted_root_cause(self, fid: str) -> str:
        """Recursive predicted attribution over the *frozen* per-task record.

        The live graph cannot answer this once upstream services have started:
        an observed task is pinned to its realised values, so its start
        operator no longer carries a prediction and the walk would bottom out
        at the release term.  Walking the frozen record instead mirrors the
        simulator's ground truth, which is likewise recorded at task start.
        """
        tid = self.pred_critical.get(fid)
        if tid is None:
            return "REL"
        seen = set()
        while tid and tid not in seen:
            seen.add(tid)
            cls, prev = self.task_binding.get(tid, (None, None))
            if cls is None:
                cls, prev = self.model.task_binding_class(tid)
            if cls != "PRED":
                return cls
            if prev is None:
                return "REL"
            tid = prev
        return "PRED"

    @staticmethod
    def _trace_at(trace, when: float):
        """The last traced value that stood at or before ``when``."""
        best = None
        for (t, v) in trace:
            if t <= when + 1e-9:
                best = v
            else:
                break
        return best if best is not None else (trace[0][1] if trace else None)

    def _critical_start(self, fl) -> float:
        """Start time of the task that actually determined C_f^ready."""
        best, ts = -1e18, fl.release
        for tid in fl.terminal:
            t = self.sim.scn.tasks.get(tid)
            if t is None or t.state != "done" or t.finish != t.finish:
                continue
            if t.finish > best:
                best, ts = t.finish, t.start
        return ts

    def _check_recovery(self, residual: float) -> None:
        if not self.pending_recovery:
            return
        keep = []
        for (t0, r0) in self.pending_recovery:
            if residual <= 1e-6 or residual < 0.2 * r0:
                self.sim.metrics.recovery_times.append(self.sim.t - t0)
            elif self.sim.t - t0 < 90.0:
                keep.append((t0, r0))
            else:
                self.sim.metrics.recovery_times.append(90.0)
        self.pending_recovery = keep

    # ------------------------------------------------------------------
    def _measure_recall(self, rec, cands, targets) -> None:
        """Compare Algorithm 2's set with the positive-leverage subset of U_t."""
        gen0 = CandidateGenerator(self.model, self.engine, mode="C0",
                                  budget=self.generator.budget,
                                  eval_budget=10 ** 6)
        universe = gen0.enumerate_universe()
        rec.universe_size = max(rec.universe_size, len(universe))
        useful = set()
        for (itype, ents, theta) in universe:
            if not gen0.feasible(itype, ents, theta):
                continue
            lev, _s, _w = gen0.counterfactual_leverage(itype, ents, theta, targets)
            if lev and max(lev.values()) > EPS_LAMBDA:
                useful.add((itype,) + tuple(ents))
        got = set((u.candidate_type,) + tuple(u.affected_resources) for u in cands)
        if useful:
            rec.recall_cand = len(got & useful) / len(useful)
        if got:
            rec.precision_cand = len(got & useful) / len(got)

    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, float]:
        sim = self.sim
        m = sim.metrics
        scn = sim.scn
        from airport.crews import utilisation_of as crew_util, workload_fairness
        from airport.gse import utilisation_of as gse_util

        done = [f for f in scn.flights.values() if not math.isnan(f.ready)]
        errs: List[float] = []
        bl_ok = bl_n = 0
        for f in done:
            ts = self._critical_start(f)
            rt = self.ready_trace.get(f.flight_id)
            if rt:
                p = self._trace_at(rt, ts)
                if p is not None and p > -1e17:
                    errs.append(abs(p - f.ready))
            truth = sim.flight_cause.get(f.flight_id)
            bt = self.binding_trace.get(f.flight_id)
            pred = bt[-1][1] if bt else None
            if truth and pred:
                bl_n += 1
                bl_ok += int(truth == pred)

        recs = self.records

        def _mean(fn):
            vals = [fn(r) for r in recs]
            vals = [v for v in vals if v == v]
            return sum(vals) / len(vals) if vals else float("nan")

        delays = sorted(m.delays)
        p95 = delays[min(len(delays) - 1, int(0.95 * len(delays)))] if delays else 0.0
        horizon = max(1.0, sim.t)
        gu = gse_util(scn.fleet, horizon)
        cu = crew_util(scn.crews, horizon)

        return {
            "ready_flights": m.ready_flights,
            "n_flights": len(scn.flights),
            "completion_rate": m.ready_flights / max(1, len(scn.flights)),
            "delay_w": m.delay_w,
            "delay_w_per_flight": m.delay_w / max(1, len(scn.flights)) if m.ready_flights==len(scn.flights) else float("nan"),
            "delay_mean": m.delay / max(1, m.ready_flights),
            "delay_p95": p95,
            "on_time_rate": m.on_time / max(1, len(scn.flights)),
            "gse_distance": m.gse_distance,
            "gse_deadhead": m.gse_deadhead,
            "gse_utilisation": sum(gu.values()) / max(1, len(gu)),
            "crew_utilisation": sum(cu.values()) / max(1, len(cu)),
            "crew_wait": m.crew_wait / max(1, m.ready_flights),
            "workload_fairness": workload_fairness(scn.crews),
            "stand_release_delay": m.stand_release_delay / max(1, m.ready_flights),
            "congestion_delay": m.congestion_delay,
            "interventions": m.interventions,
            "new_violations": m.new_violations,
            "recovery_time": (sum(m.recovery_times) / len(m.recovery_times)
                              if m.recovery_times else float("nan")),
            "e_rep_ready": sum(errs) / len(errs) if errs else float("nan"),
            "binding_accuracy": bl_ok / bl_n if bl_n else float("nan"),
            "n_decisions": len(recs),
            "mean_pf": _mean(lambda r: r.factors_touched / r.factors_total
                             if r.factors_total else float("nan")),
            "mean_factors": _mean(lambda r: float(r.factors_total)),
            "mean_messages": _mean(lambda r: float(r.messages)),
            "mean_depth": _mean(lambda r: float(r.propagation_depth)),
            "mean_t_dec": _mean(lambda r: r.t_total_ms or (r.t_prop_ms + r.t_cand_ms
                                + r.t_select_ms + r.t_validate_ms + r.t_interaction_ms)),
            "mean_t_prop": _mean(lambda r: r.t_prop_ms),
            "mean_t_cand": _mean(lambda r: r.t_cand_ms),
            "mean_t_select": _mean(lambda r: r.t_select_ms),
            "mean_t_validate": _mean(lambda r: r.t_validate_ms),
            "mean_crr": _mean(lambda r: 1.0 - r.candidate_size / r.universe_size
                              if r.universe_size else float("nan")),
            "mean_universe": _mean(lambda r: float(r.universe_size)
                                   if r.universe_size else float("nan")),
            "mean_candidates": _mean(lambda r: float(r.candidate_size)
                                     if r.universe_size else float("nan")),
            "mean_edges": _mean(lambda r: float(r.interaction_edges)
                                if r.candidate_size else float("nan")),
            "mean_bundle_err": _mean(lambda r: r.bundle_model_error
                                     if r.bundle_size else float("nan")),
            "repair_fraction": _mean(lambda r: r.repair_fraction
                                     if r.bundle_size else float("nan")),
            "budget_exhausted_rate": _mean(lambda r: 1.0 if r.budget_exhausted else 0.0),
            "mean_recall": _mean(lambda r: r.recall_cand),
            "mean_precision": _mean(lambda r: r.precision_cand),
            "cand_gse": _mean(lambda r: float(r.n_gse_cand) if r.candidate_size else float("nan")),
            "cand_crew": _mean(lambda r: float(r.n_crew_cand) if r.candidate_size else float("nan")),
            "cand_sched": _mean(lambda r: float(r.n_sched_cand) if r.candidate_size else float("nan")),
            "cand_stand": _mean(lambda r: float(r.n_stand_cand) if r.candidate_size else float("nan")),
        }
