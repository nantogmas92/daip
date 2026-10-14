"""One paired harness for every study of the revised manuscript.

Every arm of every study runs through the same simulator and controller loop:

* physical noise is keyed by (seed, kind, task, visit), so reordering tasks
  does not reassign draws; disturbance *times and families* are fixed by the
  seed, while the affected task or resource is chosen from the current state;
* all methods replan at the same gate (a change in the set of violated
  targets, a disturbance report, or six minutes without replanning), which
  does not depend on any method's maintained binding or influence records;
* every planning-model evaluation is charged to a common query budget
  (PlanningOracle), including final bundle validation;
* one outer timer covers each whole decision epoch, from state refresh to the
  post-execution refresh;
* every released flight stays in the outcome denominator; the run guard is
  the release horizon plus 600 minutes and unfinished flights are reported;
* the realised schedule is audited independently after the run.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import pickle
import platform
import statistics
import sys
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from airport.flights import AirportConfig                                  # noqa: E402
from algorithms import algorithm2_candidates as a2                         # noqa: E402
from algorithms.completion import complete                               # noqa: E402
from algorithms.control import ClosedLoopController, ControllerConfig      # noqa: E402
from algorithms.literature_recovery import (BudgetExceeded,               # noqa: E402
                                            LiteraturePolicy, PlanningOracle,
                                            raw_candidate)
from experiments.compare_literature import PairedSimulator, clean, keyed_seed  # noqa: E402
from experiments.schedule_audit import audit_schedule                      # noqa: E402
from simulation.disturbances import build_plan                             # noqa: E402

SHOCKS = ("GSE_TRAVEL_DELAY", "GSE_BREAKDOWN", "CREW_UNAVAILABLE", "SERVICE_OVERRUN",
          "APRON_CONGESTION", "STAND_RELEASE_DELAY", "ARRIVAL_DELAY")
REPLAN_PERIOD = 6.0
GUARD = 600.0            # run guard beyond the release horizon (minutes)
PENALTY = 600.0          # sensitivity score per unfinished flight (minutes)
FORECAST_LEADS = (5.0, 10.0, 20.0, 30.0)
PAIR_SHARE = 32.0 / 194.0     # screening share of the query budget (unchanged)
PAIR_BUDGET = 8               # joint pair evaluations per decision (ablation: revision_tits/abl)


@dataclass
class Spec:
    study: str
    arm: str
    seed: int
    layout: str = "stylised"
    method: str = "proposed"          # none | proposed | sms | flns | perho
    representation: str = "R6"
    propagation: str = "A2"
    candidates: str = "C2"
    validation: str = "V1"
    completion: bool = True           # fallback and deepening after selection (V1 only)
    backend: str = "beam"             # beam | greedy | cpsat | milp
    urgency: bool = False
    budget: int = 4                   # actions per bundle
    query_limit: int = 130            # planning-model evaluations per decision
    eps_gain: float = 0.5             # minimum predicted weighted gain to act
    ring: bool = True                 # one-hop structural ring in Algorithm 2
    eps_lambda: float = 0.25          # leverage threshold of Algorithm 2
    max_depth: int = 40               # backward walk depth of Algorithm 2
    n_flights: int = 20
    horizon: float = 150.0
    disturbances: str = "mixed"
    disturbance_rate: float = 1.0
    env: Dict[str, Any] = field(default_factory=dict)   # AirportConfig overrides
    snapshot_every: int = 0           # pickle the model at every k-th search epoch
    snapshot_dir: str = ""

    def key(self) -> str:
        return "%s__%s__%s__%d" % (self.study, self.arm, self.layout, self.seed)


def airport_config(spec: Spec) -> AirportConfig:
    return AirportConfig(seed=spec.seed, layout=spec.layout, n_flights=spec.n_flights,
                         horizon=spec.horizon, **spec.env)


class RevisionSimulator(PairedSimulator):
    """Paired simulator with the long run guard and a physical start check."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.deadline = self.cfg.horizon + GUARD
        self.start_defects: Counter = Counter()

    def _start_task(self, task, el):
        scn = self.scn
        stand = scn.stands[scn.flights[task.flight].stand].node
        if task.gse_class is not None:
            unit = scn.fleet.get(task.gse) if task.gse else None
            if unit is None:
                self.start_defects["no_unit"] += 1
            elif unit.status in ("down", "moving", "serving", "reserve") \
                    or unit.current_location != stand:
                self.start_defects["unit_not_idle_at_stand"] += 1
        if task.crew_skill is not None:
            crew = scn.crews.get(task.crew) if task.crew else None
            if crew is None:
                self.start_defects["no_crew"] += 1
            elif crew.status in ("unavailable", "serving"):
                self.start_defects["crew_not_available"] += 1
        super()._start_task(task, el)


def _reason(oracle, val) -> str:
    if oracle.acceptable(val):
        return "accepted"
    if "cyclic" in val.new_violations:
        return "cyclic"
    if "incompatible" in val.new_violations:
        return "incompatible"
    if val.new_violations:
        return "new_violation"
    return "low_gain"


class RevisionController(ClosedLoopController):
    def __init__(self, sim, spec: Spec):
        self.spec = spec
        # The single-factor ablation (revision_tits/abl) found no measurable
        # quality change from 8 joint pair evaluations instead of a
        # proportional share, while a larger pair budget starves screening;
        # the pair cap therefore leaves the screening share untouched.
        evals = max(1, int(round(spec.query_limit * (1.0 - PAIR_SHARE))) - 2)
        self.pair_budget = min(PAIR_BUDGET, max(0, spec.query_limit - evals - 2))
        cfg = ControllerConfig(
            representation=spec.representation, propagation=spec.propagation,
            candidates=spec.candidates, validation=spec.validation,
            backend=spec.backend, budget=spec.budget, solver_time=0.6,
            intervene=spec.method != "none", eval_budget=evals,
            urgency_only=spec.urgency)
        super().__init__(sim, cfg, spec.key())
        self.interaction.max_joint = self.pair_budget
        self.policy = (LiteraturePolicy(spec.method, keyed_seed(sim.cfg.seed, spec.method))
                       if spec.method in ("sms", "flns", "perho") else None)
        original = self.generator.feasible

        def executable(kind, ents, theta=0.0):
            # Legacy admission advance is executable only against a running
            # holder with enough remaining time (simulation/event_engine.py).
            if not original(kind, ents, theta):
                return False
            if kind == "ADVANCE_POSITION_ADMISSION":
                task = sim.scn.tasks[ents[0]]
                pos = sim.scn.stands[sim.scn.flights[task.flight].stand].positions[task.position]
                holder = sim.scn.tasks.get(pos.occupants[0]) if pos.occupants else None
                return holder is not None and holder.state == "running" \
                    and holder.finish - sim.t > 0.7
            return True
        self.generator.feasible = executable
        self.epoch_log: List[Dict[str, Any]] = []
        self.actions: List[Dict[str, Any]] = []
        self.dispatch_log: List[Dict[str, Any]] = []
        self.pending = None
        self.search_index = 0
        self.snapshot_files: List[str] = []
        self.history: Dict[str, List[str]] = defaultdict(list)

    # ------------------------------------------------------------------
    def _decide(self, ev) -> None:
        sim, spec = self.sim, self.spec
        t0 = time.perf_counter()
        excluded = 0.0
        new_f = self.model.sync(sim.t)
        dirty = self.model.refresh(sim.t)
        seeds = list(dict.fromkeys(list(new_f) + list(dirty)))
        res = self.engine.propagate(seeds) if seeds else None
        t_prop = (time.perf_counter() - t0) * 1000.0
        self._snapshot_predictions()
        targets = self.detector.detect(sim.t)
        cyclic = self.model.g.ensure_scc().n_cyclic if self.model.g.fac else 0

        delayed = None
        if self.pending and sim.t >= self.pending[0] - 1e-9:
            due, requested, actions = self.pending
            oracle = PlanningOracle(self.model, 1, spec.eps_gain)
            valid = all(self.generator.feasible(a.candidate_type, a.entities, a.theta)
                        for a in actions)
            check = oracle.evaluate(actions) if valid else None
            ok = bool(check and oracle.acceptable(check))
            if ok:
                self._apply(actions, delayed=True)
                d2 = self.model.refresh(sim.t)
                if d2:
                    self.engine.propagate(d2)
                targets = self.detector.detect(sim.t)
            delayed = ok
            self.dispatch_log.append(dict(requested=requested, due=due, actual=sim.t,
                                          executed=ok, valid=valid))
            self.pending = None

        signature = tuple(sorted(z.zid for z in targets))
        shock = ev.event_type in SHOCKS
        search = (bool(targets) and spec.method != "none" and self.pending is None
                  and (signature != self._last_signature or shock
                       or sim.t - self._last_replan >= REPLAN_PERIOD))
        row: Dict[str, Any] = dict(epoch=self.epoch, t=round(sim.t, 4), ev=ev.event_type,
                                   search=search, n_viol=len(targets),
                                   resid=round(sum(z.weight * z.residual for z in targets), 4),
                                   active=len(sim.active),
                                   factors=len(self.model.g.fac),
                                   touched=res.touched if res else 0,
                                   seeded=bool(seeds), t_prop=round(t_prop, 4))
        if cyclic:
            row["cyclic"] = cyclic
        if delayed is not None:
            row["delayed_exec"] = delayed
        if search:
            self.search_index += 1
            self._last_signature = signature
            self._last_replan = sim.t
            if spec.snapshot_every and self.search_index % spec.snapshot_every == 0:
                ts = time.perf_counter()
                self._save_snapshot(targets)
                excluded += time.perf_counter() - ts
            oracle = PlanningOracle(self.model, spec.query_limit, spec.eps_gain)
            info: Dict[str, Any] = {}
            try:
                if spec.method == "proposed":
                    chosen, info = self._proposed(targets, oracle)
                else:
                    raw = self.generator.enumerate_universe()
                    pool = [raw_candidate(self.generator, a, i) for i, a in enumerate(raw)
                            if self.generator.feasible(*a)]
                    chosen = self.policy.solve(self, oracle, pool, targets)
                    info = dict(pool=len(pool), universe=len(raw),
                                details=dict(self.policy.last_details))
            except BudgetExceeded:
                chosen, info = [], dict(budget_exhausted=True)
            check = None
            if chosen and spec.validation != "V0":
                if oracle.queries < oracle.limit:
                    check = oracle.evaluate(chosen)
                else:
                    check = oracle.cache.get(tuple(a.signature for a in chosen))
                if check is None or not oracle.acceptable(check):
                    info["final"] = "rejected"
                    chosen = []
            if chosen and check is not None:
                row["pred_gain"] = round(oracle.base.delay - check.delay, 4)
                row["worsened"] = len(check.worsened)
            if chosen and sim.cfg.action_lead_time > 0:
                self.pending = (sim.t + sim.cfg.action_lead_time, sim.t, list(chosen))
                row["pending"] = len(chosen)
            elif chosen:
                applied = self._apply(chosen)
                if applied:
                    d2 = self.model.refresh(sim.t)
                    if d2:
                        self.engine.propagate(d2)
            post = self.detector.detect(sim.t)
            pre_ids = {z.zid for z in targets}
            row["new_viol_post"] = sum(1 for z in post if z.zid not in pre_ids)
            row["resid_post"] = round(sum(z.weight * z.residual for z in post), 4)
            row["queries"] = oracle.queries
            row["cf_work"] = oracle.work
            row["bundle"] = len(chosen)
            row["base_delay"] = round(oracle.base.delay, 4)
            row.update({k: v for k, v in info.items() if k != "details"})
            if "details" in info:
                row["details"] = info["details"]
        row["ms"] = round((time.perf_counter() - t0 - excluded) * 1000.0, 4)
        self.epoch_log.append(row)

    # ------------------------------------------------------------------
    def _proposed(self, targets, oracle):
        """Screening, coordination and validation, then anytime completion."""
        if not (self.spec.completion and self.spec.validation != "V0"):
            return self._select(targets, oracle)
        try:
            chosen, info = self._select(targets, oracle)
        except BudgetExceeded:
            chosen, info = [], dict(budget_exhausted=True)
        chosen, extra = complete(self, targets, oracle, chosen)
        info.update(extra)
        return chosen, info

    def _select(self, targets, oracle):
        gen = self.generator
        info: Dict[str, Any] = dict(proposals=0, statuses=[], reasons=[])
        original = gen.counterfactual_leverage

        def counted(kind, ents, theta, violations):
            action = raw_candidate(gen, (kind, tuple(ents), theta), 0)
            value = oracle.evaluate([action])
            if not value.slacks:
                return {}, {}, 0
            lev = {z.zid: value.slacks[z.zid] - oracle.base.slacks[z.zid]
                   for z in violations if z.zid in value.slacks}
            side = {k: v - oracle.base.slacks[k] for k, v in value.slacks.items()
                    if k not in lev and k in oracle.base.slacks}
            return lev, side, value.work

        gen.counterfactual_leverage = counted
        try:
            cands, stats = gen.generate(targets, want_universe=False)
        finally:
            gen.counterfactual_leverage = original
        info.update(cands=len(cands), generated=stats.generated,
                    counterfactuals=stats.counterfactuals, truncated=bool(stats.truncated),
                    region=stats.region_size, phase_a=stats.phase_a)
        if self.spec.urgency:
            cands = self._urgency_filter(cands, targets)
            info["cands"] = len(cands)
        if not cands:
            return [], info
        original_kappa = self.interaction._kappa

        def counted_pair(a, b, violations):
            val = oracle.evaluate([a, b])
            if not val.slacks:
                return {}
            return {z.zid: val.slacks[z.zid] - oracle.base.slacks[z.zid]
                    - a.leverage_on(z.zid) - b.leverage_on(z.zid) for z in violations
                    if z.zid in val.slacks}
        self.interaction._kappa = counted_pair
        try:
            graph = self.interaction.build(cands, targets)
        finally:
            self.interaction._kappa = original_kappa
        info["pairs"] = graph.joint_evals
        nogoods = []
        for _ in range(2):
            sel = self.selector.select(cands, graph, targets, nogoods)
            info["statuses"].append(sel.status)
            if not sel.chosen:
                break
            info["proposals"] += 1
            predicted = sum(z.weight * min(z.residual, max(0.0, sel.predicted.get(z.zid, 0.0)))
                            for z in targets)
            info.setdefault("predicted_gain", []).append(round(predicted, 4))
            if self.spec.validation == "V0":
                info["reasons"].append("unvalidated")
                return sel.chosen, info
            val = oracle.evaluate(sel.chosen)
            reason = _reason(oracle, val)
            info["reasons"].append(reason)
            info.setdefault("validated_gain", []).append(
                round(oracle.base.delay - val.delay, 4) if math.isfinite(val.delay) else None)
            if reason == "accepted":
                return sel.chosen, info
            nogoods.append(tuple(a.candidate_id for a in sel.chosen))
        info["fallback_available"] = bool(oracle.best[0])
        return [], info

    # ------------------------------------------------------------------
    def _apply(self, chosen, delayed: bool = False) -> int:
        sim = self.sim
        applied = 0
        for a in chosen:
            kind, ents = a.candidate_type, tuple(a.entities)
            task = sim.scn.tasks.get(ents[0]) if ents else None
            before = None
            if task is not None and kind in ("REASSIGN_GSE", "ACTIVATE_RESERVE_GSE"):
                before = task.gse
            elif task is not None and kind == "REASSIGN_CREW":
                before = task.crew
            feasible = self.generator.feasible(kind, ents, a.theta)
            ok = bool(feasible and sim.execute(kind, ents, a.theta))
            reversal = False
            if ok and before is not None and len(ents) > 1:
                hist = self.history[ents[0]]
                if not hist:
                    hist.append(before)
                reversal = ents[1] in hist
                hist.append(ents[1])
            applied += int(ok)
            self.actions.append(dict(
                t=round(sim.t, 4), epoch=self.epoch, type=kind, entities=list(ents),
                task=ents[0] if ents else "", flight=task.flight if task else "",
                cost=a.cost, executed=ok, feasible_at_execution=bool(feasible),
                reversal=reversal, delayed=delayed))
        return applied

    # ------------------------------------------------------------------
    def _save_snapshot(self, targets) -> None:
        folder = Path(self.spec.snapshot_dir)
        folder.mkdir(parents=True, exist_ok=True)
        name = "%s__e%05d.pkl.gz" % (self.spec.key(), self.epoch)
        payload = dict(model=self.model, spec=asdict(self.spec), epoch=self.epoch,
                       t=self.sim.t, targets=[z.zid for z in targets])
        with gzip.open(folder / name, "wb", compresslevel=5) as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        self.snapshot_files.append(name)


# ---------------------------------------------------------------------------
# summaries
# ---------------------------------------------------------------------------

def _q(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    pos = q * (len(xs) - 1)
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def _stats(values: Sequence[float]) -> Dict[str, float]:
    return dict(n=len(values),
                mean=statistics.fmean(values) if values else float("nan"),
                median=_q(values, 0.5), p95=_q(values, 0.95), p99=_q(values, 0.99),
                max=max(values) if values else float("nan"))


def forecast_errors(ctrl) -> Dict[str, Dict[str, float]]:
    """Error of the ready-time forecast standing a fixed lead before readiness."""
    out: Dict[str, Dict[str, float]] = {}
    sim = ctrl.sim
    for lead in FORECAST_LEADS:
        errs = []
        for fid, fl in sim.scn.flights.items():
            if fl.ready != fl.ready:
                continue
            trace = ctrl.ready_trace.get(fid)
            t_q = fl.ready - lead
            if not trace or t_q < trace[0][0]:
                continue
            pred = ctrl._trace_at(trace, t_q)
            if pred is None or abs(pred) > 1e17:
                continue
            errs.append(pred - fl.ready)
        out["lead%d" % int(lead)] = dict(
            n=len(errs), bias=statistics.fmean(errs) if errs else float("nan"),
            mae=statistics.fmean(abs(e) for e in errs) if errs else float("nan"))
    return out


def summarise(ctrl, spec: Spec, wall: float) -> Dict[str, Any]:
    sim = ctrl.sim
    base = ClosedLoopController.summary(ctrl)
    flights = list(sim.scn.flights.values())
    n = len(flights)
    lateness, weighted, flight_rows = [], 0.0, []
    unfinished = []
    for f in flights:
        done = f.ready == f.ready
        if not done:
            unfinished.append(f.flight_id)
        end = f.ready if done else sim.t
        late = max(0.0, end - f.target)
        lateness.append(late)
        weighted += f.weight * late
        flight_rows.append(dict(f=f.flight_id, type=f.actype, w=f.weight, stand=f.stand,
                                release=round(f.release, 4), target=round(f.target, 4),
                                ready=round(f.ready, 4) if done else None,
                                late=round(late, 4), mtt=round(f.min_turnaround, 3)))
    executed = [a for a in ctrl.actions if a["executed"]]
    per_task = Counter(a["task"] for a in executed)
    searches = [r for r in ctrl.epoch_log if r["search"]]
    updates = [r for r in ctrl.epoch_log if not r["search"]]
    propagating = [r for r in ctrl.epoch_log if r["seeded"] and r["factors"]]
    statuses = Counter(s for r in searches for s in r.get("statuses", []))
    reasons = Counter(s for r in searches for s in r.get("reasons", []))
    audit = audit_schedule(sim)
    return dict(
        key=spec.key(), study=spec.study, arm=spec.arm, layout=spec.layout, seed=spec.seed,
        method=spec.method, spec=asdict(spec), config=asdict(sim.scn.cfg),
        n_flights=n, completed=n - len(unfinished), unfinished=unfinished,
        final_time=sim.t,
        delay_w_per_flight=weighted / n,
        delay_w_penalised=(weighted + sum(PENALTY * f.weight for f in flights
                                          if f.flight_id in unfinished)) / n,
        delay_mean=statistics.fmean(lateness),
        delay_p90=_q(lateness, 0.90), delay_p95=_q(lateness, 0.95), delay_max=max(lateness),
        on_time_rate=sum(1 for f in flights if f.ready == f.ready
                         and f.ready <= f.target + 1e-9) / n,
        interventions=len(executed), failed_actions=len(ctrl.actions) - len(executed),
        intervention_cost=sum(a["cost"] for a in executed),
        actions_by_type=dict(Counter(a["type"] for a in executed)),
        reserve_activations=sum(1 for a in executed if a["type"] == "ACTIVATE_RESERVE_GSE"),
        tasks_changed=len(per_task),
        max_changes_per_task=max(per_task.values()) if per_task else 0,
        reversals=sum(1 for a in executed if a["reversal"]),
        delayed_executions=sum(1 for a in executed if a["delayed"]),
        delayed_rejections=sum(1 for r in ctrl.dispatch_log if not r["executed"]),
        search_epochs=len(searches), decision_epochs=len(ctrl.epoch_log),
        proposals=sum(r.get("proposals", 0) for r in searches),
        accepted_bundles=sum(1 for r in searches if r.get("bundle")),
        solver_status=dict(statuses), validation_reasons=dict(reasons),
        fallback_available=sum(1 for r in searches if r.get("fallback_available")),
        completion=dict(Counter(r["completion"] for r in searches
                                if r.get("bundle") and r.get("completion"))),
        worsened_in_accepted=sum(r.get("worsened", 0) for r in searches if r.get("bundle")),
        budget_exhausted=sum(1 for r in searches if r.get("budget_exhausted")),
        queries_total=sum(r.get("queries", 0) for r in searches),
        queries_max=max((r.get("queries", 0) for r in searches), default=0),
        cf_factor_work=sum(r.get("cf_work", 0) for r in searches),
        latency_update_ms=_stats([r["ms"] for r in updates]),
        latency_search_ms=_stats([r["ms"] for r in searches]),
        latency_all_ms=_stats([r["ms"] for r in ctrl.epoch_log]),
        pf_propagating=statistics.fmean(r["touched"] / r["factors"] for r in propagating)
        if propagating else float("nan"),
        pf_all=statistics.fmean(r["touched"] / r["factors"] for r in ctrl.epoch_log
                                if r["factors"]) if ctrl.epoch_log else float("nan"),
        t_prop_ms=_stats([r["t_prop"] for r in propagating]),
        factors_mean=statistics.fmean(r["factors"] for r in ctrl.epoch_log) if ctrl.epoch_log else 0,
        factors_max=max((r["factors"] for r in ctrl.epoch_log), default=0),
        active_mean=statistics.fmean(r["active"] for r in ctrl.epoch_log) if ctrl.epoch_log else 0,
        active_peak=max((r["active"] for r in ctrl.epoch_log), default=0),
        candidates_mean=statistics.fmean(r["cands"] for r in searches if "cands" in r)
        if any("cands" in r for r in searches) else float("nan"),
        e_rep_ready=base.get("e_rep_ready"), binding_accuracy=base.get("binding_accuracy"),
        forecast=forecast_errors(ctrl),
        new_violations_post=sum(r.get("new_viol_post", 0) for r in searches),
        audit=audit, start_defects=dict(sim.start_defects),
        live_cyclic_max=max((r.get("cyclic", 0) for r in ctrl.epoch_log), default=0),
        snapshots=ctrl.snapshot_files, wall_s=wall,
        flights=flight_rows,
    )


RUNTIME = ("airport", "algorithms", "core", "optimization", "simulation")
RUNTIME_FILES = ("experiments/revision_harness.py", "experiments/compare_literature.py",
                 "experiments/schedule_audit.py")


def source_files() -> List[Path]:
    files = [p for d in RUNTIME for p in (ENGINE / d).rglob("*.py")
             if "__pycache__" not in p.parts]
    return sorted(files + [ENGINE / f for f in RUNTIME_FILES])


def source_digest() -> str:
    """SHA-256 over every module that can affect a run's behaviour."""
    h = hashlib.sha256()
    for p in source_files():
        h.update(str(p.relative_to(ENGINE)).replace(os.sep, "/").encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def run(spec: Spec) -> tuple:
    a2.EPS_LAMBDA = spec.eps_lambda
    a2.STRUCTURAL_RING = spec.ring
    a2.MAX_CAUSAL_DEPTH = spec.max_depth
    cfg = airport_config(spec)
    plan = build_plan(spec.seed, spec.horizon, spec.disturbances, rate=spec.disturbance_rate)
    sim = RevisionSimulator(cfg, plan)
    ctrl = RevisionController(sim, spec)
    start = time.perf_counter()
    ctrl.run()
    wall = time.perf_counter() - start
    return summarise(ctrl, spec, wall), ctrl


def run_and_save(args) -> Dict[str, Any]:
    spec, out_dir = args
    out = Path(out_dir)
    path = out / (spec.key() + ".json")
    if path.exists():
        return dict(key=spec.key(), existing=True)
    try:
        summary, ctrl = run(spec)
        summary["source_sha256"] = source_digest()
        summary["python"] = sys.version.split()[0]
        summary["platform"] = platform.platform()
        logs = dict(epochs=ctrl.epoch_log, actions=ctrl.actions, dispatch=ctrl.dispatch_log)
        with gzip.open(out / (spec.key() + ".logs.json.gz"), "wt", encoding="utf-8") as fh:
            json.dump(clean(logs), fh)
        path.write_text(json.dumps(clean(summary), indent=1, allow_nan=False), encoding="utf-8")
        return dict(key=spec.key(), delay=round(summary["delay_w_per_flight"], 3),
                    completed=summary["completed"], wall=round(summary["wall_s"], 1))
    except Exception:
        err = dict(key=spec.key(), error=traceback.format_exc())
        (out / (spec.key() + ".error.txt")).write_text(err["error"], encoding="utf-8")
        return err


def run_many(specs: Sequence[Spec], out_dir: str, workers: int = 1) -> List[Dict[str, Any]]:
    os.environ["PYTHONHASHSEED"] = "0"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    jobs = [(s, out_dir) for s in specs]
    results = []
    if workers <= 1:
        for j in jobs:
            r = run_and_save(j)
            results.append(r)
            print(json.dumps(r)[:300], flush=True)
        return results
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(run_and_save, j) for j in jobs]
        for i, fut in enumerate(as_completed(futures)):
            r = fut.result()
            results.append(r)
            if "error" in r or (i + 1) % 25 == 0:
                print(i + 1, "/", len(jobs), json.dumps(r)[:300], flush=True)
    return results
