"""Regression tests for the corrections made in the September 2026 revision."""
import pickle

import pytest

from airport.calibration import minimum_turnaround
from airport.flights import AirportConfig, TurnaroundScenario, TURNAROUND_THETA_MTT
from airport.stands import POSITION_CAPACITY
from airport.turnaround_tasks import critical_path_length
from experiments.revision_harness import RevisionSimulator, RevisionController, Spec, run
from experiments.schedule_audit import audit_schedule
from simulation.disturbances import build_plan


def _controller(seed=1001, layout="stylised", **env):
    spec = Spec(study="test", arm="t", seed=seed, layout=layout, env=env)
    cfg = AirportConfig(seed=seed, layout=layout, n_flights=spec.n_flights,
                        horizon=spec.horizon, **env)
    sim = RevisionSimulator(cfg, build_plan(seed, spec.horizon, "mixed"))
    return RevisionController(sim, spec)


def test_targets_rest_on_stand_feasible_minimum_turnaround():
    scn = TurnaroundScenario(AirportConfig(seed=1001, n_flights=20, horizon=150.0))
    theta = TURNAROUND_THETA_MTT["moderate"]
    for fl in scn.flights.values():
        cp = critical_path_length(fl.tasks)
        mtt = minimum_turnaround(fl.tasks, fl.conflicts, POSITION_CAPACITY)
        assert mtt >= cp - 0.005 * len(fl.tasks)   # 0.01-min grid
        assert fl.min_turnaround == pytest.approx(mtt)
        delay = fl.release - fl.scheduled_release
        assert fl.target >= fl.scheduled_release + theta * mtt - 1e-6
        codes = {t.code: t.duration for t in fl.tasks.values()}
        if {"DEBOARD", "CLEAN", "CATER_F", "BOARD", "PBPREP"} <= set(codes):
            forced = sum(codes[c] for c in ("DEBOARD", "CLEAN", "CATER_F", "BOARD", "PBPREP"))
            assert mtt >= forced - 0.005 * 5  # forward-left door used serially (0.01-min grid)


def test_legacy_target_basis_is_still_available():
    new = TurnaroundScenario(AirportConfig(seed=1001, n_flights=6, horizon=60.0))
    old = TurnaroundScenario(AirportConfig(seed=1001, n_flights=6, horizon=60.0,
                                           target_basis="cp"))
    assert [f.release for f in new.flights.values()][:1] == \
        [f.release for f in old.flights.values()][:1]
    assert any(abs(a.target - b.target) > 1e-6
               for a, b in zip(new.flights.values(), old.flights.values()))


def test_forecast_trace_is_the_aircraft_ready_node():
    ctrl = _controller()
    sim = ctrl.sim
    while not sim.finished() and ctrl.epoch < 60:
        ev = sim.step()
        if ev is not None:
            ctrl.epoch += 1
            ctrl._decide(ev)
    checked = 0
    for fid in ctrl.model.tracked_flights:
        rv = ctrl.model.g.var.get("ready:" + fid)
        if rv is not None and ctrl.ready_trace.get(fid):
            assert ctrl.ready_trace[fid][-1][1] == pytest.approx(rv.value)
            checked += 1
    assert checked > 0


def test_counterfactual_reassignment_matches_live_execution():
    """Validated slacks equal the planning model after executing the action."""
    from algorithms.counterfactual import apply_synthetic
    from algorithms.algorithm1_propagation import PropagationEngine
    ctrl = _controller(seed=1002)
    sim = ctrl.sim
    while not sim.finished() and len(sim.active) < 6:
        ev = sim.step()
        if ev is not None:
            ctrl.epoch += 1
            ctrl._decide(ev)
    blob = pickle.dumps(ctrl.model)
    gen = ctrl.generator
    actions = [a for a in gen.enumerate_universe()
               if a[0] in ("REASSIGN_GSE", "REASSIGN_CREW", "SWAP_GSE_SEQUENCE",
                           "SWAP_CREW_SEQUENCE", "ACTIVATE_RESERVE_GSE")
               and gen.feasible(*a)][:40]
    assert actions
    compared = 0
    for kind, ents, theta in actions:
        model = pickle.loads(blob)
        g = model.g
        g.start_journal()
        seeds = apply_synthetic(model, kind, ents, theta)
        if g.ensure_scc().n_cyclic:
            g.rollback()
            continue
        PropagationEngine(model, "A1").propagate(seeds, collect_messages=False)
        predicted = {v: g.var[v].value for v in g.var if v.startswith("L:")}
        g.rollback()
        live = pickle.loads(blob)
        assert live.sim.execute(kind, ents, theta)
        dirty = live.refresh(live.sim.t)
        PropagationEngine(live, "A2").propagate(dirty)
        for v, val in predicted.items():
            assert live.g.var[v].value == pytest.approx(val, abs=1e-6), (kind, ents, v)
        compared += 1
    assert compared > 0


def test_calendar_mode_admits_into_a_vacant_gap():
    from core.factors import CapacityFactor
    from types import SimpleNamespace as NS
    f = CapacityFactor("f", ["a", "b"], "out", capacity=1)
    f.peer_starts = {"b": 0.0}                 # peer ahead in rank, not started
    f.peer_start_vars = {"b": "sb"}
    f.duration = 5.0
    g = NS(var={"a": NS(value=0.0), "b": NS(value=20.0), "sb": NS(value=10.0)})
    assert f.project(g, 0.0)[0] == 20.0        # conservative map blocks [0, 20)
    f.calendar_mode = True
    assert f.project(g, 0.0)[0] == 0.0         # [0, 5) fits before the peer
    f.duration = 12.0
    assert f.project(g, 0.0)[0] == 20.0        # [0, 12) would overlap [10, 20)


def test_prepositioning_waits_only_on_running_predecessors():
    sim = RevisionSimulator(AirportConfig(seed=1003, n_flights=4, horizon=40.0),
                            build_plan(1003, 40.0, "none"))
    while not sim.active:
        sim.step()
    fid = next(iter(sim.active))
    fl = sim.scn.flights[fid]
    task = next(t for t in fl.tasks.values() if t.predecessors and t.gse)
    unit = sim.scn.fleet[task.gse]
    if all(sim.scn.tasks[p].state == "waiting" for p in task.predecessors):
        assert not sim._imminent(unit, task)
    for p in task.predecessors:
        sim.scn.tasks[p].state = "done"
    assert sim._imminent(unit, task)


def test_delayed_report_hides_an_outage_from_the_controller():
    from airport.observations import observed_status, resource_release
    sim = RevisionSimulator(AirportConfig(seed=1004, n_flights=4, horizon=40.0,
                                          info_delay=10.0), [])
    unit = next(iter(sim.scn.fleet.values()))
    sim.t = 5.0
    unit.status, unit.availability_time = "down", 30.0
    sim._report(("gse", unit.resource_id))
    assert observed_status(sim, unit) == "idle"
    assert resource_release(sim, unit, observed=True) == pytest.approx(5.0)
    assert resource_release(sim, unit) == pytest.approx(30.0)
    sim.t = 15.0
    assert observed_status(sim, unit) == "down"
    assert resource_release(sim, unit, observed=True) == pytest.approx(30.0)


def test_propagation_mode_does_not_change_decisions_in_primary_catalogue():
    digests = set()
    for mode in ("A1", "A2", "AS"):
        summary, ctrl = run(Spec(study="test", arm=mode, seed=1005, propagation=mode,
                                 n_flights=10, horizon=90.0))
        digests.add(tuple((a["t"], a["type"], tuple(a["entities"])) for a in ctrl.actions))
        assert summary["completed"] == summary["n_flights"]
    assert len(digests) == 1


def test_schedule_audit_detects_a_double_booked_unit():
    summary, ctrl = run(Spec(study="test", arm="audit", seed=1006, n_flights=6,
                             horizon=60.0, method="none"))
    assert not any(summary["audit"].values())
    sim = ctrl.sim
    done = [t for t in sim.scn.tasks.values() if t.gse and t.state == "done"]
    a = done[0]
    b = next(t for t in done if t.gse_class == a.gse_class and t.tid != a.tid)
    b.gse = a.gse
    b.start, b.finish = a.start + 0.1, a.finish + 0.1
    assert audit_schedule(sim)["unit"] >= 1


def _bundle_consistency(seed, n_active, max_bundles=60):
    """Counterfactual slacks of random bundles against executing them live."""
    import random
    from algorithms.algorithm1_propagation import PropagationEngine
    from algorithms.counterfactual import apply_bundle
    from algorithms.literature_recovery import compatible, raw_candidate
    ctrl = _controller(seed=seed)
    sim = ctrl.sim
    while not sim.finished() and len(sim.active) < n_active:
        ev = sim.step()
        if ev is not None:
            ctrl.epoch += 1
            ctrl._decide(ev)
    blob = pickle.dumps(ctrl.model)
    gen = ctrl.generator
    feas = [raw_candidate(gen, a, i) for i, a in enumerate(gen.enumerate_universe())
            if a[0] != "RESEQUENCE_TASKS" and gen.feasible(*a)]
    rng = random.Random(seed)
    # Prefer the pattern that exposed the defect: one task leaves the unit
    # another task joins in the same bundle.
    chained = [(a, b) for a in feas for b in feas
               if a.candidate_type == b.candidate_type == "REASSIGN_GSE"
               and sim.scn.tasks[a.entities[0]].gse == b.entities[1] and compatible([a, b])]
    bundles = [list(p) for p in chained[:max_bundles // 2]]
    while len(bundles) < max_bundles:
        k = rng.choice((2, 3, 4))
        b = rng.sample(feas, min(k, len(feas)))
        if compatible(b):
            bundles.append(b)
    checked = 0
    for bundle in bundles:
        model = pickle.loads(blob)
        g = model.g
        g.start_journal()
        seeds = apply_bundle(model, [(a.candidate_type, a.entities, a.theta) for a in bundle])
        cyclic = g.ensure_scc().n_cyclic
        if not cyclic:
            PropagationEngine(model, "A1").propagate(seeds, collect_messages=False)
        predicted = {v: g.var[v].value for v in g.var if v.startswith("L:")}
        g.rollback()
        if cyclic:
            continue
        live = pickle.loads(blob)
        if not all(live.sim.execute(a.candidate_type, a.entities, a.theta) for a in bundle):
            continue
        PropagationEngine(live, "A2").propagate(live.refresh(live.sim.t))
        for v, val in predicted.items():
            assert live.g.var[v].value == pytest.approx(val, abs=1e-6), \
                ([(a.candidate_type, a.entities) for a in bundle], v)
        checked += 1
    return checked, len(chained)


def test_counterfactual_bundles_match_live_execution():
    checked, chained = _bundle_consistency(1002, 8)
    assert checked >= 30
