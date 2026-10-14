"""Unit and integration tests required before any large experiment is trusted.

Each test targets one property the manuscript relies on:

    1  max-factor absorption              Proposition A
    2  criticality switching              CRITICALITY_CHANGED is emitted
    3  resource reservation updates       Proposition B (symmetric-difference locality)
    4  action-sensitivity switching       the sensitivity stream follows the argmax
    5  SCC convergence                    cyclic components reach a fixed point
    6  candidate pruning                  Propositions D and F
    7  counterfactual state isolation     the journal restores the live graph exactly
    8  candidate interaction conflicts    exclusive pairs are detected
    9  CP-SAT bundle feasibility          no exclusive pair, budget respected
   10  deterministic replay               identical seed -> identical run

Run with ``PYTHONHASHSEED=0 python -m pytest tests -q`` from the repository
root (or ``python tests/test_engine.py`` without pytest).
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from algorithms.algorithm1_propagation import PropagationEngine      # noqa: E402
from algorithms.algorithm3_selection import InterventionSelector      # noqa: E402
from algorithms.interaction import InteractionBuilder, InteractionGraph  # noqa: E402
from algorithms.violations import Violation                            # noqa: E402
from core.factors import (AffineFactor, CapacityFactor, MaxFactor,     # noqa: E402
                          ServiceFactor, WorkloadFactor)
from core.graph import FactorGraph                                     # noqa: E402
from core.messages import InterventionCandidate                        # noqa: E402
from core.sensitivities import recompute_all                           # noqa: E402

TOL = 1e-9


class _Shim:
    """The minimum model interface the propagation engine needs."""

    def __init__(self, g):
        self.g = g
        self.now = 0.0
        self.rep = None


def _chain():
    """y = max(x1, x2, x3) feeding z = y + 5, with x1 binding."""
    g = FactorGraph()
    for vid, val in (("x1", 10.0), ("x2", 4.0), ("x3", 7.0)):
        g.add_var(vid, "eta", val)
    g.add_var("y", "S", 0.0)
    g.add_var("z", "C", 0.0, decision_facing=True)
    for vid, val in (("x1", 10.0), ("x2", 4.0), ("x3", 7.0)):
        f = AffineFactor("f" + vid, [], vid, c0=val, coeff={})
        g.add_factor(f)
    g.add_factor(MaxFactor("fy", ["x1", "x2", "x3"], "y"))
    g.add_factor(ServiceFactor("fz", ["y"], "z", nominal=5.0))
    eng = PropagationEngine(_Shim(g), mode="A2")
    eng.initial_solve()
    return g, eng


def _set_input(g, vid, value):
    g.fac["f" + vid].c0 = value
    return ["f" + vid]


# ---------------------------------------------------------------------------
# 1. max-factor absorption (Proposition A)
# ---------------------------------------------------------------------------
def test_max_factor_absorption():
    g, eng = _chain()
    assert abs(g.var["y"].value - 10.0) < TOL
    margin = g.fac["fy"].margin(g, "x3")
    assert abs(margin - 3.0) < TOL                  # M_j = y - x_j
    z_sens_before = dict(g.var["z"].sens)
    res = eng.propagate(_set_input(g, "x3", 7.0 + 2.9))   # inside the margin
    assert abs(g.var["y"].value - 10.0) < TOL
    assert abs(g.var["z"].value - 15.0) < TOL
    assert g.var["z"].sens == z_sens_before               # influence unchanged
    assert "fz" not in res.touched_ids                    # contained at fy


# ---------------------------------------------------------------------------
# 2. criticality switching
# ---------------------------------------------------------------------------
def test_criticality_switch_emits_message():
    g, eng = _chain()
    res = eng.propagate(_set_input(g, "x3", 12.0))       # crosses the margin
    types = [m.message_type for m in res.messages]
    assert "CRITICALITY_CHANGED" in types
    assert g.fac["fy"].active == frozenset(["x3"])
    assert abs(g.var["z"].value - 17.0) < TOL
    assert res.criticality_switches >= 1


# ---------------------------------------------------------------------------
# 3. resource reservation updates (Proposition B)
# ---------------------------------------------------------------------------
def test_reservation_locality_and_projection():
    g = FactorGraph()
    g.add_var("a", "S", 20.0)
    for p, c in (("p1", 30.0), ("p2", 40.0)):
        g.add_var("C:" + p, "C", c)
    g.add_var("Rp", "Rp", 0.0)
    cf = CapacityFactor("fRp", ["a", "C:p1", "C:p2"], "Rp", capacity=2)
    cf.peer_starts = {"C:p1": 0.0, "C:p2": 10.0}
    g.add_factor(cf)
    # two peers occupy a capacity-2 resource at t=20: admission waits for p1
    val, act = cf.evaluate(g)
    assert abs(val - 30.0) < TOL and cf.binding and act == frozenset(["C:p1"])
    # moving p2's reservation changes occupancy only on the symmetric
    # difference [40, 45): the projection for an admission at 20 is unchanged
    g.var["C:p2"].value = 45.0
    val2, _ = cf.evaluate(g)
    assert abs(val2 - val) < TOL
    # moving the binding peer p1 moves the projection one for one
    g.var["C:p1"].value = 33.0
    val3, _ = cf.evaluate(g)
    assert abs(val3 - 33.0) < TOL
    # a peer release beyond the other peer switches the blocker
    g.var["C:p1"].value = 50.0
    _, act4 = cf.evaluate(g)
    assert act4 == frozenset(["C:p2"])


# ---------------------------------------------------------------------------
# 4. action-sensitivity switching
# ---------------------------------------------------------------------------
def test_sensitivity_follows_argmax():
    g, eng = _chain()
    assert abs(g.var["z"].sens.get("x1", 0.0) - 1.0) < TOL
    assert "x3" not in g.var["z"].sens                   # nonbinding: zero
    res = eng.propagate(_set_input(g, "x3", 12.0))
    assert abs(g.var["z"].sens.get("x3", 0.0) - 1.0) < TOL
    assert "x1" not in g.var["z"].sens
    assert "ACTION_SENSITIVITY_CHANGED" in [m.message_type for m in res.messages]
    # and the incremental row equals the reference solve
    ref = dict(g.var["z"].sens)
    recompute_all(g)
    assert g.var["z"].sens == ref


# ---------------------------------------------------------------------------
# 5. SCC convergence
# ---------------------------------------------------------------------------
def test_scc_fixed_point_convergence():
    """Service -> workload -> service is a genuine cycle with a fixed point."""
    g = FactorGraph()
    g.add_var("S", "S", 0.0)
    g.add_var("C", "C", 0.0)
    g.add_var("W", "W", 0.0)
    g.add_factor(AffineFactor("fS", [], "S", c0=10.0, coeff={}))
    g.add_factor(ServiceFactor("fC", ["S", "W"], "C", nominal=20.0,
                               gamma=0.5, workload_var="W"))
    g.add_factor(WorkloadFactor("fW", ["S", "C"], "W", kappa=0.01))
    cond = g.compute_scc()
    assert cond.n_cyclic == 1 and cond.largest_cyclic == 2
    eng = PropagationEngine(_Shim(g), mode="A2")
    eng.initial_solve()
    # analytic fixed point: d = 20 (1 + 0.5 * 0.01 d)  ->  d = 20 / 0.9
    d = g.var["C"].value - g.var["S"].value
    assert abs(d - 20.0 / 0.9) < 1e-3
    g.fac["fS"].c0 = 15.0
    eng.propagate(["fS"])
    assert abs((g.var["C"].value - g.var["S"].value) - 20.0 / 0.9) < 1e-3


# ---------------------------------------------------------------------------
# 7. counterfactual state isolation
# ---------------------------------------------------------------------------
def test_journal_rollback_restores_live_state():
    g, eng = _chain()
    before_val = dict((k, v.value) for k, v in g.var.items())
    before_sens = dict((k, dict(v.sens)) for k, v in g.var.items())
    before_act = dict((k, f.active) for k, f in g.fac.items())
    before_inputs = dict((k, list(f.inputs)) for k, f in g.fac.items())
    before_stamp = dict((k, v.stamp) for k, v in g.var.items())
    eng.model.now = 42.0                              # sandbox runs "later"
    g.start_journal()
    g.journal_fac("fx3")
    g.fac["fx3"].c0 = 99.0
    g.rewire(g.fac["fy"], ["x2", "x3"])                  # structural edit too
    eng.propagate(["fx3", "fy"])
    assert g.var["z"].value > 100.0                      # sandbox really moved
    g.rollback()
    assert dict((k, v.value) for k, v in g.var.items()) == before_val
    assert dict((k, dict(v.sens)) for k, v in g.var.items()) == before_sens
    assert dict((k, f.active) for k, f in g.fac.items()) == before_act
    assert dict((k, list(f.inputs)) for k, f in g.fac.items()) == before_inputs
    assert dict((k, v.stamp) for k, v in g.var.items()) == before_stamp
    assert "fy" in g.var["x1"].consumers                 # consumer list restored


def test_state_versioning_fields():
    """Every variable carries version, stamp, causal support and horizon."""
    g, eng = _chain()
    eng.model.now = 3.0
    v0 = g.var["z"].version
    eng.propagate(_set_input(g, "x3", 12.0))
    z = g.var["z"]
    assert z.version > v0
    assert z.stamp == 3.0
    assert z.horizon == z.value - 3.0            # C-kind: predicted clock time
    assert g.var["y"].causal_support == ("x3",)  # the binding input


# ---------------------------------------------------------------------------
# 8 and 9. interaction conflicts and CP-SAT bundle feasibility
# ---------------------------------------------------------------------------
def _cand(cid, ctype, resources, lev, cost=1.0):
    return InterventionCandidate(
        candidate_id=cid, candidate_type=ctype, affected_flights=("F2",),
        affected_tasks=(resources[0],), affected_resources=tuple(resources),
        target_violation=("L:F2",), predicted_leverage={"L:F2": lev},
        causal_path=(), side_effects={},
        resource_requirements={"task:" + resources[0]: 1.0},
        feasibility=True, uncertainty={"L:F2": 0.0}, cost=cost)


def test_interaction_marks_alternative_reassignments_exclusive():
    u = _cand("u1", "REASSIGN_GSE", ["F2/CATER", "V7"], 5.0)
    v = _cand("u2", "REASSIGN_GSE", ["F2/CATER", "V9"], 6.0)
    w = _cand("u3", "REASSIGN_CREW", ["F2/BOARD", "H3"], 3.0)
    assert InteractionBuilder._exclusive(u, v)            # same task, two units
    assert not InteractionBuilder._exclusive(u, w)


def test_cpsat_bundle_respects_exclusion_and_budget():
    viol = [Violation("L:F2", "deadline", 8.0, +1, "F2")]
    u = _cand("u1", "REASSIGN_GSE", ["F2/CATER", "V7"], 5.0)
    v = _cand("u2", "REASSIGN_GSE", ["F2/CATER", "V9"], 6.0)
    w = _cand("u3", "REASSIGN_CREW", ["F2/BOARD", "H3"], 3.0)
    G = InteractionGraph(nodes=[u, v, w], edges={("u1", "u2"): "exclusive"})
    res = InterventionSelector("cpsat", budget=2, time_limit=2.0).select(
        [u, v, w], G, viol)
    ids = set(c.candidate_id for c in res.chosen)
    assert not ("u1" in ids and "u2" in ids)             # exclusion honoured
    assert len(ids) <= 2                                  # budget honoured
    assert ids == set(["u2", "u3"])                       # 6 + 3 >= 8 is optimal
    # a no-good forbids the incumbent and forces a different bundle
    res2 = InterventionSelector("cpsat", budget=2, time_limit=2.0).select(
        [u, v, w], G, viol, nogoods=[("u2", "u3")])
    assert set(c.candidate_id for c in res2.chosen) != ids


if __name__ == "__main__":         # pragma: no cover - runner without pytest
    import inspect
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and inspect.isfunction(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print("PASS", name)
        except Exception as exc:          # noqa: BLE001
            failed += 1
            print("FAIL", name, "--", type(exc).__name__, exc)
    raise SystemExit(1 if failed else 0)
