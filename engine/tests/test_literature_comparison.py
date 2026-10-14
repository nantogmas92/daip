import copy
from itertools import product
import math
import numpy as np
import pytest

from algorithms.literature_recovery import (BudgetExceeded, PlanningOracle, OrderAware,
    Evaluation, LiteraturePolicy, compatible, minsum, raw_candidate)
from experiments.compare_literature import PairedSimulator, ComparisonController
from airport.flights import AirportConfig
from simulation.disturbances import build_plan


def test_minsum_matches_exhaustive_tree_optimum():
    unary=[np.array([0.,2.]),np.array([1.,0.]),np.array([.2,3.])]
    pairs={(0,1):np.array([[4.,.3],[0.,2.]]),(1,2):np.array([[0.,5.],[3.,0.]])}
    selected,_=minsum(unary,pairs,rounds=100,damping=0.)
    def cost(x):
        return sum(unary[i][a] for i,a in enumerate(x))+sum(m[x[i],x[j]] for (i,j),m in pairs.items())
    assert cost(selected)==pytest.approx(min(cost(x) for x in product(range(2),repeat=3)))


def test_oass_transition_credit_and_mixed_objectives():
    policy=OrderAware(6)
    policy.update(2,(4.,3.),(2.,3.))
    assert policy.log_weights[0,2]==pytest.approx(.5)
    policy.update(1,(2.,3.),(3.,4.))
    assert policy.log_weights[2,1]==pytest.approx(-1.)
    policy.update(0,(3.,4.),(2.,5.))
    assert policy.log_weights[1,0]==0.


def test_keyed_noise_invariant_to_unrelated_call_order():
    cfg=AirportConfig(n_flights=3,seed=9001,horizon=40.)
    a,b=PairedSimulator(cfg),PairedSimulator(cfg)
    expected=a._sample_noise('service','task-a',.1)
    b._sample_noise('travel','unrelated',.1)
    assert b._sample_noise('service','task-a',.1)==expected
    assert a._disturbance_rng({'time':5,'kind':'D2'}).random()==b._disturbance_rng({'time':5,'kind':'D2'}).random()


def live_controller():
    cfg=AirportConfig(n_flights=6,seed=9002,horizon=65.)
    sim=PairedSimulator(cfg,build_plan(9002,65.,'mixed'))
    ctrl=ComparisonController(sim,'none')
    for _ in range(3000):
        event=sim.step()
        if event:
            ctrl.epoch+=1; ctrl._decide(event)
            raw=ctrl.generator.enumerate_universe()
            pool=[raw_candidate(ctrl.generator,a,i) for i,a in enumerate(raw) if ctrl.generator.feasible(*a)]
            if len(pool)>3 and ctrl.detector.detect(sim.t): return ctrl,pool
    pytest.fail('No nontrivial live decision found')


def test_oracle_rollback_budget_and_resource_conflicts():
    ctrl,pool=live_controller()
    g=ctrl.model.g
    before={k:(v.value,v.version,v.active_set,dict(v.sens),list(v.consumers)) for k,v in g.var.items()}
    factors={k:copy.deepcopy(v.__dict__) for k,v in g.fac.items()}
    simulator=copy.deepcopy(ctrl.sim.scn.tasks)
    oracle=PlanningOracle(ctrl.model,2)
    assert not compatible([pool[0],pool[0]])
    value=oracle.evaluate([pool[0]])
    assert math.isfinite(value.delay)
    assert oracle.evaluate([pool[0]]).score==value.score
    with pytest.raises(BudgetExceeded): oracle.evaluate([pool[0]])
    assert before=={k:(v.value,v.version,v.active_set,dict(v.sens),list(v.consumers)) for k,v in g.var.items()}
    assert factors=={k:v.__dict__ for k,v in g.fac.items()}
    # NaN-valued unobserved start/finish fields are unchanged by rollback.
    from dataclasses import asdict
    for key, before_task in simulator.items():
        before_fields, after_fields = asdict(before_task), asdict(ctrl.sim.scn.tasks[key])
        assert before_fields.keys() == after_fields.keys()
        for name, value in before_fields.items():
            other = after_fields[name]
            assert (isinstance(value, float) and isinstance(other, float)
                    and math.isnan(value) and math.isnan(other)) or value == other


def test_budget_exhaustion_does_not_allow_unchecked_action():
    ctrl,pool=live_controller()
    oracle=PlanningOracle(ctrl.model,1)
    oracle.evaluate([pool[0]])
    chosen,val=oracle.best
    if chosen:
        assert oracle.acceptable(val)
        assert tuple(a.signature for a in chosen) in oracle.cache


def test_sms_evaluates_complementary_actions_jointly():
    from dataclasses import replace
    ctrl,pool=live_controller()
    a=replace(pool[0],candidate_id='a',affected_resources=('task-a',),
              resource_requirements={'task:task-a':1.})
    b=replace(a,candidate_id='b',affected_resources=('task-b',),
              resource_requirements={'task:task-b':1.})
    class ToyOracle:
        limit=194
        queries=0
        cache={}
        base=Evaluation({'L:F':-10.},{},10.,0.,())
        best=([],base)
        def evaluate(self, actions):
            self.queries+=1
            if self.queries>self.limit: raise BudgetExceeded()
            ids={x.candidate_id for x in actions}
            delay=0. if ids=={'a','b'} else (8. if 'a' in ids else 9. if 'b' in ids else 10.)
            value=Evaluation({'L:F':-delay},{},delay,float(len(actions)),())
            self.cache[tuple(x.signature for x in actions)]=value
            if value.score<self.best[1].score: self.best=(list(actions),value)
            return value
    oracle=ToyOracle()
    policy=LiteraturePolicy('sms',3)
    policy.order_pool=lambda *args: [a,b]
    chosen=policy.solve(None,oracle,[a,b],[])
    assert {x.candidate_id for x in chosen}=={'a','b'}
    assert oracle.best[1].delay==0.
    assert policy.last_details.get('state_rounds',0)>=1


def test_feedback_repair_exact_cut_keeps_complementary_superset():
    from types import SimpleNamespace
    from dataclasses import replace
    ctrl,pool=live_controller()
    a=replace(pool[0],candidate_id='a',affected_resources=('task-a',),
              resource_requirements={'task:task-a':1.})
    b=replace(a,candidate_id='b',affected_resources=('task-b',),
              resource_requirements={'task:task-b':1.})
    base=Evaluation({'L:F':-10.},{},10.,0.,())
    oracle=SimpleNamespace(base=base,
        cache={(a.signature,):Evaluation({'L:F':-5.},{},5.,1.,()),
               (b.signature,):Evaluation({'L:F':-5.},{},5.,1.,())},
        model=SimpleNamespace(sim=SimpleNamespace(scn=SimpleNamespace(flights={'F':SimpleNamespace(weight=1.)}))))
    # Rejection of A alone must not forbid the useful joint bundle A+B.
    result=LiteraturePolicy._feedback_repair([a,b],[],oracle,'F',{('a',)})
    assert {x.candidate_id for x in result}=={'a','b'}
