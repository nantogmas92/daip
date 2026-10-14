from types import SimpleNamespace as NS
import pytest
from optimization.cpsat import solve
from core.factors import CapacityFactor


def test_resource_release_excludes_hidden_event_times():
    from airport.observations import resource_release
    crew=NS(rate=1.2,workload_gain=0.,workload_state=0.,status='serving',current_assignment='t',availability_time=80.)
    task=NS(tid='t',state='running',start=10.,duration=20.,crew='c',overrun=0.,expedite=0.,finish=80.)
    sim=NS(t=12.,scn=NS(tasks={'t':task},crews={'c':crew}),enroute={})
    assert resource_release(sim,crew)==34.
    task.finish=800.;crew.availability_time=900.
    assert resource_release(sim,crew)==34.
    assert resource_release(sim,crew,lambda tid,kind:1.3)==pytest.approx(41.2)
    unit=NS(status='moving',current_assignment='t',speed=1.,availability_time=900.)
    sim.scn.apron=NS(travel_time=lambda *a,**kw:5.)
    sim.enroute={'t':NS(depart=10.,arrive=100.,legs=('e',))}
    assert resource_release(sim,unit)==15.
    sim.enroute['t'].arrive=1000.
    assert resource_release(sim,unit)==15.

def setup():
    return [NS(candidate_id=k, resource_requirements={}, harm=0., cost=0.) for k in ('a','b')], NS(edges={},kappa={})

def test_fractional_priorities_and_exact_rejection():
    c,g=setup()
    z=[NS(zid='z1',weight=1.,residual=1.9),NS(zid='z2',weight=1.6,residual=1.)]
    lev={('a','z1'):1.9,('b','z2'):1.}
    for staged in (False,True):
        chosen,status,_=solve(c,g,z,lambda u,k:lev.get((u.candidate_id,k),0.),1,2.,two_stage=staged)
        assert chosen==[0] and status=='optimal'
    z=[NS(zid='z',weight=1.,residual=2.)]
    chosen,_,_=solve(c,g,z,lambda u,k:1.,2,2.,nogoods=[('a',)])
    assert chosen==[0,1]

def test_calendar_finds_gap_and_respects_half_open_capacity():
    f=CapacityFactor('f',['a','b'],'out',capacity=1)
    f.peer_starts={'b':10.}
    g=NS(var={'a':NS(value=0.),'b':NS(value=20.)})
    assert f.project(g,0.)[0]==20.
    assert f.calendar_project(g,0.,5.)[0]==0.
    assert f.calendar_project(g,0.,15.)[0]==20.
    assert f.calendar_project(g,20.,15.)[0]==20.

def test_information_error_does_not_change_physical_draws():
    from airport.flights import AirportConfig
    from experiments.compare_literature import PairedSimulator,ComparisonController
    a=PairedSimulator(AirportConfig(seed=9100,n_flights=3))
    b=PairedSimulator(AirportConfig(seed=9100,n_flights=3,observation_sigma=.3))
    assert a._sample_noise('service','t',.1)==b._sample_noise('service','t',.1)
    ctrl=ComparisonController(b,'none')
    assert ctrl.model._forecast_multiplier('t','service')!=1.
    assert not ctrl.generator.feasible('REROUTE_GSE',('t',),2.)
    assert not b.execute('ADVANCE_POSITION_ADMISSION',('t',))

def test_departure_revokes_old_equipment_arrival_marker():
    from airport.flights import AirportConfig
    from experiments.compare_literature import PairedSimulator
    sim=PairedSimulator(AirportConfig(seed=9100,n_flights=6,horizon=65.))
    for fid in sim.scn.flights:sim._plan_flight(fid)
    unit=next(u for u in sim.scn.fleet.values() if len(u.sequence)>=2)
    old,new=unit.sequence[:2]
    sim.gse_at_stand[old]=0.
    sim._send_unit(unit,sim.scn.tasks[new])
    assert old not in sim.gse_at_stand
    assert unit.current_assignment==new and unit.status=='moving'

def test_cyclic_counterfactual_is_rejected_and_rolled_back(monkeypatch):
    import algorithms.literature_recovery as recovery
    from test_literature_comparison import live_controller
    ctrl,pool=live_controller();g=ctrl.model.g
    f=next(f for f in g.fac.values() if f.output.startswith('S:'))
    original=list(f.inputs);original_cond=g.cond
    def cycle(model,actions):
        model.g.rewire(f,[f.output])
        return [f.fid]
    monkeypatch.setattr(recovery,'apply_bundle',cycle)
    oracle=recovery.PlanningOracle(ctrl.model,2)
    val=oracle.evaluate([pool[0]])
    assert val.new_violations==('cyclic',) and not oracle.acceptable(val)
    assert f.inputs==original and g.cond is original_cond
