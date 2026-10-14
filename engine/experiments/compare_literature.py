"""Fresh paired comparison; see revision_audit/comparison/protocol.md.

python experiments/compare_literature.py --pilot
python experiments/compare_literature.py --seeds 20 --start-seed 100
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'causal-coordination-engine'))
from airport.flights import AirportConfig
from algorithms.control import ClosedLoopController, ControllerConfig
from algorithms.algorithm1_propagation import PropagationEngine
from algorithms.literature_recovery import (BudgetExceeded, LiteraturePolicy,
    PlanningOracle, raw_candidate, compatible)
from core.messages import DecisionRecord
from simulation.disturbances import build_plan
from simulation.event_engine import TurnaroundSimulator

METHODS = ('none','proposed','sms','flns','perho')


def keyed_seed(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str,parts)).encode()).digest()[:8], 'big')


class PairedSimulator(TurnaroundSimulator):
    def __init__(self, *args, **kwargs):
        self.noise_counts = defaultdict(int)
        super().__init__(*args, **kwargs)
        self.deadline = self.cfg.horizon + 600.

    def _sample_noise(self, kind, task_id, sigma):
        key = (kind,task_id)
        visit = self.noise_counts[key]
        self.noise_counts[key] += 1
        return random.Random(keyed_seed(self.cfg.seed,kind,task_id,visit)).gauss(1.,sigma)

    def _disturbance_rng(self, disturbance):
        return random.Random(keyed_seed(self.cfg.seed,'disturbance',disturbance.get('time',self.t),disturbance['kind']))


class ComparisonController(ClosedLoopController):
    def __init__(self, sim, method, eval_budget=194, action_budget=4):
        self.method = method
        self.query_limit = eval_budget
        cfg = ControllerConfig(propagation='A0' if method=='global' else 'A2',
                               candidates='C1' if method=='structural' else 'C2', budget=action_budget,
                               eval_budget=min(160,eval_budget-34), solver_time=.6,
                               intervene=method!='none')
        super().__init__(sim,cfg,'literature_'+method)
        original_feasible=self.generator.feasible
        def executable(kind,ents,theta=0.):
            if not original_feasible(kind,ents,theta): return False
            if kind=='ADVANCE_POSITION_ADMISSION':
                task=sim.scn.tasks[ents[0]]
                fl=sim.scn.flights[task.flight]
                position=sim.scn.stands[fl.stand].positions[task.position]
                holder=sim.scn.tasks.get(position.occupants[0]) if position.occupants else None
                return holder is not None and holder.state=='running' and holder.finish-sim.t>.7
            return True
        self.generator.feasible=executable
        self.policy = LiteraturePolicy(method, keyed_seed(sim.cfg.seed,method)) if method in ('sms','flns','perho') else None
        self.epoch_audit = []
        self.decision_ms = []
        self.applied_actions = []
        self.pending = None
        self.dispatch_audit = []

    def _apply(self, chosen):
        for action in chosen:
            feasible = self.generator.feasible(action.candidate_type,action.entities,action.theta)
            ok = feasible and self.sim.execute(action.candidate_type,action.entities,action.theta)
            self.applied_actions.append({'t':self.sim.t,'type':action.candidate_type,
                'entities':action.entities,'theta':action.theta,'cost':action.cost,'executed':bool(ok)})

    def _decide(self, ev):
        t0=time.perf_counter(); sim=self.sim
        rec=DecisionRecord(experiment_id=self.experiment_id,seed=sim.cfg.seed,
                           epoch=self.epoch,simulation_time=sim.t)
        dirty=list(dict.fromkeys(self.model.sync(sim.t)+self.model.refresh(sim.t)))
        res=self.engine.propagate(dirty) if dirty else None
        rec.factors_total=len(self.model.g.fac)
        rec.factors_touched=res.touched if res else 0
        rec.t_prop_ms=(time.perf_counter()-t0)*1000
        self._snapshot_predictions()
        targets=self.detector.detect(sim.t)
        if self.pending and sim.t >= self.pending[0]:
            due, requested, actions = self.pending
            oracle_due = PlanningOracle(self.model, 1)
            valid = all(self.generator.feasible(a.candidate_type,a.entities,a.theta) for a in actions)
            check = oracle_due.evaluate(actions) if valid else None
            accepted = bool(check and oracle_due.acceptable(check))
            if accepted:
                self._apply(actions)
                dirty=self.model.refresh(sim.t)
                if dirty: self.engine.propagate(dirty)
                targets=self.detector.detect(sim.t)
            self.dispatch_audit.append({'requested':requested,'due':due,'actual':sim.t,
                'accepted':accepted,'queries':oracle_due.queries})
            self.pending=None
        rec.n_violations=len(targets)
        rec.residual_pre=sum(z.weight*z.residual for z in targets)
        # Common gate independent of maintained sensitivity/binding records.
        signature=tuple(sorted(z.zid for z in targets))
        shock=ev.event_type in ('GSE_TRAVEL_DELAY','GSE_BREAKDOWN','CREW_UNAVAILABLE',
                               'SERVICE_OVERRUN','APRON_CONGESTION','STAND_RELEASE_DELAY','ARRIVAL_DELAY')
        search=bool(targets) and self.method!='none' and self.pending is None and (
            signature!=self._last_signature or shock or sim.t-self._last_replan>=6.)
        queries=work=0; chosen=[]; status='no_replan'; pool=[]
        if search:
            self._last_signature=signature; self._last_replan=sim.t
            oracle=PlanningOracle(self.model,self.query_limit)
            if self.method in ('proposed','structural','global'):
                chosen,status=self._proposed(targets,oracle,rec)
            else:
                raw=self.generator.enumerate_universe()
                pool=[raw_candidate(self.generator,a,i) for i,a in enumerate(raw)
                      if self.generator.feasible(*a)]
                rec.universe_size=len(raw)
                chosen=self.policy.solve(self,oracle,pool,targets)
                status=self.method
                rec.candidate_size=len(pool)
            # Final evaluation is shared, paid, and never bypassed.
            if chosen and oracle.queries<oracle.limit:
                check=oracle.evaluate(chosen)
                if not oracle.acceptable(check):
                    chosen=[]; status+=':rejected'
            elif chosen:
                # Cached jointly checked incumbent is already a valid check;
                # budget exhaustion cannot grant an unvalidated proposal.
                check=oracle.cache.get(tuple(a.signature for a in chosen))
                if check is None or not oracle.acceptable(check):
                    chosen=[]; status+=':unvalidated'
            if chosen and sim.cfg.action_lead_time:
                self.pending=(sim.t+sim.cfg.action_lead_time,sim.t,list(chosen))
            else:
                self._apply(chosen)
            if chosen:
                dirty=self.model.refresh(sim.t)
                if dirty: self.engine.propagate(dirty)
            post=self.detector.detect(sim.t)
            pre_ids={z.zid for z in targets}
            fresh=[z for z in post if z.zid not in pre_ids]
            sim.metrics.new_violations+=len(fresh)
            rec.new_violations=len(fresh)
            rec.bundle_size=len(chosen)
            rec.residual_post=sum(z.weight*z.residual for z in post)
            queries,work=oracle.queries,oracle.work
            assert queries<=self.query_limit
        elapsed=(time.perf_counter()-t0)*1000
        rec.t_total_ms=elapsed
        # Capture all overhead, including interaction construction and refresh.
        rec.t_cand_ms=max(0.,elapsed-rec.t_prop_ms)
        self.records.append(rec)
        self.decision_ms.append(elapsed)
        self.epoch_audit.append({'epoch':self.epoch,'t':sim.t,'search':search,
            'queries':queries,'work':work,'ms':elapsed,'actions':len(chosen),
            'status':status,'pool':len(pool),
            'live_cyclic_sccs':self.model.g.ensure_scc().n_cyclic,
            'details':self.policy.last_details if search and self.policy else {}})

    def _proposed(self, targets, oracle, rec):
        # Preserve Algorithms 1-3; replace only the measurement backend with the
        # same counted oracle and forward every violation for the shared objective.
        original=self.generator.counterfactual_leverage
        def counted(kind,ents,theta,violations):
            action=raw_candidate(self.generator,(kind,tuple(ents),theta),0)
            value=oracle.evaluate([action])
            lev={z.zid:value.slacks[z.zid]-oracle.base.slacks[z.zid] for z in violations}
            side={k:v-oracle.base.slacks[k] for k,v in value.slacks.items() if k not in lev}
            return lev,side,value.work
        self.generator.counterfactual_leverage=counted
        try:
            cands,stats=self.generator.generate(targets,want_universe=True)
        finally:
            self.generator.counterfactual_leverage=original
        rec.universe_size=stats.universe; rec.candidate_size=len(cands)
        if not cands: return [],'empty'
        # Count the same pairwise propagation calls used by the existing builder.
        original_kappa=self.interaction._kappa
        def counted_pair(a,b,violations):
            val=oracle.evaluate([a,b])
            if not val.slacks: return {}
            return {z.zid:val.slacks[z.zid]-oracle.base.slacks[z.zid]
                    -a.leverage_on(z.zid)-b.leverage_on(z.zid) for z in violations}
        self.interaction._kappa=counted_pair
        try:
            graph=self.interaction.build(cands,targets)
        finally:
            self.interaction._kappa=original_kappa
        nogoods=[]
        for _ in range(2):
            selected=self.selector.select(cands,graph,targets,nogoods)
            if not selected.chosen: return [],selected.status
            val=oracle.evaluate(selected.chosen)
            if oracle.acceptable(val): return selected.chosen,selected.status
            nogoods.append(tuple(a.candidate_id for a in selected.chosen))
        return [],'rejected'

    def summary(self):
        out=super().summary()
        values=sorted(self.decision_ms)
        searches=[r for r in self.epoch_audit if r['search']]
        out.update(mean_t_dec=statistics.mean(values) if values else 0.,
                   median_t_dec=statistics.median(values) if values else 0.,
                   median_search_ms=statistics.median(r['ms'] for r in searches) if searches else 0.,
                   p95_search_ms=sorted(r['ms'] for r in searches)[min(len(searches)-1,int(.95*len(searches)))] if searches else 0.,
                   intervention_cost=sum(a['cost'] for a in self.applied_actions if a['executed']),
                   delayed_rechecks=sum(r['queries'] for r in self.dispatch_audit),
                   delayed_rejections=sum(not r['accepted'] for r in self.dispatch_audit),
                   p95_t_dec=values[min(len(values)-1,int(.95*len(values)))] if values else 0.,
                   mean_search_ms=statistics.mean(r['ms'] for r in searches) if searches else 0.,
                   search_epochs=len(searches),total_queries=sum(r['queries'] for r in searches),
                   max_queries=max((r['queries'] for r in searches),default=0),
                   total_cf_factors=sum(r['work'] for r in searches),
                   failed_executions=sum(not a['executed'] for a in self.applied_actions))
        return out


def clean(value):
    if isinstance(value,dict): return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [clean(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value): return None
    return value


def run_and_save(job):
    method,seed,layout,pilot,budget,folder=job
    output=Path(folder)/f'{layout}_{seed}_{method}.json'
    if output.exists(): return {'existing':output.name}
    result,ctrl=run_one(method,seed,layout,n_flights=6 if pilot else 20,
                        horizon=65. if pilot else 150.,budget=budget)
    source_files=[Path(__file__),ROOT/'causal-coordination-engine/algorithms/literature_recovery.py',
                  ROOT/'causal-coordination-engine/simulation/event_engine.py']
    result['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    output.with_suffix('.epochs.json').write_text(json.dumps(clean(ctrl.epoch_audit),allow_nan=False),encoding='utf-8')
    output.with_suffix('.actions.json').write_text(json.dumps(ctrl.applied_actions),encoding='utf-8')
    return {k:result[k] for k in ['method','seed','layout','complete','delay_w_per_flight',
                      'on_time_rate','mean_t_dec','max_queries','failed_executions','wall_s']}


def run_one(method,seed,layout,n_flights=20,horizon=150.,budget=194):
    cfg=AirportConfig(seed=seed,layout=layout,n_flights=n_flights,horizon=horizon)
    plan=build_plan(seed,horizon,'mixed')
    sim=PairedSimulator(cfg,plan)
    controller=ComparisonController(sim,method,budget)
    start=time.perf_counter()
    result=controller.run()
    result.update(method=method,seed=seed,layout=layout,wall_s=time.perf_counter()-start,
                  complete=result['ready_flights']==n_flights,query_limit=budget,
                  config=asdict(cfg),disturbance_plan=plan,
                  final_time=sim.t,execution_digest=hashlib.sha256(
                      json.dumps(controller.applied_actions,sort_keys=True).encode()).hexdigest())
    return clean(result),controller


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--seeds',type=int,default=20)
    parser.add_argument('--start-seed',type=int,default=100)
    parser.add_argument('--methods',nargs='+',default=list(METHODS))
    parser.add_argument('--layouts',nargs='+',default=['stylised','geometry'])
    parser.add_argument('--budget',type=int,default=194)
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--tag',default=None)
    args=parser.parse_args()
    folder=ROOT/'results'/'literature_comparison'/(args.tag or ('pilot_v2' if args.pilot else 'primary'))
    folder.mkdir(parents=True,exist_ok=True)
    seeds=[9000] if args.pilot else range(args.start_seed,args.start_seed+args.seeds)
    jobs=[]
    for seed in seeds:
        for layout in (['stylised'] if args.pilot else args.layouts):
            order=list(args.methods)
            random.Random(keyed_seed(seed,layout,'order')).shuffle(order)
            for method in order:
                jobs.append((method,seed,layout,args.pilot,args.budget,str(folder)))
    if args.workers==1:
        for job in jobs: print(json.dumps(run_and_save(job)),flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures=[executor.submit(run_and_save,job) for job in jobs]
            for future in as_completed(futures): print(json.dumps(future.result()),flush=True)


if __name__=='__main__': main()
