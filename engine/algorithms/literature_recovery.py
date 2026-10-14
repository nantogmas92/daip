"""Explicit benchmark adaptations, NOT reproductions of published systems.

SMS: Chati et al., AAMAS 2024, Algorithm 1, state-conditioned min-sum.
F-LNS: Gok et al., Annals OR 2023, Algorithms 1-2, delayed-route destruction,
CP repair and simulation feedback. Here a bounded intervention neighbourhood
replaces unrestricted route rebuilding and fleet sizing.
OASS: Feng et al., Electronics 2026, Algorithms 1 and Eqs. 8-11.

All use the same current-state planning oracle, feasible action types, global
delay objective, commitment constraints and final acceptance rule. The main
comparison uses nominal forecasts: SMS is the one-scenario specialisation;
F-LNS does not reproduce the original stochastic robustness guarantee.
Full adaptation details and reasons are recorded in the comparison protocol.
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
import math
import random
import time
import numpy as np

from algorithms.algorithm1_propagation import PropagationEngine
from algorithms.algorithm2_candidates import COST, CandidateGenerator
from algorithms.counterfactual import apply_bundle
from algorithms.interaction import InteractionBuilder, InteractionGraph
from algorithms.algorithm3_selection import InterventionSelector
from core.messages import InterventionCandidate


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Evaluation:
    slacks: dict
    starts: dict
    delay: float
    cost: float
    new_violations: tuple
    work: int = 0
    worsened: tuple = ()

    @property
    def score(self):
        return (self.delay, self.cost)


def compatible(actions, budget=4):
    if len(actions) > budget:
        return False
    used = set()
    for action in actions:
        resources = {r for r in action.resource_requirements
                     if r.startswith(('task:', 'gse:', 'crew:'))}
        if used & resources:
            return False
        used |= resources
    return not any(InteractionBuilder._exclusive(a, b) for a, b in combinations(actions, 2))


def raw_candidate(generator, raw, index):
    kind, ents, theta = raw
    task = generator.model.sim.scn.tasks.get(ents[0])
    return InterventionCandidate(
        candidate_id='raw%d' % index, candidate_type=kind,
        affected_flights=generator._flights(ents),
        affected_tasks=tuple(e for e in ents if e in generator.model.sim.scn.tasks),
        affected_resources=ents, target_violation=(), predicted_leverage={},
        causal_path=(), side_effects={},
        resource_requirements=generator._resources(kind, ents, task),
        feasibility=True, uncertainty={}, theta=theta, cost=COST.get(kind, 1.0))


class PlanningOracle:
    """Count each joint model evaluation, including final validation.

    Results never mutate simulator assignments or the live graph. Cached hits
    are charged too, preventing an algorithm from evading the query limit.
    Wall time includes actual work; the budget is on model queries, not a
    purported identical number of elementary operations.
    """
    def __init__(self, model, limit=194, min_gain=.5):
        self.model = model
        self.limit = limit
        self.min_gain = min_gain
        self.queries = 0
        self.work = 0
        self.cache = {}
        # L- and S-variable ids of the planning graph, fixed for the oracle's
        # lifetime (one decision epoch: counterfactuals rewire factors but
        # never create variables).  Scanning every variable per query made
        # _read O(V) per evaluation.
        self._ls = None
        self.base = self._read((), {}, 0)
        self.best = ([], self.base)

    def _read(self, actions, before, work):
        g = self.model.g
        if self._ls is None:
            self._ls = ([v.vid for v in g.var.values() if v.kind == 'L'],
                        [k for k in g.var if k.startswith('S:')])
        lvids, sids = self._ls
        slacks = {vid: g.var[vid].value for vid in lvids}
        starts = {k: g.var[k].value for k in sids}
        scn = self.model.sim.scn

        def weight(key):
            # R0 exposes per-task slack only; it is weighted by the task's flight.
            ent = key.split(':', 1)[1]
            fl = scn.flights.get(ent) or scn.flights.get(scn.tasks[ent].flight)
            return fl.weight
        delay = sum(weight(k) * max(0., -v) for k, v in slacks.items())
        new = tuple(k for k, v in slacks.items() if before.get(k, -1.) >= 0 and v < -1e-6)
        worse = tuple(k for k, v in slacks.items() if k in before and before[k] < 0 and v < before[k] - 1e-6)
        return Evaluation(slacks, starts, delay, sum(a.cost for a in actions), new, work, worse)

    def evaluate(self, actions):
        if self.queries >= self.limit:
            raise BudgetExceeded()
        self.queries += 1
        actions = list(actions)
        if not compatible(actions):
            return Evaluation({}, {}, math.inf, math.inf, ('incompatible',))
        # Preserve order: the benchmark's synthetic edits are not commutative.
        key = tuple(a.signature for a in actions)
        if key in self.cache:
            return self.cache[key]
        g = self.model.g
        g.start_journal()
        try:
            seeds = apply_bundle(self.model, [(a.candidate_type, a.entities, a.theta) for a in actions])
            if g.ensure_scc().n_cyclic:
                # The controller's guarantee and experiment concern DAGs.
                val = Evaluation(dict(self.base.slacks), {}, math.inf, math.inf, ('cyclic',), 0)
            else:
                engine = PropagationEngine(self.model, mode='A1')
                res = engine.propagate(seeds, collect_messages=False) if seeds else None
                work = res.touched if res else 0
                val = self._read(actions, self.base.slacks, work)
                self.work += work
        finally:
            g.rollback()
        self.cache[key] = val
        if self.acceptable(val) and val.score < self.best[1].score:
            self.best = (actions, val)
        return val

    def acceptable(self, val):
        return not val.new_violations and self.base.delay - val.delay > self.min_gain


def minsum(unary, pairs, rounds=20, damping=.5):
    """Normalised damped min-sum, Eq. 11; tested against tree enumeration."""
    neighbors = [[] for _ in unary]
    edge = {}
    for (i, j), costs in pairs.items():
        neighbors[i].append(j); neighbors[j].append(i)
        edge[i, j] = costs; edge[j, i] = costs.T
    messages = {(i, j): np.zeros(len(unary[j])) for i, j in edge}
    for _ in range(rounds):
        updated = {}
        for i, j in messages:
            local = np.array(unary[i], dtype=float)
            for k in neighbors[i]:
                if k != j:
                    local += messages[k, i]
            value = (local[:, None] + edge[i, j]).min(axis=0)
            value -= value.min()
            updated[i, j] = damping * messages[i, j] + (1-damping) * value
        delta = max((np.max(np.abs(updated[k]-messages[k])) for k in messages), default=0.)
        messages = updated
        if delta < 1e-8:
            break
    beliefs = []
    for i, u in enumerate(unary):
        beliefs.append(np.asarray(u) + sum((messages[k, i] for k in neighbors[i]), start=np.zeros(len(u))))
    return [int(b.argmin()) for b in beliefs], beliefs


class OrderAware:
    """Feng et al. Eqs. 8-11, using delay and intervention cost objectives."""
    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.log_weights = np.zeros((3, 3))
        self.previous = 0

    def choose(self):
        row = self.log_weights[self.previous]
        p = np.exp(row - row.max())
        return self.rng.choices(range(3), weights=p, k=1)[0]

    def update(self, strategy, before, after):
        improved = sum(a < b - 1e-8 for a, b in zip(after, before))
        degraded = sum(a > b + 1e-8 for a, b in zip(after, before))
        delta = improved / len(before) if improved and not degraded else (
            -degraded / len(before) if degraded and not improved else 0.)
        self.log_weights[self.previous, strategy] += delta
        self.previous = strategy


class LiteraturePolicy:
    def __init__(self, name, seed):
        self.name = name
        self.rng = random.Random(seed)
        self.oass = OrderAware(seed)
        self.previous_performance = None
        self.last_details = {}

    def solve(self, controller, oracle, pool, targets):
        self.last_details = {}
        try:
            if self.name == 'sms':
                self._sms(controller, oracle, pool)
            elif self.name == 'flns':
                self._flns(controller, oracle, pool, targets)
            elif self.name == 'perho':
                self._perho(controller, oracle, pool)
            else:
                raise ValueError(self.name)
        except BudgetExceeded:
            self.last_details['budget_exhausted'] = True
        return oracle.best[0]

    @staticmethod
    def order_pool(controller, pool, strategy=0, rng=None):
        model, sim = controller.model, controller.sim
        if strategy == 2:
            result = list(pool); rng.shuffle(result); return result
        def key(action):
            task = sim.scn.tasks[action.entities[0]]
            flight = sim.scn.flights[task.flight]
            value = model.g.var.get('S:' + task.tid)
            earliest = value.value if value else flight.release
            return (earliest if strategy == 0 else flight.release, task.tid, action.cost, action.candidate_id)
        return sorted(pool, key=key)

    def _perho(self, controller, oracle, pool):
        strategy = self.oass.choose()
        ordered = self.order_pool(controller, pool, strategy, self.rng)
        # Same currently released waiting-task window for every method.
        # Retain running/completed commitments; consider a task at a time.
        chosen = []; current = oracle.base
        tasks = list(dict.fromkeys(a.entities[0] for a in ordered))
        for tid in tasks:
            alternatives = [a for a in ordered if a.entities[0] == tid and compatible(chosen+[a])]
            best_action, best_eval = None, current
            for a in alternatives:
                if oracle.queries >= oracle.limit-1:
                    break
                result = oracle.evaluate(chosen+[a])
                if not result.new_violations and result.score < best_eval.score:
                    best_action, best_eval = a, result
            if best_action is not None:
                chosen.append(best_action); current = best_eval
            if len(chosen) == 4 or oracle.queries >= oracle.limit-1:
                break
        performance = (current.delay / max(1., oracle.base.delay), current.cost)
        if self.previous_performance is not None:
            self.oass.update(strategy, self.previous_performance, performance)
        else:
            self.oass.previous = strategy
        self.previous_performance = performance
        self.last_details['ordering'] = ['ESTP','ATP','RSTP'][strategy]

    def _sms(self, controller, oracle, pool):
        # One action agent per task; complete pair costs over a bounded domain.
        # Unary/edge potentials are refreshed about the current joint state,
        # preserving the state/action iteration rather than static max-sum.
        grouped = {}
        for a in self.order_pool(controller, pool):
            grouped.setdefault(a.entities[0], []).append(a)
        # Round-robin evaluation prevents the first task consuming the budget.
        scored = {k: [] for k in grouped}
        rows = list(grouped.items())
        for rank in range(max((len(v) for v in grouped.values()), default=0)):
            for tid, actions in rows:
                if rank < len(actions) and oracle.queries < min(80, oracle.limit//2):
                    a = actions[rank]; val = oracle.evaluate([a])
                    scored[tid].append((val.delay, a.cost, a.candidate_id, a))
            if oracle.queries >= min(80, oracle.limit//2):
                break
        domains = [[None]+[r[3] for r in sorted(items)[:2]]
                   for tid, items in scored.items() if items]
        # Choose agents by their evaluated best option, not flight identifier.
        domains.sort(key=lambda d: oracle.cache[(d[1].signature,)].delay)
        domains = domains[:4]
        if not domains:
            return
        selected = [None]*len(domains)
        rounds = 0
        for outer in range(3):
            if oracle.queries >= oracle.limit-2:
                break
            context = [a for a in selected if a]
            base = oracle.evaluate(context)
            unary = []; single = {}
            for i, domain in enumerate(domains):
                costs = []
                for ai, action in enumerate(domain):
                    others = [a for j, a in enumerate(selected) if j != i and a]
                    bundle = others + ([action] if action else [])
                    val = oracle.evaluate(bundle)
                    single[i, ai] = val.delay
                    costs.append((val.delay-base.delay) + .001*sum(a.cost for a in bundle)
                                 if math.isfinite(val.delay) else 1e8)
                unary.append(np.array(costs))
            pairs = {}
            for i, j in combinations(range(len(domains)), 2):
                table = np.zeros((len(domains[i]), len(domains[j])))
                for ai, a in enumerate(domains[i]):
                    for bj, b in enumerate(domains[j]):
                        others = [x for k, x in enumerate(selected) if k not in (i,j) and x]
                        bundle = others+[x for x in (a,b) if x]
                        if not compatible(bundle):
                            table[ai,bj] = 1e8
                        else:
                            val = oracle.evaluate(bundle)
                            table[ai,bj] = val.delay-single[i,ai]-single[j,bj]+base.delay
                pairs[i,j] = np.nan_to_num(table, nan=1e8, posinf=1e8, neginf=-1e8)
            choices, beliefs = minsum(unary, pairs)
            proposal = [domains[i][a] for i,a in enumerate(choices)]
            # Resource conflict resolution: discard the least-supported choice.
            ranked = sorted(((beliefs[i][a]-beliefs[i][0], i, domains[i][a])
                             for i,a in enumerate(choices) if a), key=lambda x:x[:2])
            accepted = []
            for _, i, action in ranked:
                if compatible(accepted+[action]):
                    accepted.append(action)
                else:
                    proposal[i] = None
            oracle.evaluate(accepted)
            rounds += 1
            self.last_details['state_rounds'] = rounds
            if proposal == selected:
                break
            selected = proposal
        self.last_details['state_rounds'] = rounds

    def _flns(self, controller, oracle, pool, targets):
        # Delayed route and nearby same-type routes define each destroy region.
        # CP repairs the bounded action model; joint feedback enforces actual
        # improvement and rejects the exact failed bundle, not its supersets.
        model, sim = controller.model, controller.sim
        incumbent = []; rejected = set(); rounds = 0
        raw_by_id = {a.candidate_id:a for a in pool}
        for outer in range(4):
            delayed = sorted(sim.active, key=lambda f: model.g.var.get('L:'+f).value
                             if model.g.var.get('L:'+f) else 0)
            if not delayed:
                return
            focus = delayed[outer % len(delayed)]
            resources = {sim.scn.tasks[t].gse for t in sim.scn.flights[focus].tasks}
            resources |= {sim.scn.tasks[t].crew for t in sim.scn.flights[focus].tasks}
            nearby = [a for a in pool if focus in a.affected_flights or
                      any(r in resources for r in a.entities[1:])]
            others = [a for a in pool if a not in nearby]
            self.rng.shuffle(others)
            proposals = nearby + others
            cands = []
            for a in proposals:
                if oracle.queries >= oracle.limit-8 or len(cands) >= 32:
                    break
                val = oracle.evaluate([a])
                lev = {z.zid: val.slacks.get(z.zid, oracle.base.slacks[z.zid])-oracle.base.slacks[z.zid]
                       for z in targets}
                from dataclasses import replace
                cands.append(replace(a, predicted_leverage=lev, target_violation=tuple(lev)))
            if not cands:
                break
            # Preserve route commitments outside the destroyed neighbourhood.
            keep = [a for a in incumbent if a not in nearby]
            available = [a for a in cands if compatible(keep+[a])]
            if not available:
                continue
            for _ in range(3):
                bundle = self._feedback_repair(available, keep, oracle, focus, rejected)
                key = tuple(a.candidate_id for a in bundle)
                if not bundle or key in rejected:
                    break
                val = oracle.evaluate(bundle)
                if oracle.acceptable(val) and val.score <= oracle.best[1].score:
                    incumbent = [raw_by_id[a.candidate_id] for a in bundle]
                    break
                rejected.add(key)
            rounds += 1
            self.last_details['repair_rounds'] = rounds
        self.last_details['repair_rounds'] = rounds

    @staticmethod
    def _feedback_repair(available, keep, oracle, focus, rejected):
        """Independent CP repair with focal slack feedback and exact no-goods.

        Fits the publication's feedback idea to finite intervention choices:
        retain outside commitments, increase focal slack, and do not worsen
        predicted aggregate delay. Uses every flight, including satisfied ones.
        """
        from ortools.sat.python import cp_model
        actions = list(dict((a.candidate_id,a) for a in keep+available).values())
        model=cp_model.CpModel()
        y=[model.NewBoolVar('a%d'%i) for i in range(len(actions))]
        keep_ids={a.candidate_id for a in keep}
        for i,a in enumerate(actions):
            if a.candidate_id in keep_ids: model.Add(y[i]==1)
        model.Add(sum(y)<=4)
        for i,j in combinations(range(len(actions)),2):
            if not compatible([actions[i],actions[j]]): model.Add(y[i]+y[j]<=1)
        for rejected_ids in rejected:
            ids=set(rejected_ids)
            if ids.issubset({a.candidate_id for a in actions}):
                model.Add(sum((1-y[i]) if a.candidate_id in ids else y[i]
                              for i,a in enumerate(actions))>=1)
        terms=[]
        for zid,slack in oracle.base.slacks.items():
            coefficients=[]
            for a in actions:
                val=oracle.cache.get((a.signature,),oracle.base)
                coefficients.append(round(100*(val.slacks.get(zid,slack)-slack)))
            predicted=round(100*slack)+sum(c*y[i] for i,c in enumerate(coefficients))
            deficit=model.NewIntVar(0,100000000,'d'+zid)
            model.Add(deficit>=-predicted)
            if slack>=0: model.Add(predicted>=0)
            if zid=='L:'+focus:
                model.Add(predicted>=round(100*slack)+1)
            weight=oracle.model.sim.scn.flights[zid.split(':',1)[1]].weight
            terms.append(round(100*weight)*deficit)
        model.Add(sum(terms)<=round(10000*oracle.base.delay))
        model.Minimize(100000*sum(terms)+sum(round(100*a.cost)*y[i] for i,a in enumerate(actions)))
        solver=cp_model.CpSolver()
        solver.parameters.num_search_workers=1
        solver.parameters.random_seed=0
        solver.parameters.max_time_in_seconds=.2
        status=solver.Solve(model)
        if status not in (cp_model.OPTIMAL,cp_model.FEASIBLE): return []
        return [a for i,a in enumerate(actions) if solver.Value(y[i])]
