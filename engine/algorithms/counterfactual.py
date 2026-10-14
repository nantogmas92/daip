"""The synthetic-intervention operator T_u used by Algorithms 2 and 3.

``apply_synthetic`` edits the factor graph *inside an open journal* and returns
the seed factors that Algorithm 1 must re-evaluate.  Because Algorithm 1
provides safe containment (Propositions A-C), evaluating a counterfactual is
local: its cost is proportional to the causal closure of the intervention, not
to the size of the apron.  The caller is responsible for opening the journal
and rolling it back, so no counterfactual ever reaches the live operational
state.

Operators approximate the corresponding commands in
``simulation/event_engine.execute`` within the planning model. Graph
recomputation equivalence checks internal maintenance correctness; it does
not establish equivalence to realised physical execution. Stochastic travel,
future dispatch and state-dependent disruptions can change realised gains.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from core.factors import (
    CapacityFactor,
    IncompatibilityFactor,
    MaxFactor,
    ServiceFactor,
    TravelFactor,
)

CREW_TRANSFER = 2.0


def _touch(g, fid: str) -> None:
    g.journal_fac(fid)


class BundleState:
    """Resource queues and positions as they stand after earlier bundle actions.

    Actions of one bundle are executed in order, so a later action must see
    the queue and position edits of the earlier ones: a task that joins a
    unit's queue after another task has left it must not be linked behind the
    departed task.  The live objects are never modified permanently.
    """

    def __init__(self, model):
        self.scn = model.sim.scn
        self.seqs: Dict[Tuple[str, Optional[str]], List[str]] = {}
        self.owner: Dict[Tuple[str, str], Optional[str]] = {}
        self.position: Dict[str, str] = {}

    def seq(self, kind: str, rid: Optional[str]) -> List[str]:
        key = (kind, rid)
        if key not in self.seqs:
            res = None
            if rid:
                res = self.scn.fleet.get(rid) if kind == "gse" else self.scn.crews.get(rid)
            self.seqs[key] = list(res.sequence) if res is not None else []
        return self.seqs[key]

    def owner_of(self, kind: str, tid: str) -> Optional[str]:
        if (kind, tid) in self.owner:
            return self.owner[(kind, tid)]
        t = self.scn.tasks.get(tid)
        return (t.gse if kind == "gse" else t.crew) if t is not None else None

    def move(self, kind: str, tid: str, rid: str) -> None:
        old = self.owner_of(kind, tid)
        if old and old != rid:
            seq = self.seq(kind, old)
            if tid in seq:
                seq.remove(tid)
        new = self.seq(kind, rid)
        if tid not in new:
            new.append(tid)
        self.owner[(kind, tid)] = rid


def _refresh_positions(model, flight_id: str, state: BundleState) -> List[str]:
    """Re-derive every capacity factor of one aircraft under the bundle's positions."""
    g = model.g
    scn = model.sim.scn
    fl = scn.flights.get(flight_id)
    if fl is None:
        return []
    saved = {tid: scn.tasks[tid].position for tid in state.position}
    seeds: List[str] = []
    try:
        for tid, pos in state.position.items():
            scn.tasks[tid].position = pos
        for tid, task in fl.tasks.items():
            fp = g.fac.get("fRp:" + tid)
            if not isinstance(fp, CapacityFactor):
                continue
            peers = model._position_peers(fl, task, model.now)
            cap = model._pos_capacity(fl, task)
            resource = "%s:%s" % (fl.stand, task.position)
            if peers != fp.peer_starts or cap != fp.capacity or fp.resource != resource:
                _touch(g, fp.fid)
                fp.peer_starts = peers
                fp.peer_start_vars = model._peer_start_vars(peers)
                fp.capacity = cap
                fp.resource = resource
                g.rewire(fp, model.capacity_inputs(tid, fp))
                seeds.append(fp.fid)
    finally:
        for tid, pos in saved.items():
            scn.tasks[tid].position = pos
    return seeds


def apply_synthetic(model, itype: str, entities: Sequence[str],
                    theta: float = 0.0,
                    state: Optional[BundleState] = None) -> List[str]:
    """Apply T_u to the (journalled) graph; return the seed factor ids."""
    g = model.g
    scn = model.sim.scn
    seeds: List[str] = []
    state = state or BundleState(model)

    try:
        if itype == "REROUTE_GSE":
            tid = entities[0]
            f = g.fac.get("feta:" + tid)
            if not isinstance(f, TravelFactor):
                return []
            _touch(g, f.fid)
            mv = model.sim.enroute.get(tid)
            unit = scn.fleet.get(scn.tasks[tid].gse) if tid in scn.tasks else None
            if mv is not None and unit is not None and model.rep.spatial:
                # a genuine alternative path, not a free time reduction
                alt = scn.apron.alternate_route(mv.src, mv.dst, mv.legs[:1])
                if alt:
                    f.legs = tuple(alt)
                    g.rewire(f, model._mu_inputs(alt)
                             + [x for x in f.inputs if not x.startswith("mu:")])
                else:
                    f.base -= max(0.0, theta)
            else:
                f.base -= max(0.0, theta)
            seeds.append(f.fid)

        elif itype in ("REASSIGN_GSE", "ACTIVATE_RESERVE_GSE"):
            tid, rid = entities[0], entities[1]
            f = g.fac.get("feta:" + tid)
            unit = scn.fleet.get(rid)
            task = scn.tasks.get(tid)
            if not isinstance(f, TravelFactor) or unit is None or task is None:
                return []
            _touch(g, f.fid)
            fl = scn.flights[task.flight]
            dst = scn.stands[fl.stand].node
            legs = scn.apron.route(unit.current_location, dst)
            f.legs = tuple(legs)
            f.speed = unit.speed / model._forecast_multiplier(tid, "travel")
            f.base = max(model.now,model.resource_release(unit))
            inputs = model._mu_inputs(legs)
            if model.rep.crossflight:
                prev = _sequence_predecessor(model, state.seq("gse", rid), tid)
                if prev:
                    inputs.append("C:" + prev)
            g.rewire(f, inputs)
            seeds.append(f.fid)
            old = state.owner_of("gse", tid)
            if old and old != rid:
                seeds += _release_successor(model, state.seq("gse", old), tid, "gse")
            state.move("gse", tid, rid)

        elif itype == "SWAP_GSE_SEQUENCE":
            a, b, rid = entities[0], entities[1], entities[2]
            unit = scn.fleet.get(rid)
            seq = state.seq("gse", rid)
            if unit is None or a not in seq or b not in seq:
                return []
            i, j = seq.index(a), seq.index(b)
            seq[i], seq[j] = seq[j], seq[i]
            seeds += _rewire_sequence(model, seq, "gse", min(i, j), max(i, j))

        elif itype == "REASSIGN_CREW":
            tid, cid = entities[0], entities[1]
            f = g.fac.get("fRh:" + tid)
            crew = scn.crews.get(cid)
            if not isinstance(f, MaxFactor) or crew is None:
                return []
            _touch(g, f.fid)
            inputs = []
            if model.rep.crossflight:
                prev = _sequence_predecessor(model, state.seq("crew", cid), tid)
                if prev:
                    inputs.append("C:" + prev)
            g.rewire(f, inputs)
            f.floor = max(model.now,model.resource_release(crew) + CREW_TRANSFER)
            seeds.append(f.fid)
            fc = g.fac.get("fC:" + tid)
            if isinstance(fc, ServiceFactor):
                _touch(g, fc.fid)
                fc.rate = crew.rate * model._forecast_multiplier(tid, "service")
                seeds.append(fc.fid)
            old = state.owner_of("crew", tid)
            if old and old != cid:
                seeds += _release_successor(model, state.seq("crew", old), tid, "crew")
            state.move("crew", tid, cid)

        elif itype == "SWAP_CREW_SEQUENCE":
            a, b, cid = entities[0], entities[1], entities[2]
            crew = scn.crews.get(cid)
            seq = state.seq("crew", cid)
            if crew is None or a not in seq or b not in seq:
                return []
            i, j = seq.index(a), seq.index(b)
            seq[i], seq[j] = seq[j], seq[i]
            seeds += _rewire_sequence(model, seq, "crew", min(i, j), max(i, j))

        elif itype == "CHANGE_SERVICE_POSITION":
            tid, pos = entities[0], entities[1]
            task = scn.tasks.get(tid)
            if task is None or not isinstance(g.fac.get("fRp:" + tid), CapacityFactor):
                return []
            # The move changes the peer calendar of every task at the old and
            # the new position, not only the moved task's own admission.
            state.position[tid] = pos
            seeds += _refresh_positions(model, task.flight, state)

        elif itype == "ADVANCE_POSITION_ADMISSION":
            tid = entities[0]
            fp = g.fac.get("fRp:" + tid)
            if not isinstance(fp, CapacityFactor):
                return []
            _touch(g, fp.fid)
            fp.peer_starts = {}
            g.rewire(fp, ["S0:" + tid])
            seeds.append(fp.fid)

        elif itype == "RESEQUENCE_TASKS":
            a, b = entities[0], entities[1]
            f0 = g.fac.get("fS0:" + b)
            if not isinstance(f0, MaxFactor) or ("C:" + a) not in g.var:
                return []
            _touch(g, f0.fid)
            if ("C:" + a) not in f0.inputs:
                g.rewire(f0, list(f0.inputs) + ["C:" + a])
            seeds.append(f0.fid)
            fsa = g.fac.get("fSf:" + a)
            if isinstance(fsa, IncompatibilityFactor) and ("C:" + b) in fsa.inputs:
                _touch(g, fsa.fid)
                g.rewire(fsa, [x for x in fsa.inputs if x != ("C:" + b)])
                seeds.append(fsa.fid)

    except (KeyError, IndexError, ValueError):
        return []
    return [s for s in dict.fromkeys(seeds) if s in g.fac]


def _sequence_predecessor(model, sequence: Sequence[str], tid: str) -> Optional[str]:
    """The task the resource would serve immediately before ``tid``.

    Must follow ``TurnaroundModel._resource_predecessor`` exactly.  A task
    reassigned to a unit is appended to its queue, so its predecessor is the
    last *unfinished* entry, not the literal queue tail: returning a completed
    tail would link the hypothetical graph to a different task than the live
    graph links after execution, and a cycle created by the live link would be
    missed by the check on the hypothetical one.
    """
    scn = model.sim.scn
    stop = list(sequence).index(tid) if tid in sequence else len(sequence)
    for j in range(stop - 1, -1, -1):
        prev = sequence[j]
        pt = scn.tasks.get(prev)
        if pt is None or pt.state == "done":
            continue
        return prev if ("C:" + prev) in model.g.var else None
    return None


def _release_successor(model, sequence: Sequence[str], tid: str,
                       kind: str) -> List[str]:
    """Re-link the old queue once ``tid`` leaves it.

    The first unfinished entry after ``tid`` no longer waits for ``tid``; after
    execution the live model links it to ``tid``'s own predecessor, so the
    counterfactual must do the same or validation scores a different graph from
    the one that exists after the command is carried out.
    """
    if tid not in sequence:
        return []
    seq = [x for x in sequence if x != tid]
    idx = list(sequence).index(tid)
    scn = model.sim.scn
    for nxt in list(sequence)[idx + 1:]:
        nt = scn.tasks.get(nxt)
        if nt is None or nt.state == "done":
            continue
        pos = seq.index(nxt)
        return _rewire_sequence(model, seq, kind, pos, pos - 1)
    return []


def _rewire_sequence(model, seq: Sequence[str], kind: str,
                     lo: int, hi: int) -> List[str]:
    """Re-link the readiness terms whose predecessor identity a swap changes.

    Only the swapped span changes: positions before ``lo`` are untouched, and
    the first position after ``hi`` is the last whose predecessor identity is
    altered.  Bounding the rewire this way is what keeps the counterfactual
    local rather than re-linking the whole resource queue.
    """
    g = model.g
    scn = model.sim.scn
    seeds: List[str] = []
    hi = min(hi + 1, len(seq) - 1)
    for pos in range(lo, hi + 1):
        tid = seq[pos]
        fid = ("feta:" + tid) if kind == "gse" else ("fRh:" + tid)
        f = g.fac.get(fid)
        if f is None:
            continue
        _touch(g, fid)
        prev = None
        for j in range(pos - 1, -1, -1):
            cand = seq[j]
            ct = scn.tasks.get(cand)
            if ct is None or ct.state == "done":
                continue
            if ("C:" + cand) in g.var:
                prev = cand
            break
        if kind == "gse" and isinstance(f, TravelFactor):
            keep = model._mu_inputs(f.legs)
            g.rewire(f, keep + (["C:" + prev] if prev and model.rep.crossflight else []))
        elif isinstance(f, MaxFactor):
            g.rewire(f, ["C:" + prev] if prev and model.rep.crossflight else [])
        seeds.append(fid)
    return seeds


def apply_bundle(model, actions: Sequence[Tuple[str, Sequence[str], float]]
                 ) -> List[str]:
    """Apply a bundle in execution order, each action seeing the earlier ones."""
    seeds: List[str] = []
    state = BundleState(model)
    for (itype, ents, theta) in actions:
        seeds += apply_synthetic(model, itype, ents, theta, state)
    return list(dict.fromkeys(seeds))
