"""Discrete-event turnaround simulator: the ground truth X_{t+1} = F(X_t, U_t, W_t).

The simulator is deterministic given a seed, supports full event logging and
exact replay, and is completely independent of the controller's
representation: every representation variant is executed against the *same*
realised disturbance stream, so paired comparisons with common random numbers
are valid.

Algorithms 1-3 are deliberately *not* embedded in the physical transition
logic.  The engine exposes ``step`` (advance to the next event) and
``execute`` (apply one validated intervention) and knows nothing else about
the controller.  The dispatcher below is the same baseline policy under every
representation, so representation fidelity affects only corrective control.
"""

from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from airport.flights import AirportConfig, TurnaroundScenario
from airport.stands import position_id
from core.messages import AirportOperationalEvent

from . import disturbances as dist
from . import service_execution as sx
from .movement import Movement, edge_loads, plan_movement

EV_PRIORITY = {
    "FLIGHT_RELEASE": 0,
    "GSE_ARRIVAL": 1,
    "CREW_ARRIVAL": 2,
    "TASK_DONE": 3,
    "STAND_RELEASE": 4,
    "DISTURBANCE": 5,
    "CONGESTION_UPDATE": 6,
    "CONTROL_TICK": 7,
    "RESTORE": 8,
}

# Events after which the decision controller is invoked.
CONTROL_EVENTS = {"FLIGHT_RELEASE", "TASK_DONE", "GSE_ARRIVAL",
                  "DISTURBANCE", "CONTROL_TICK", "STAND_RELEASE"}

CREW_TRANSFER = 2.0        # minutes for a crew to reach a stand once released
SPILLOVER = 0.75           # share of an expedite paid back by other vehicles
PREPOSITION_LEAD = 1.0     # minutes a unit may arrive before its task can start


@dataclass
class Metrics:
    ready_flights: int = 0
    delay_w: float = 0.0
    delay: float = 0.0
    delays: List[float] = field(default_factory=list)
    on_time: int = 0
    gse_distance: float = 0.0
    gse_deadhead: float = 0.0
    crew_wait: float = 0.0
    stand_release_delay: float = 0.0
    congestion_delay: float = 0.0
    interventions: int = 0
    new_violations: int = 0
    recovery_times: List[float] = field(default_factory=list)
    spill: float = 0.0
    n_movements: int = 0


class TurnaroundSimulator:
    """Event-driven apron execution engine."""

    def __init__(self, cfg: AirportConfig,
                 disturbance_plan: Optional[Sequence[Dict[str, object]]] = None):
        self.cfg = cfg
        self.scn = TurnaroundScenario(cfg)
        self.rng = random.Random(cfg.seed * 104729 + 7)
        self.t = 0.0
        self.seq = 0
        self.event_id = 0
        self.queue: List[Tuple[float, int, int, str, tuple]] = []
        self.events: List[AirportOperationalEvent] = []
        self.metrics = Metrics()

        self.active: Dict[str, object] = {}          # released, not yet ready
        self.enroute: Dict[str, Movement] = {}       # task id -> vehicle movement
        self.movements: List[Movement] = []
        self.gse_at_stand: Dict[str, float] = {}     # task id -> vehicle arrival
        self.crew_ready_at: Dict[str, float] = {}    # task id -> crew arrival
        self.position_free_at: Dict[str, float] = {}
        self.block_since: Dict[str, float] = {}      # queue-wait bookkeeping
        self.task_cause: Dict[str, str] = {}         # ground-truth binding class
        self.flight_cause: Dict[str, str] = {}
        self.last_disturbance: Optional[str] = None
        self.disturbance_plan = list(disturbance_plan or [])
        # controller-side reporting of disturbances (airport/observations.py)
        self.report_time: Dict[tuple, float] = {}
        self.outage_onset: Dict[tuple, float] = {}
        self.hidden_overrun: Dict[str, List[Tuple[float, float]]] = {}
        # Flights are released up to ``horizon``; each turnaround then needs its
        # own span to complete, so the run guard must extend well beyond it.
        # Truncating at the release horizon would silently drop the last bank
        # and make tardiness sums incomparable between configurations.
        self.deadline = cfg.horizon + 300.0

        for fl in self.scn.flights.values():
            self._push(fl.release, "FLIGHT_RELEASE", (fl.flight_id,))
        for d in self.disturbance_plan:
            self._push(float(d["time"]), "DISTURBANCE", (str(d["kind"]), d))
        self._push(0.0, "CONGESTION_UPDATE", ())
        self._push(0.0, "CONTROL_TICK", ())

    # ------------------------------------------------------------------
    # queue helpers
    # ------------------------------------------------------------------
    def _push(self, t: float, kind: str, payload: tuple) -> None:
        self.seq += 1
        heapq.heappush(self.queue,
                       (t, EV_PRIORITY.get(kind, 5), self.seq, kind, payload))

    def _record(self, kind: str, flight: str = "", task: str = "",
                resources: Tuple[str, ...] = (), location: str = "",
                delta: Optional[Dict[str, float]] = None
                ) -> AirportOperationalEvent:
        self.event_id += 1
        ev = AirportOperationalEvent(
            event_id=self.event_id, simulation_time=self.t, event_type=kind,
            flight_id=flight, task_id=task, resource_ids=resources,
            location_id=location, observed_delta=dict(delta or {}),
            seed=self.cfg.seed)
        self.events.append(ev)
        return ev

    # ------------------------------------------------------------------
    # planning: commit every task of a released flight to a unit and a crew
    # ------------------------------------------------------------------
    def _plan_flight(self, fid: str) -> None:
        """Greedy earliest-completion commitment.

        Committing the whole turnaround at release time (rather than deciding
        at the last moment) is what makes the resource *sequences* exist, and
        the sequences are what couple flights: a unit committed to F1 and then
        F2 transmits F1's delay to F2 even though F2 was never disturbed.
        """
        scn = self.scn
        fl = scn.flights[fid]
        stand_node = scn.stands[fl.stand].node
        order = sorted(fl.tasks.values(),
                       key=lambda x: (len(x.predecessors), x.tid))
        for task in order:
            if task.gse_class is not None:
                unit = self._best_unit(task, stand_node)
                if unit is not None:
                    task.gse = unit.resource_id
                    unit.sequence.append(task.tid)
            if task.crew_skill is not None:
                crew = self._best_crew(task)
                if crew is not None:
                    task.crew = crew.crew_id
                    crew.sequence.append(task.tid)
            task.state = "ready" if not task.predecessors else "waiting"
        self._reorder_queues(fl)

    def _reorder_queues(self, fl) -> None:
        """Sort every touched resource queue by the global scheduling key.

        Independent greedy commitment of equipment and crews can otherwise
        produce queues whose orders disagree, and disagreeing FIFO queues
        deadlock: unit A waits for a task held by crew B while crew B waits for
        a task held by unit A.  Because the key is consistent with precedence,
        ordering every queue by it makes the wait-for relation strictly
        decreasing and therefore acyclic.
        """
        scn = self.scn
        touched = set()
        for task in fl.tasks.values():
            if task.gse:
                touched.add(("gse", task.gse))
            if task.crew:
                touched.add(("crew", task.crew))
        for (kind, rid) in sorted(touched):
            res = scn.fleet.get(rid) if kind == "gse" else scn.crews.get(rid)
            if res is None:
                continue
            fixed, movable = [], []
            for tid in res.sequence:
                t = scn.tasks.get(tid)
                if t is None or t.state in ("running", "done"):
                    fixed.append(tid)
                else:
                    movable.append(tid)
            movable.sort(key=scn.rank)
            res.sequence = fixed + movable

    def _unit_free_estimate(self, unit) -> float:
        """When the unit is expected to finish everything already committed."""
        from airport.observations import resource_release
        t = max(self.t,resource_release(self, unit))
        for tid in unit.sequence:
            tk = self.scn.tasks.get(tid)
            if tk is None or tk.state in ("done", "running"):
                continue
            t += tk.duration + 3.0        # service plus a nominal repositioning
        return t

    def _best_unit(self, task, stand_node: str):
        scn = self.scn
        best, best_cost = None, 1e18
        for unit in scn.fleet.values():
            if not scn.gse_capable(unit, task) or unit.status == "down":
                continue
            if unit.reserve:
                continue                   # reserves are an explicit intervention
            free = self._unit_free_estimate(unit)
            legs = scn.apron.route(unit.current_location, stand_node)
            cost = free + scn.apron.nominal_travel(legs, unit.speed)
            if cost < best_cost:
                best, best_cost = unit, cost
        return best

    def _crew_free_estimate(self, crew) -> float:
        from airport.observations import resource_release
        t = max(self.t,resource_release(self, crew))
        for tid in crew.sequence:
            tk = self.scn.tasks.get(tid)
            if tk is None or tk.state in ("done", "running"):
                continue
            t += tk.duration
        return t

    def _best_crew(self, task):
        scn = self.scn
        best, best_cost = None, 1e18
        for crew in scn.crews.values():
            if not scn.crew_qualified(crew, task) or crew.status == "unavailable":
                continue
            cost = self._crew_free_estimate(crew) + 0.5 * len(crew.sequence)
            if cost < best_cost:
                best, best_cost = crew, cost
        return best

    # ------------------------------------------------------------------
    # dispatch: move idle units toward the head of their sequence
    # ------------------------------------------------------------------
    def _imminent(self, unit, task) -> bool:
        """Whether ``unit`` should pre-position for ``task`` now.

        Every predecessor must be done or running, and the running ones must
        be expected (from observable nominal durations) to finish no later than
        the unit's current travel time plus ``PREPOSITION_LEAD``.  A unit
        therefore never waits on a service that has not started, so
        pre-positioning cannot monopolise a vehicle indefinitely.
        """
        scn = self.scn
        remaining = 0.0
        for p in tuple(task.predecessors) + tuple(task.holds):
            pt = scn.tasks.get(p)
            if pt is None or pt.state == "done":
                continue
            if pt.state != "running":
                return False
            crew = scn.crews.get(pt.crew) if pt.crew else None
            nominal = scn.service_duration(pt, crew) - pt.expedite
            remaining = max(remaining, pt.start + nominal - self.t)
        if remaining <= 0.0:
            return True
        dst = scn.stands[scn.flights[task.flight].stand].node
        legs = scn.apron.route(unit.current_location, dst)
        travel = scn.apron.travel_time(legs, unit.speed, congested=True)
        return remaining <= travel + PREPOSITION_LEAD

    def _dispatch(self) -> None:
        scn = self.scn
        for unit in scn.fleet.values():
            if unit.status in ("moving", "serving", "down"):
                continue
            nxt = None
            for tid in unit.sequence:
                tk = scn.tasks.get(tid)
                if tk is not None and tk.state not in ("done", "running"):
                    # A task whose precedence is far from satisfied must not
                    # monopolise a vehicle at its stand; one whose remaining
                    # predecessors are about to finish is worth pre-positioning.
                    if not self._imminent(unit, tk):
                        continue
                    nxt = tk
                    break
            if nxt is None or nxt.tid in self.gse_at_stand or nxt.tid in self.enroute:
                continue
            if nxt.flight not in self.active:
                continue
            self._send_unit(unit, nxt)
        for crew in scn.crews.values():
            if crew.status == "unavailable":
                continue
            for tid in crew.sequence:
                tk = scn.tasks.get(tid)
                if tk is None or tk.state in ("done", "running"):
                    continue
                if tk.flight in self.active and tid not in self.crew_ready_at:
                    self.crew_ready_at[tid] = max(self.t, crew.availability_time) \
                        + CREW_TRANSFER
                break

    def _send_unit(self, unit, task, detour: Sequence[str] = ()) -> None:
        scn = self.scn
        # An arrival marker is a current-location observation, not a permanent
        # licence to start this unit's earlier tasks at a different stand.
        for tid in list(self.gse_at_stand):
            previous = scn.tasks.get(tid)
            if previous is not None and previous.gse == unit.resource_id:
                self.gse_at_stand.pop(tid, None)
        dst = scn.stands[scn.flights[task.flight].stand].node
        mv = plan_movement(scn.apron, unit, dst, self.t, task.tid,
                           productive=True, detour=detour)
        noise = max(0.6, self._sample_noise("travel", task.tid, self.cfg.eta_noise))
        mv.arrive = mv.depart + mv.duration * noise
        unit.status = "moving"
        unit.current_assignment = task.tid
        unit.route = mv.legs
        self.enroute[task.tid] = mv
        self.movements.append(mv)
        self.metrics.n_movements += 1
        dist_km = scn.apron.path_length(mv.legs)
        self.metrics.gse_distance += dist_km
        self._push(mv.arrive, "GSE_ARRIVAL", (unit.resource_id, task.tid))
        self._record("GSE_DISPATCHED", task.flight, task.tid,
                     (unit.resource_id,), dst, {"eta": mv.arrive})

    def queue_block_since(self, tid: str) -> float:
        """When this task first became blocked purely by a resource queue."""
        return self.block_since.setdefault(tid, self.t)

    # ------------------------------------------------------------------
    # service execution
    # ------------------------------------------------------------------
    def _pump(self) -> None:
        """Start every task whose max-factor terms are all satisfied."""
        scn = self.scn
        progressed = True
        guard = 0
        while progressed and guard < 6:
            guard += 1
            progressed = False
            for fid in sorted(self.active):
                fl = scn.flights[fid]
                for tid in sorted(fl.tasks):
                    task = fl.tasks[tid]
                    if task.state in ("running", "done"):
                        continue
                    if task.predecessors and all(
                            scn.tasks[p].state == "done" for p in task.predecessors):
                        task.state = "ready"
                    el = sx.evaluate(self, task)
                    if not el.ready:
                        continue
                    self._start_task(task, el)
                    progressed = True

    def _start_task(self, task, el) -> None:
        scn = self.scn
        fl = scn.flights[task.flight]
        crew = scn.crews.get(task.crew) if task.crew else None
        unit = scn.fleet.get(task.gse) if task.gse else None

        self.task_cause[task.tid] = el.cause
        if crew is not None:
            arrive = self.crew_ready_at.get(task.tid, self.t)
            self.metrics.crew_wait += max(0.0, self.t - arrive)
            crew.waiting += max(0.0, self.t - arrive)

        dur = scn.service_duration(task, crew)
        dur *= max(0.55, self._sample_noise("service", task.tid, self.cfg.duration_noise))
        dur = max(0.5, dur - task.expedite)

        task.state = "running"
        task.start = self.t
        task.finish = self.t + dur
        pos = scn.stands[fl.stand].positions.get(task.position)
        if pos is not None:
            pos.occupants.append(task.tid)
            pos.reservations.append((task.tid, self.t, task.finish))
        if unit is not None:
            unit.status = "serving"
            unit.current_assignment = task.tid
            unit.availability_time = task.finish
            unit.busy_time += dur
        if crew is not None:
            crew.status = "serving"
            crew.current_assignment = task.tid
            crew.availability_time = task.finish
            crew.worked += dur
            if crew.workload_gain > 0.0:
                crew.workload_state = min(1.0, crew.workload_state
                                          + crew.workload_gain * dur)
        self._push(task.finish, "TASK_DONE", (task.tid,))
        self._record("TASK_START", task.flight, task.tid,
                     tuple(x for x in (task.gse, task.crew) if x),
                     position_id(fl.stand, task.position),
                     {"duration": dur, "cause": 0.0})

    def _finish_task(self, tid: str) -> None:
        scn = self.scn
        task = scn.tasks.get(tid)
        if task is None or task.state == "done":
            return
        fl = scn.flights[task.flight]
        task.state = "done"
        pos = scn.stands[fl.stand].positions.get(task.position)
        if pos is not None and tid in pos.occupants:
            pos.occupants.remove(tid)
            self.position_free_at[position_id(fl.stand, task.position)] = self.t
        unit = scn.fleet.get(task.gse) if task.gse else None
        if unit is not None:
            unit.status = "idle"
            unit.current_assignment = None
            unit.availability_time = self.t
            unit.current_location = scn.stands[fl.stand].node
        crew = scn.crews.get(task.crew) if task.crew else None
        if crew is not None:
            crew.status = "idle"
            crew.current_assignment = None
            crew.availability_time = self.t
            crew.current_location = scn.stands[fl.stand].node
        self.enroute.pop(tid, None)
        self._record("TASK_COMPLETE", task.flight, tid, (), "",
                     {"finish": self.t})
        self._check_ready(task.flight)

    def _check_ready(self, fid: str) -> None:
        scn = self.scn
        fl = scn.flights.get(fid)
        if fl is None or fid not in self.active:
            return
        if any(fl.tasks[t].state != "done" for t in fl.tasks):
            return
        ready = max(fl.tasks[t].finish for t in fl.terminal
                    if not math.isnan(fl.tasks[t].finish))
        fl.ready = ready
        self.active.pop(fid, None)
        lateness = max(0.0, ready - fl.target)
        m = self.metrics
        m.ready_flights += 1
        m.delay += lateness
        m.delay_w += fl.weight * lateness
        m.delays.append(lateness)
        m.on_time += int(lateness <= 1e-9)
        self.flight_cause[fid] = sx.flight_binding_cause(self, fl)
        st = scn.stands[fl.stand]
        st.occupant = None
        st.released_at = self.t
        fl.stand_released = self.t
        self._push(self.t, "STAND_RELEASE", (fid,))
        self._record("AIRCRAFT_READY", fid, "", (), fl.stand,
                     {"ready": ready, "lateness": lateness})

    # ------------------------------------------------------------------
    # main step
    # ------------------------------------------------------------------
    def step(self) -> Optional[AirportOperationalEvent]:
        if not self.queue:
            return None
        t, _, _, kind, payload = heapq.heappop(self.queue)
        self.t = max(self.t, t)
        ev: Optional[AirportOperationalEvent] = None
        scn = self.scn

        if kind == "FLIGHT_RELEASE":
            fid = payload[0]
            fl = scn.flights[fid]
            st = scn.stands[fl.stand]
            # the aircraft cannot go on-block until the previous occupant has
            # cleared the stand and stand access is open: this is the physical
            # channel through which a stand-release overrun (D8) reaches a
            # downstream flight that suffered no disturbance of its own
            if (st.occupant not in (None, fid)) or st.access_state != "open":
                fl.release = self.t + 1.0
                self._push(fl.release, "FLIGHT_RELEASE", (fid,))
                self._dispatch()
                self._pump()
                return None
            self.active[fid] = fl
            fl.started = True
            scn.stands[fl.stand].occupant = fid
            self._plan_flight(fid)
            ev = self._record("FLIGHT_ON_BLOCK", fid, "", (), fl.stand,
                              {"release": fl.release, "target": fl.target})

        elif kind == "GSE_ARRIVAL":
            rid, tid = payload
            unit = scn.fleet.get(rid)
            mv = self.enroute.get(tid)
            task = scn.tasks.get(tid)
            if unit is not None and task is not None and task.state != "done":
                unit.status = "idle"
                unit.current_location = mv.dst if mv else unit.current_location
                self.gse_at_stand[tid] = self.t
                if mv is not None:
                    self.metrics.congestion_delay += max(
                        0.0, mv.duration
                        - scn.apron.nominal_travel(mv.legs, unit.speed))
                ev = self._record("GSE_ARRIVED", task.flight, tid, (rid,),
                                  unit.current_location, {"t": self.t})
            self.enroute.pop(tid, None)

        elif kind == "TASK_DONE":
            self._finish_task(payload[0])
            ev = self.events[-1] if self.events else None

        elif kind == "STAND_RELEASE":
            fid = payload[0]
            fl = scn.flights.get(fid)
            if fl is not None:
                slip = max(0.0, self.t - fl.target)
                self.metrics.stand_release_delay += slip
                ev = self._record("STAND_RELEASE", fid, "", (), fl.stand,
                                  {"slip": slip})

        elif kind == "DISTURBANCE":
            ev = self._apply_disturbance(payload[1])

        elif kind == "CONGESTION_UPDATE":
            loads = edge_loads([m for m in self.enroute.values()], self.t)
            scn.apron.set_loads(loads)
            if self.t < self.deadline:
                self._push(self.t + 2.0, "CONGESTION_UPDATE", ())

        elif kind == "CONTROL_TICK":
            if self.t < self.deadline:
                self._push(self.t + 4.0, "CONTROL_TICK", ())
            ev = self._record("CONTROL_TICK")

        elif kind == "RESTORE":
            ev = self._restore(payload[0])

        self._dispatch()
        self._pump()
        if kind in CONTROL_EVENTS and ev is not None:
            return ev
        return None

    # ------------------------------------------------------------------
    # disturbance families
    # ------------------------------------------------------------------
    def arrival_margin(self, tid: str) -> float:
        """How much a vehicle arrival may slip before it becomes binding."""
        task = self.scn.tasks.get(tid)
        mv = self.enroute.get(tid)
        if task is None or mv is None:
            return 0.0
        el = sx.evaluate(self, task)
        others = [v for k, v in el.terms.items()
                  if k != sx.CAUSE_GSE and v > -1e17]
        if not others:
            return 0.0
        return max(0.0, max(others) - mv.arrive)

    def _delay_arrival(self, tid: str, mag: float) -> None:
        mv = self.enroute.get(tid)
        if mv is None:
            return
        mv.arrive += mag
        newq = []
        for entry in self.queue:
            t, pr, sq, kind, payload = entry
            if kind == "GSE_ARRIVAL" and payload and payload[1] == tid:
                newq.append((t + mag, pr, sq, kind, payload))
            else:
                newq.append(entry)
        self.queue = newq
        heapq.heapify(self.queue)

    def _sample_noise(self, kind: str, task_id: str, sigma: float) -> float:
        """Default campaign stream; comparison subclasses can key draws by task."""
        return self.rng.gauss(1.0, sigma)

    def _disturbance_rng(self, disturbance):
        return self.rng

    def _apply_disturbance(self, d) -> Optional[AirportOperationalEvent]:
        kind = str(d["kind"])
        self.last_disturbance = kind
        mag = float(d.get("magnitude", 6.0))
        scn = self.scn
        rng = self._disturbance_rng(d)

        if kind in ("D1", "D2", "D3"):
            if kind == "D3":
                pick = dist.pick_switchable(self, rng)
                if pick is None:
                    return None
                tid, mag = pick
            else:
                tid = dist.pick_enroute(self, binding=(kind == "D2"), rng=rng)
                if tid is None:
                    return None
                if kind == "D1":
                    mag = min(mag, max(0.5, 0.7 * self.arrival_margin(tid)))
            task = scn.tasks.get(tid)
            self._delay_arrival(tid, mag)
            return self._record("GSE_TRAVEL_DELAY",
                                task.flight if task else "", tid,
                                (task.gse,) if task and task.gse else (),
                                "", {"delay": mag})

        if kind == "D4":
            live = [u for u in scn.fleet.values()
                    if u.status == "idle" and not u.reserve and u.sequence]
            if not live:
                return None
            unit = rng.choice(sorted(live, key=lambda u: u.resource_id))
            unit.status = "down"
            unit.availability_time = max(unit.availability_time, self.t + 12.0 + mag)
            self._report(("gse", unit.resource_id))
            self._push(self.t + 12.0 + mag, "RESTORE", (("gse", unit.resource_id),))
            return self._record("GSE_BREAKDOWN", "", "", (unit.resource_id,),
                                unit.current_location, {"down": 1.0})

        if kind == "D5":
            live = [c for c in scn.crews.values()
                    if c.status == "idle" and c.sequence]
            if not live:
                return None
            crew = rng.choice(sorted(live, key=lambda c: c.crew_id))
            crew.status = "unavailable"
            crew.availability_time = max(crew.availability_time, self.t + mag)
            self._report(("crew", crew.crew_id))
            self._push(self.t + mag, "RESTORE", (("crew", crew.crew_id),))
            return self._record("CREW_UNAVAILABLE", "", "", (crew.crew_id,),
                                "", {"delay": mag})

        if kind == "D6":
            running = [t for t in scn.tasks.values() if t.state == "running"]
            if not running:
                return None
            task = rng.choice(sorted(running, key=lambda x: x.tid))
            task.overrun += mag
            task.finish += mag
            report = math.inf if self.cfg.info_missing else self.t + self.cfg.info_delay
            if report > self.t:
                self.hidden_overrun.setdefault(task.tid, []).append((report, mag))
            newq = []
            for entry in self.queue:
                t, pr, sq, k, payload = entry
                if k == "TASK_DONE" and payload and payload[0] == task.tid:
                    newq.append((t + mag, pr, sq, k, payload))
                else:
                    newq.append(entry)
            self.queue = newq
            heapq.heapify(self.queue)
            return self._record("SERVICE_OVERRUN", task.flight, task.tid, (),
                                "", {"overrun": mag})

        if kind == "D7":
            eids = sorted(scn.apron.edges)
            eid = rng.choice(eids)
            edge = scn.apron.edges[eid]
            old = edge.capacity
            edge.capacity = max(1.0, old * 0.4)
            scn.apron.set_loads(dict(scn.apron.load))
            self._push(self.t + 20.0 + mag, "RESTORE", (("edge", eid, old),))
            return self._record("APRON_CONGESTION", "", "", (), eid,
                                {"capacity": edge.capacity})

        if kind == "D8":
            done = [f for f in scn.flights.values()
                    if f.started and f.flight_id not in self.active]
            if not done:
                return None
            fl = rng.choice(sorted(done, key=lambda f: f.flight_id))
            st = scn.stands[fl.stand]
            st.access_state = "restricted"
            self._push(self.t + mag, "RESTORE", (("stand", fl.stand),))
            return self._record("STAND_RELEASE_DELAY", fl.flight_id, "", (),
                                fl.stand, {"delay": mag})

        if kind == "D9":
            future = [f for f in scn.flights.values() if not f.started]
            if not future:
                return None
            fl = rng.choice(sorted(future, key=lambda f: f.flight_id))
            fl.release += mag
            newq = []
            for entry in self.queue:
                t, pr, sq, k, payload = entry
                if k == "FLIGHT_RELEASE" and payload and payload[0] == fl.flight_id:
                    newq.append((t + mag, pr, sq, k, payload))
                else:
                    newq.append(entry)
            self.queue = newq
            heapq.heapify(self.queue)
            return self._record("ARRIVAL_DELAY", fl.flight_id, "", (), fl.stand,
                                {"delay": mag})

        if kind == "D10":
            self._apply_disturbance({"kind": "D7", "magnitude": mag})
            self._apply_disturbance({"kind": "D4", "magnitude": mag * 0.8})
            return self._record("APRON_CONGESTION", "", "", (), "",
                                {"combined": 1.0})
        return None

    def _report(self, key) -> None:
        self.outage_onset[key] = self.t
        self.report_time[key] = self.t + self.cfg.info_delay

    def _restore(self, payload) -> Optional[AirportOperationalEvent]:
        what = payload[0]
        if what == "gse":
            unit = self.scn.fleet.get(payload[1])
            if unit is not None:
                unit.status = "idle"
                unit.availability_time = max(unit.availability_time, self.t)
                return self._record("GSE_RESTORED", "", "", (unit.resource_id,))
        elif what == "crew":
            crew = self.scn.crews.get(payload[1])
            if crew is not None:
                crew.status = "idle"
                return self._record("CREW_RESTORED", "", "", (crew.crew_id,))
        elif what == "edge":
            edge = self.scn.apron.edges.get(payload[1])
            if edge is not None:
                edge.capacity = float(payload[2])
                self.scn.apron.set_loads(dict(self.scn.apron.load))
        elif what == "stand":
            st = self.scn.stands.get(payload[1])
            if st is not None:
                st.access_state = "open"
        return None

    # ------------------------------------------------------------------
    # intervention execution (the only channel from the controller)
    # ------------------------------------------------------------------
    def execute(self, itype: str, entities: Sequence[str], theta: float = 0.0) -> bool:
        scn = self.scn
        if not self.cfg.allow_time_compression and itype in ("REROUTE_GSE", "ADVANCE_POSITION_ADMISSION"):
            return False
        if not getattr(self.cfg, "allow_resequencing", True) and itype == "RESEQUENCE_TASKS":
            return False
        try:
            if itype == "REROUTE_GSE":
                tid = entities[0]
                mv = self.enroute.get(tid)
                task = scn.tasks.get(tid)
                if mv is None or task is None:
                    return False
                gain = min(theta, 0.6 * max(0.0, mv.arrive - self.t))
                if gain <= 0.3:
                    return False
                task.expedite += 0.0
                self._delay_arrival(tid, -gain)
                self.metrics.interventions += 1
                # Apron externality: giving one vehicle priority consumes shared
                # service-road capacity, so part of the time it saves is paid
                # for by the other vehicles on the same segments.  A
                # representation without spatial state cannot observe this.
                legs = set(mv.legs)
                others = [k for k, o in self.enroute.items()
                          if k != tid and legs & set(o.legs)]
                if others:
                    spill = SPILLOVER * gain / len(others)
                    self.metrics.spill += SPILLOVER * gain
                    for k in others:
                        self._delay_arrival(k, spill)
                return True

            if itype in ("REASSIGN_GSE", "ACTIVATE_RESERVE_GSE"):
                tid, rid = entities[0], entities[1]
                if tid in self.enroute:
                    return False
                task, unit = scn.tasks.get(tid), scn.fleet.get(rid)
                if task is None or unit is None or task.state in ("running", "done"):
                    return False
                if not scn.gse_capable(unit, task) or unit.status == "down":
                    return False
                old = scn.fleet.get(task.gse) if task.gse else None
                if old is not None and tid in old.sequence:
                    old.sequence.remove(tid)
                    old.n_reassign += 1
                if old is not None and old.status == "moving" \
                        and old.current_assignment == tid:
                    old.status = "idle"
                stale = self.enroute.pop(tid, None)
                if stale is not None:
                    # distance already flown toward a task the unit will no
                    # longer perform is deadhead by definition
                    frac = min(1.0, max(0.0, (self.t - stale.depart)
                                        / max(1e-9, stale.arrive - stale.depart)))
                    self.metrics.gse_deadhead += frac * scn.apron.path_length(stale.legs)
                    stale.productive = False
                self.gse_at_stand.pop(tid, None)
                task.gse = rid
                if tid not in unit.sequence:
                    unit.sequence.append(tid)
                if unit.reserve:
                    unit.reserve = False
                    unit.status = "idle" if unit.status == "reserve" else unit.status
                self.metrics.interventions += 1
                return True

            if itype == "SWAP_GSE_SEQUENCE":
                a, b, rid = entities[0], entities[1], entities[2]
                unit = scn.fleet.get(rid)
                if unit is None or a not in unit.sequence or b not in unit.sequence:
                    return False
                ta, tb = scn.tasks.get(a), scn.tasks.get(b)
                if ta is None or tb is None:
                    return False
                if ta.state in ("running", "done") or tb.state in ("running", "done"):
                    return False
                i, j = unit.sequence.index(a), unit.sequence.index(b)
                unit.sequence[i], unit.sequence[j] = unit.sequence[j], unit.sequence[i]
                self.metrics.interventions += 1
                return True

            if itype == "REASSIGN_CREW":
                tid, cid = entities[0], entities[1]
                task, crew = scn.tasks.get(tid), scn.crews.get(cid)
                if task is None or crew is None or task.state in ("running", "done"):
                    return False
                if not scn.crew_qualified(crew, task) or crew.status == "unavailable":
                    return False
                old = scn.crews.get(task.crew) if task.crew else None
                if old is not None and tid in old.sequence:
                    old.sequence.remove(tid)
                task.crew = cid
                if tid not in crew.sequence:
                    crew.sequence.append(tid)
                self.crew_ready_at.pop(tid, None)
                self.metrics.interventions += 1
                return True

            if itype == "SWAP_CREW_SEQUENCE":
                a, b, cid = entities[0], entities[1], entities[2]
                crew = scn.crews.get(cid)
                if crew is None or a not in crew.sequence or b not in crew.sequence:
                    return False
                ta, tb = scn.tasks.get(a), scn.tasks.get(b)
                if ta is None or tb is None:
                    return False
                if ta.state in ("running", "done") or tb.state in ("running", "done"):
                    return False
                i, j = crew.sequence.index(a), crew.sequence.index(b)
                crew.sequence[i], crew.sequence[j] = crew.sequence[j], crew.sequence[i]
                self.metrics.interventions += 1
                return True

            if itype == "CHANGE_SERVICE_POSITION":
                tid, pos = entities[0], entities[1]
                task = scn.tasks.get(tid)
                if task is None or task.state in ("running", "done"):
                    return False
                fl = scn.flights[task.flight]
                st = scn.stands[fl.stand]
                if pos not in st.positions:
                    return False
                task.position = pos
                self.metrics.interventions += 1
                return True

            if itype == "ADVANCE_POSITION_ADMISSION":
                tid = entities[0]
                task = scn.tasks.get(tid)
                if task is None or task.state in ("running", "done"):
                    return False
                fl = scn.flights[task.flight]
                p = scn.stands[fl.stand].positions.get(task.position)
                if p is None or not p.occupants:
                    return False
                # release the position by preempting a not-yet-critical holder
                holder = p.occupants[0]
                ht = scn.tasks.get(holder)
                if ht is None or ht.state != "running":
                    return False
                shave = min(2.0, max(0.0, ht.finish - self.t - 0.5))
                if shave <= 0.2:
                    return False
                ht.finish -= shave
                newq = []
                for entry in self.queue:
                    t, pr, sq, k, payload = entry
                    if k == "TASK_DONE" and payload and payload[0] == holder:
                        newq.append((t - shave, pr, sq, k, payload))
                    else:
                        newq.append(entry)
                self.queue = newq
                heapq.heapify(self.queue)
                self.metrics.interventions += 1
                return True

            if itype == "RESEQUENCE_TASKS":
                a, b = entities[0], entities[1]
                ta, tb = scn.tasks.get(a), scn.tasks.get(b)
                if ta is None or tb is None or ta.flight != tb.flight:
                    return False
                if ta.state in ("running", "done") or tb.state in ("running", "done"):
                    return False
                # Give ``a`` stand-side priority over ``b`` by installing a
                # dynamic precedence b <- a.  This is the "delay a noncritical
                # operation to release a resource" intervention: it is refused
                # if it would close a cycle in the turnaround DAG.
                if a in tb.holds:
                    return False
                if self._reaches(a, b):
                    return False
                tb.holds = tuple(tb.holds) + (a,)
                scn.recompute_ranks(ta.flight)
                self._reorder_queues(scn.flights[ta.flight])
                self.metrics.interventions += 1
                return True
        except (KeyError, IndexError, ValueError):
            return False
        return False

    # ------------------------------------------------------------------
    def _reaches(self, src: str, dst: str) -> bool:
        """True when ``src`` already depends on ``dst`` (cycle guard)."""
        stack, seen = [src], set()
        while stack:
            cur = stack.pop()
            if cur == dst:
                return True
            if cur in seen:
                continue
            seen.add(cur)
            t = self.scn.tasks.get(cur)
            if t is not None:
                stack.extend(tuple(t.predecessors) + tuple(t.holds))
        return False

    # ------------------------------------------------------------------
    def finished(self) -> bool:
        if not self.queue or self.t > self.deadline:
            return True
        return (self.metrics.ready_flights >= len(self.scn.flights)
                and not self.active)

    def edge_load_view(self) -> Dict[str, float]:
        return edge_loads(list(self.enroute.values()), self.t)

    def position_calendar(self, stand: str, position: str,
                          exclude: str = "") -> List[Tuple[str, float, float]]:
        st = self.scn.stands.get(stand)
        if st is None:
            return []
        p = st.positions.get(position)
        if p is None:
            return []
        return [r for r in p.reservations if r[0] != exclude]
