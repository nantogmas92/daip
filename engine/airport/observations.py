"""What the controller can observe about resources and running services.

Forecast resource release uses observable state, never hidden event times.
Disturbance reports can additionally be delayed or missing
(``AirportConfig.info_delay`` / ``info_missing``).  Only the controller's
planning model and action feasibility read these views; physical execution and
the baseline dispatcher act on the true state.
"""
from __future__ import annotations

# Planning default for an outage whose end time is not reported (minutes).
NOMINAL_OUTAGE = {"gse": 20.0, "crew": 8.0}


def _key(resource):
    rid = getattr(resource, "resource_id", None)
    return ("gse", rid) if rid is not None else ("crew", resource.crew_id)


def _pending(sim, key):
    """Report time of an outage the controller has not yet been told about."""
    reports = getattr(sim, "report_time", None)
    if not reports:
        return None
    t = reports.get(key)
    return t if t is not None and sim.t < t else None


def observed_status(sim, resource):
    status = resource.status
    if status in ("down", "unavailable") and _pending(sim, _key(resource)) is not None:
        return "idle"
    return status


def observed_outage_end(sim, resource):
    """Announced end of a reported outage, or the planning default if missing."""
    if getattr(sim.cfg, "info_missing", False):
        kind, _ = _key(resource)
        onset = getattr(sim, "outage_onset", {}).get(_key(resource), sim.t)
        return max(sim.t, onset + NOMINAL_OUTAGE[kind])
    return max(sim.t, resource.availability_time)


def observed_overrun(sim, task):
    hidden = getattr(sim, "hidden_overrun", None)
    if not hidden or task.state == "done":
        return task.overrun
    return task.overrun - sum(mag for (report, mag) in hidden.get(task.tid, ())
                              if sim.t < report)


def resource_release(sim, resource, multiplier=None, observed=False):
    """Current commitment's nominal release, or an observed past idle release.

    Finish/arrival times stored for physical event scheduling are deliberately
    excluded. With ``observed`` the controller's view of disturbance reports is
    used; announced outage end times are otherwise assumed known.
    """
    scn=sim.scn
    factor=multiplier or (lambda task,kind:1.0)
    tid=resource.current_assignment
    task=scn.tasks.get(tid) if tid else None
    status=observed_status(sim, resource) if observed else resource.status
    if status=='serving' and task is not None and task.state=='running':
        crew=scn.crews.get(task.crew) if task.crew else None
        rate=crew.rate if crew else 1.0
        workload=(1.0+crew.workload_gain*crew.workload_state) if crew else 1.0
        overrun=observed_overrun(sim, task) if observed else task.overrun
        duration=task.duration*rate*workload*factor(task.tid,'service')+overrun-task.expedite
        return max(sim.t,task.start+max(.5,duration))
    if status=='moving' and tid in sim.enroute:
        mv=sim.enroute[tid]
        duration=scn.apron.travel_time(mv.legs,resource.speed,congested=True)
        return max(sim.t,mv.depart+duration*factor(tid,'travel'))
    if status in ('down','unavailable'):
        if observed:
            return observed_outage_end(sim, resource)
        return max(sim.t,resource.availability_time)
    return min(sim.t,resource.availability_time) if status==resource.status else sim.t
