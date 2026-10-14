"""Local deterministic operators y = f(x_1, ..., x_n) of the factor graph.

The semantic turnaround model is a typed temporal hypergraph whose hyperedges
couple flights, service tasks, ground support equipment, crews, stands,
service positions and apron road segments.  This module realises those
hyperedges computationally as a factor graph of *small local operators*, each
of which caches

    D_f = (N_f, type_f, state_f, A_f, J_f, SCC_f)

- ``N_f``      incident variables (``inputs`` / ``output``),
- ``type_f``   the propagation operator (``ftype``),
- ``state_f``  cached parameters bound to the physical state,
- ``A_f``      the current binding set (``active``),
- ``J_f``      the local sensitivity representation (``jacobian``),
- ``SCC_f``    the strongly connected component containing the factor.

Every factor must expose ``evaluate`` (value and binding set) and ``jacobian``
(local first-order derivatives over the *currently binding* structure).  The
distinction between the two is what separates G^bind from G^inf: a max factor
with a tied argmax has two binding inputs but splits its derivative between
them, and a saturated workload factor has a binding input whose derivative is
zero.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from .variables import NEG_INF, POS_INF

TIE_TOL = 1e-12


class Factor:
    """Base class: a deterministic local mapping with cached local state."""

    ftype = "generic"

    def __init__(self, fid: str, inputs: Sequence[str], output: str,
                 entity: str = "", layer: str = "task"):
        self.fid = fid
        self.inputs: List[str] = list(inputs)
        self.output = output
        self.entity = entity
        self.layer = layer
        self.scc = -1
        self.active: FrozenSet[str] = frozenset()
        self.params: Dict[str, float] = {}
        self.cost = 1.0                 # relative evaluation cost c_f

    # -- interface ---------------------------------------------------------
    def evaluate(self, g) -> Tuple[float, FrozenSet[str]]:
        raise NotImplementedError

    def jacobian(self, g) -> Dict[str, float]:
        """d f / d x_j over the currently binding structure (sparse)."""
        return {v: 1.0 for v in self.active if v in g.var}

    def margin(self, g, vid: str) -> float:
        """Disturbance-absorption margin M_j for a nonbinding input.

        Zero by default: an affine factor absorbs nothing and transmits every
        input unconditionally.
        """
        return 0.0

    def __repr__(self) -> str:          # pragma: no cover - debugging aid
        return "<%s %s>" % (self.ftype, self.fid)


# ---------------------------------------------------------------------------
# Elementary operators
# ---------------------------------------------------------------------------


class ConstFactor(Factor):
    """y = c.  Surrogate for a component the representation omits."""

    ftype = "const"

    def evaluate(self, g):
        return self.params.get("value", 0.0), frozenset()

    def jacobian(self, g):
        return {}


class MaxFactor(Factor):
    """y = max(floor, x_1, ..., x_n).

    This is the synchronisation operator of the whole model: task starts,
    aircraft readiness and every resource-readiness term are max factors.  The
    factor maintains its argmax (the binding cause set A_y) *and* the
    second-largest input, so that the exact absorption margin

        M_j = M_1 - x_j

    of Proposition 1 can be queried in constant time.
    """

    ftype = "max"

    def __init__(self, fid, inputs, output, entity="", layer="task",
                 floor=NEG_INF, floor_label="__floor__"):
        super().__init__(fid, inputs, output, entity, layer)
        self.floor = floor
        self.floor_label = floor_label
        self.m1 = NEG_INF
        self.m2 = NEG_INF

    def evaluate(self, g):
        best, second = self.floor, NEG_INF
        arg: Set[str] = {self.floor_label} if self.floor > NEG_INF else set()
        for vid in self.inputs:
            v = g.var.get(vid)
            if v is None:
                continue
            x = v.value
            if x > best + TIE_TOL:
                second = best
                best, arg = x, {vid}
            elif abs(x - best) <= TIE_TOL:
                arg.add(vid)
            elif x > second:
                second = x
        self.m1, self.m2 = best, second
        return best, frozenset(arg)

    def jacobian(self, g):
        live = [v for v in self.active if v in g.var]
        if not live:
            return {}
        w = 1.0 / len(live)
        return {v: w for v in live}

    def margin(self, g, vid: str) -> float:
        if vid in self.active:
            return 0.0
        out = g.var.get(self.output)
        src = g.var.get(vid)
        if out is None or src is None:
            return 0.0
        return out.value - src.value


class MinFactor(Factor):
    """y = min(x_1, ..., x_n): first-service time of a flight."""

    ftype = "min"

    def evaluate(self, g):
        best = POS_INF
        arg: Set[str] = set()
        for vid in self.inputs:
            v = g.var.get(vid)
            if v is None:
                continue
            if v.value < best - TIE_TOL:
                best, arg = v.value, {vid}
            elif abs(v.value - best) <= TIE_TOL:
                arg.add(vid)
        if not arg:
            return 0.0, frozenset()
        return best, frozenset(arg)

    def jacobian(self, g):
        live = [v for v in self.active if v in g.var]
        if not live:
            return {}
        w = 1.0 / len(live)
        return {v: w for v in live}


class AffineFactor(Factor):
    """y = c0 + sum_j a_j x_j.  Used for C = S + p and L = d - C_ready."""

    ftype = "affine"

    def __init__(self, fid, inputs, output, entity="", layer="task",
                 c0=0.0, coeff=None):
        super().__init__(fid, inputs, output, entity, layer)
        self.c0 = c0
        self.coeff = dict(coeff or {v: 1.0 for v in inputs})

    def evaluate(self, g):
        val = self.c0
        for vid in self.inputs:
            v = g.var.get(vid)
            if v is not None:
                val += self.coeff.get(vid, 1.0) * v.value
        return val, frozenset(v for v in self.inputs if v in g.var)

    def jacobian(self, g):
        return {v: self.coeff.get(v, 1.0) for v in self.inputs if v in g.var}


# ---------------------------------------------------------------------------
# Spatial / mobile-resource operators
# ---------------------------------------------------------------------------


class TravelFactor(Factor):
    """Congestion-sensitive ETA of a mobile resource at a service position.

        eta = max(release, {C_p : p in Pred}) + setup + sum_e l_e mu_e / s

    The edge multipliers mu_e are graph variables, so an apron congestion
    update reaches every vehicle traversing that segment through ordinary
    propagation rather than through a side channel.  ``base`` carries the
    exogenous release time of the vehicle, and the C-kind inputs are the
    completions this movement must wait for -- including the vehicle's
    *previous flight*, which is how cross-flight coupling enters the graph.
    """

    ftype = "travel"

    def __init__(self, fid, inputs, output, entity="", layer="resource",
                 legs=(), base=0.0, speed=1.0, setup=0.0, congestion=True):
        super().__init__(fid, inputs, output, entity, layer)
        self.legs: Tuple[str, ...] = tuple(legs)
        self.base = base
        self.speed = max(1e-6, speed)
        self.setup = setup
        self.congestion = congestion
        self.cost = 1.0 + 0.2 * len(self.legs)

    def _travel(self, g) -> float:
        t = 0.0
        for e in self.legs:
            if self.congestion:
                mv = g.var.get("mu:" + e)
                mu = mv.value if mv is not None else 1.0
            else:
                mu = 1.0
            t += g.edge_length.get(e, 0.0) * mu / self.speed
        return t

    def evaluate(self, g):
        start = self.base
        active: Set[str] = {"__release__"}
        for vid in self.inputs:
            v = g.var.get(vid)
            if v is None or v.kind not in ("C", "ready"):
                continue
            if v.value > start + TIE_TOL:
                start, active = v.value, {vid}
            elif abs(v.value - start) <= TIE_TOL:
                active.add(vid)
        val = start + self.setup + self._travel(g)
        if self.congestion:
            active |= set("mu:" + e for e in self.legs if ("mu:" + e) in g.var)
        return val, frozenset(active)

    def jacobian(self, g):
        j: Dict[str, float] = {}
        for vid in self.active:
            if vid.startswith("mu:"):
                e = vid[3:]
                j[vid] = g.edge_length.get(e, 0.0) / self.speed
            elif vid in g.var:
                j[vid] = 1.0
        return j

    def margin(self, g, vid: str) -> float:
        if vid in self.active or vid.startswith("mu:"):
            return 0.0
        src = g.var.get(vid)
        if src is None:
            return 0.0
        cands = [self.base]
        for i in self.inputs:
            v = g.var.get(i)
            if v is not None and v.kind in ("C", "ready"):
                cands.append(v.value)
        return max(cands) - src.value


class CongestionFactor(Factor):
    """mu_e = 1 + alpha (rho_e / cap_e)^beta on one apron road segment.

    ``observed_load`` is refreshed from the physical state.  This factor closes
    the reservation <-> predicted-congestion strongly connected component.
    """

    ftype = "congestion"

    def __init__(self, fid, inputs, output, entity="", layer="spatial",
                 alpha=0.55, beta=1.6, cap=6.0):
        super().__init__(fid, inputs, output, entity, layer)
        self.alpha = alpha
        self.beta = beta
        self.cap = max(1e-6, cap)
        self.observed_load = 0.0

    def evaluate(self, g):
        rho = max(0.0, self.observed_load)
        val = 1.0 + self.alpha * (rho / self.cap) ** self.beta
        return val, frozenset(v for v in self.inputs if v in g.var)

    def jacobian(self, g):
        return {}


# ---------------------------------------------------------------------------
# Service and resource-state operators
# ---------------------------------------------------------------------------


class ServiceFactor(Factor):
    """C_j = S_j + p_j^0 * r_h * (1 + gamma W_h) + overrun.

    The crew workload state W_h enters as a *structural* input, not merely as
    something the factor reads: G^inf must be a sub-graph of G^struct, so an
    influence edge without a structural edge would let a workload update bypass
    propagation and break Proposition C.
    """

    ftype = "service"

    def __init__(self, fid, inputs, output, entity="", layer="task",
                 nominal=10.0, rate=1.0, gamma=0.0, workload_var=None):
        super().__init__(fid, inputs, output, entity, layer)
        self.nominal = nominal
        self.rate = rate
        self.gamma = gamma
        self.workload_var = workload_var
        self.overrun = 0.0            # observed service-duration overrun
        self.cost = 1.1

    def duration(self, g) -> float:
        w = 0.0
        if self.workload_var and self.workload_var in g.var:
            w = g.var[self.workload_var].value
        return self.nominal * self.rate * (1.0 + self.gamma * w) + self.overrun

    def evaluate(self, g):
        s = 0.0
        if self.inputs and self.inputs[0] in g.var:
            s = g.var[self.inputs[0]].value
        return s + self.duration(g), frozenset(v for v in self.inputs if v in g.var)

    def jacobian(self, g):
        if not self.inputs:
            return {}
        j = {self.inputs[0]: 1.0}
        if self.workload_var and self.workload_var in g.var:
            j[self.workload_var] = self.nominal * self.rate * self.gamma
        return j


class WorkloadFactor(Factor):
    """W_h = clip(W_h^0 + kappa * sum_j (C_j - S_j)) for a service crew.

    The accumulated quantity is *service duration*, not absolute clock time:
    ``inputs`` is a flat list of alternating (S_j, C_j) variables so each served
    task contributes C_j - S_j.  Using absolute completion times would make the
    loop gain grow with the simulation clock rather than with the workload.
    A saturated crew transmits no first-order influence, which is why the
    Jacobian is empty at the clip.
    """

    ftype = "workload"

    def __init__(self, fid, inputs, output, entity="", layer="resource",
                 base=0.0, kappa=0.0016, cap=1.0, span=600.0):
        super().__init__(fid, inputs, output, entity, layer)
        self.base = base
        self.kappa = kappa
        self.cap = cap
        self.span = span

    def _pairs(self):
        it = iter(self.inputs)
        return list(zip(it, it))

    def _load(self, g) -> float:
        load = 0.0
        for (sv, cv) in self._pairs():
            a, b = g.var.get(sv), g.var.get(cv)
            if a is None or b is None:
                continue
            load += min(self.span, max(0.0, b.value - a.value))
        return load

    def evaluate(self, g):
        val = min(self.cap, max(0.0, self.base + self.kappa * self._load(g)))
        return val, frozenset(v for v in self.inputs if v in g.var)

    def jacobian(self, g):
        if self.base + self.kappa * self._load(g) >= self.cap - 1e-9:
            return {}                       # saturated: no first-order influence
        j: Dict[str, float] = {}
        for (sv, cv) in self._pairs():
            a, b = g.var.get(sv), g.var.get(cv)
            if a is None or b is None:
                continue
            d = b.value - a.value
            if 0.0 < d < self.span:
                j[cv] = self.kappa
                j[sv] = -self.kappa
        return j


class CapacityFactor(Factor):
    """Finite-capacity projection R = P_q(a) over an interval calendar.

        N_q(tau)  = sum_j 1_{[s_j, c_j)}(tau)
        P_q(a, p) = inf{ t >= a : N_q(u) < Q_q for all u in [t, t+p) }

    Used for stand service positions, shared staging areas and any other
    finite-capacity resource.  Calendar entries are the model's *predicted*
    occupancy intervals of the peers currently holding or queued for the same
    resource, so the peers' completion variables are genuine structural inputs:
    that is what couples flights sharing a stand-side position in G^struct and,
    when the resource saturates, in G^bind and G^inf as well.
    """

    ftype = "capacity"

    def __init__(self, fid, inputs, output, entity="", layer="resource",
                 capacity=1, resource=""):
        super().__init__(fid, inputs, output, entity, layer)
        self.capacity = max(1, int(capacity))
        self.resource = resource
        self.peer_starts: Dict[str, float] = {}
        self.binding = False
        self.infeasible = False
        self.blocker: Optional[str] = None
        self.cost = 1.6
        # Exact earliest-fit reference: each unstarted peer occupies its own
        # predicted interval [S_peer, C_peer) instead of [now, C_peer), and a
        # request may use a vacant gap before a peer if its whole duration fits.
        self.calendar_mode = False
        self.peer_start_vars: Dict[str, str] = {}
        self.duration = 0.0

    def calendar_project(self, g, a, duration=None):
        """Exact earliest slot for the supplied half-open intervals."""
        duration = self.duration if duration is None else duration
        intervals = self._intervals(g)
        candidates = sorted({a} | {c for _, _, c in intervals if c >= a})
        for start in candidates:
            end = start + max(1e-9, duration)
            points = sorted({start} | {s for _, s, c in intervals if start < s < end})
            if all(sum(s <= x < c for _, s, c in intervals) < self.capacity for x in points):
                blockers = sorted(v for v, _, c in intervals if abs(c-start) < 1e-9)
                return start, (blockers[0] if start > a and blockers else None)
        raise RuntimeError("Finite calendar must have a free slot after its last completion")

    def _intervals(self, g) -> List[Tuple[str, float, float]]:
        out: List[Tuple[str, float, float]] = []
        for vid, s0 in self.peer_starts.items():
            v = g.var.get(vid)
            if self.calendar_mode and vid in self.peer_start_vars:
                sv = g.var.get(self.peer_start_vars[vid])
                if sv is not None and abs(sv.value) < 1e17:
                    s0 = sv.value
            if v is None or v.value <= s0:
                continue
            out.append((vid, s0, v.value))
        return out

    def project(self, g, a: float) -> Tuple[float, Optional[str]]:
        if self.calendar_mode:
            return self.calendar_project(g, a)
        iv = [x for x in self._intervals(g) if x[2] > a]
        if len(iv) < self.capacity:
            return a, None
        iv.sort(key=lambda x: x[2])
        k = len(iv) - self.capacity
        return max(a, iv[k][2]), iv[k][0]

    def evaluate(self, g):
        a = 0.0
        if self.inputs and self.inputs[0] in g.var:
            a = g.var[self.inputs[0]].value
        r, blocker = self.project(g, a)
        self.binding = r > a + 1e-9
        self.blocker = blocker
        self.infeasible = False
        if self.binding and blocker:
            act = set([blocker])
        elif self.inputs:
            act = set([self.inputs[0]])
        else:
            act = set()
        return r, frozenset(v for v in act if v in g.var)

    def jacobian(self, g):
        if self.binding and self.blocker and self.blocker in g.var:
            return {self.blocker: 1.0}
        if self.inputs and self.inputs[0] in g.var:
            return {self.inputs[0]: 1.0}
        return {}

    def margin(self, g, vid: str) -> float:
        """How far a peer release may slip before it becomes the blocker."""
        iv = sorted(self._intervals(g), key=lambda x: x[2])
        if len(iv) < self.capacity:
            return POS_INF
        k = len(iv) - self.capacity
        cur = iv[k][2]
        for (pv, _s, c) in iv:
            if pv == vid:
                return max(0.0, cur - c)
        return 0.0


class IncompatibilityFactor(Factor):
    """Configurable safety / service-position incompatibility separation.

        S_j >= C_i + sep    for every task i incompatible with j at one stand

    The pairwise relation K_{q1,q2} in {compatible, incompatible, conditional}
    is supplied by the airport layer's configurable constraint matrix rather
    than hard-coded here, because concurrency rules for aircraft servicing are
    operator- and airport-specific.  ``sep`` is the required temporal
    separation for a conditional pair and zero for a strictly sequential one.
    """

    ftype = "incompat"

    def __init__(self, fid, inputs, output, entity="", layer="task", sep=0.0):
        super().__init__(fid, inputs, output, entity, layer)
        self.sep = sep
        self.cost = 1.2

    def evaluate(self, g):
        best, arg = NEG_INF, set()
        for vid in self.inputs:
            v = g.var.get(vid)
            if v is None:
                continue
            x = v.value + self.sep
            if x > best + TIE_TOL:
                best, arg = x, set([vid])
            elif abs(x - best) <= TIE_TOL:
                arg.add(vid)
        if not arg:
            return NEG_INF, frozenset()
        return best, frozenset(arg)

    def jacobian(self, g):
        live = [v for v in self.active if v in g.var]
        if not live:
            return {}
        return {v: 1.0 / len(live) for v in live}

    def margin(self, g, vid: str) -> float:
        if vid in self.active:
            return 0.0
        out, src = g.var.get(self.output), g.var.get(vid)
        if out is None or src is None:
            return 0.0
        return out.value - (src.value + self.sep)


FACTOR_CLASSES = dict(
    (c.ftype, c) for c in (
        ConstFactor, MaxFactor, MinFactor, AffineFactor, TravelFactor,
        CongestionFactor, ServiceFactor, WorkloadFactor, CapacityFactor,
        IncompatibilityFactor,
    )
)
