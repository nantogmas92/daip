"""Apron spatial model: G^apron = (V^apron, E^apron).

Nodes represent aircraft stands, GSE depots, staging areas, service-road
intersections and holding positions.  Edges represent traversable apron and
service-road segments.  Each edge carries

    tau_e(t) = tau_e^0 + c_e(rho_e(t)),

with tau_e^0 the nominal traversal time, rho_e(t) the current occupancy of the
segment and c_e a congestion-delay function.  A vehicle route time is
T_v(t) = sum_{e in P_v} tau_e(t), and a physical disturbance to route time
enters the factor graph as an updated vehicle ETA rather than through a side
channel.

The layout generator produces a stylised two-pier apron with terminal stands,
remote stands, depots, staging areas and multiple alternative service-road
paths, which is the principal controlled experimental environment.  A
geometry-inspired variant is provided for structural robustness checks; it
abstracts published apron geometry only and makes no claim that its traversal
times or procedures reproduce any real operation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

CONGESTION_ALPHA = {"low": 0.22, "medium": 0.60, "high": 1.15}

# Service-road speed used to convert apron geometry into traversal time.
# Node coordinates are metres; a nominal ground-service vehicle covers
# SPEED_SCALE metres per minute (250 m/min ~ 15 km/h), so an edge length in
# minutes is metres / SPEED_SCALE.  Keeping the conversion here means the core
# engine only ever sees times, never geometry.
SPEED_SCALE = 250.0
LAYOUT_SCALE = 2.5          # coordinate unit -> metres


@dataclass
class ApronNode:
    nid: str
    kind: str                 # stand | depot | staging | junction | holding
    x: float
    y: float


@dataclass
class ApronEdge:
    eid: str
    a: str
    b: str
    length: float             # nominal traversal time in minutes at unit speed
    capacity: float           # vehicles that may occupy the segment freely
    metres: float = 0.0       # physical length, reported as GSE travel distance


class ApronGraph:
    """Directed service-road graph with congestion-sensitive traversal."""

    def __init__(self, alpha: float = 0.60, beta: float = 1.6):
        self.nodes: Dict[str, ApronNode] = {}
        self.edges: Dict[str, ApronEdge] = {}
        self.adj: Dict[str, List[Tuple[str, str]]] = {}     # node -> (edge, node)
        self.mu: Dict[str, float] = {}
        self.load: Dict[str, float] = {}
        self.alpha = alpha
        self.beta = beta
        self._route_cache: Dict[Tuple[str, str], Tuple[str, ...]] = {}

    # ------------------------------------------------------------------
    def add_node(self, nid: str, kind: str, x: float, y: float) -> ApronNode:
        n = ApronNode(nid, kind, x, y)
        self.nodes[nid] = n
        self.adj.setdefault(nid, [])
        return n

    def add_edge(self, a: str, b: str, capacity: float = 6.0,
                 metres: Optional[float] = None) -> ApronEdge:
        eid = "E_%s_%s" % (a, b)
        if metres is None:
            na, nb = self.nodes[a], self.nodes[b]
            metres = math.hypot(na.x - nb.x, na.y - nb.y) * LAYOUT_SCALE
        metres = max(40.0, metres)
        e = ApronEdge(eid, a, b, metres / SPEED_SCALE, capacity, metres)
        self.edges[eid] = e
        self.mu[eid] = 1.0
        self.load[eid] = 0.0
        self.adj.setdefault(a, []).append((eid, b))
        self.adj.setdefault(b, []).append((eid, a))     # service roads are bidirectional
        self._route_cache.clear()
        return e

    # ------------------------------------------------------------------
    def route(self, src: str, dst: str) -> Tuple[str, ...]:
        """Nominal-cost shortest path, returned as an edge sequence.

        Routes are computed on nominal length so that the *path* is stable
        while congestion changes only the traversal time along it.  A
        rerouting intervention is therefore a distinct, explicit action rather
        than an implicit side effect of a congestion update.
        """
        key = (src, dst)
        if key in self._route_cache:
            return self._route_cache[key]
        if src == dst or src not in self.nodes or dst not in self.nodes:
            self._route_cache[key] = ()
            return ()
        import heapq
        dist = {src: 0.0}
        prev: Dict[str, Tuple[str, str]] = {}
        pq = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if u == dst:
                break
            if d > dist.get(u, 1e18) + 1e-9:
                continue
            for (eid, v) in self.adj.get(u, ()):
                nd = d + self.edges[eid].length
                if nd < dist.get(v, 1e18) - 1e-9:
                    dist[v] = nd
                    prev[v] = (eid, u)
                    heapq.heappush(pq, (nd, v))
        if dst not in prev and dst != src:
            self._route_cache[key] = ()
            return ()
        legs: List[str] = []
        cur = dst
        while cur != src:
            eid, cur = prev[cur]
            legs.append(eid)
        legs.reverse()
        out = tuple(legs)
        self._route_cache[key] = out
        return out

    def alternate_route(self, src: str, dst: str, avoid: Sequence[str]) -> Tuple[str, ...]:
        """Shortest path avoiding a set of edges; used by REROUTE_GSE."""
        import heapq
        block = set(avoid)
        if src == dst:
            return ()
        dist = {src: 0.0}
        prev: Dict[str, Tuple[str, str]] = {}
        pq = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if u == dst:
                break
            if d > dist.get(u, 1e18) + 1e-9:
                continue
            for (eid, v) in self.adj.get(u, ()):
                if eid in block:
                    continue
                nd = d + self.edges[eid].length
                if nd < dist.get(v, 1e18) - 1e-9:
                    dist[v] = nd
                    prev[v] = (eid, u)
                    heapq.heappush(pq, (nd, v))
        if dst not in prev:
            return ()
        legs: List[str] = []
        cur = dst
        while cur != src:
            eid, cur = prev[cur]
            legs.append(eid)
        legs.reverse()
        return tuple(legs)

    # ------------------------------------------------------------------
    def travel_time(self, legs: Sequence[str], speed: float,
                    congested: bool = True) -> float:
        t = 0.0
        for e in legs:
            edge = self.edges.get(e)
            if edge is None:
                continue
            mult = self.mu.get(e, 1.0) if congested else 1.0
            t += edge.length * mult / max(1e-6, speed)
        return t

    def nominal_travel(self, legs: Sequence[str], speed: float) -> float:
        return self.travel_time(legs, speed, congested=False)

    def path_length(self, legs: Sequence[str]) -> float:
        """Physical route length in metres (used for the GSE-travel metric)."""
        return sum(self.edges[e].metres for e in legs if e in self.edges)

    # ------------------------------------------------------------------
    def set_loads(self, loads: Dict[str, float]) -> Dict[str, float]:
        """Recompute edge multipliers from the current in-transit occupancy."""
        changed: Dict[str, float] = {}
        for eid, edge in self.edges.items():
            rho = loads.get(eid, 0.0)
            self.load[eid] = rho
            mu = 1.0 + self.alpha * (max(0.0, rho) / max(1e-6, edge.capacity)) ** self.beta
            if abs(mu - self.mu[eid]) > 1e-9:
                changed[eid] = mu
            self.mu[eid] = mu
        return changed

    # ------------------------------------------------------------------
    def nodes_of_kind(self, kind: str) -> List[str]:
        return sorted(n for n, v in self.nodes.items() if v.kind == kind)

    def summary(self) -> Dict[str, int]:
        return {
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "stands": len(self.nodes_of_kind("stand")),
            "depots": len(self.nodes_of_kind("depot")),
            "staging": len(self.nodes_of_kind("staging")),
        }


# ---------------------------------------------------------------------------
# Layout generators
# ---------------------------------------------------------------------------


def stylised_apron(n_terminal: int, n_remote: int, n_depot: int = 2,
                   n_staging: int = 2, edge_capacity: float = 6.0,
                   congestion_level: str = "medium", beta: float = 1.6
                   ) -> ApronGraph:
    """Scenario A: reproducible synthetic apron with alternative paths.

    Two piers of terminal stands face a service road; a remote apron sits
    beyond a second road; depots and staging areas connect to both roads
    through junctions, so at least two distinct paths exist between any depot
    and any stand.  That redundancy is what makes a rerouting intervention
    meaningful rather than vacuous.
    """
    g = ApronGraph(alpha=CONGESTION_ALPHA.get(congestion_level, 0.60), beta=beta)

    # service-road spine: junctions J0..Jk along y = 0 and a parallel road at y = 120
    span = max(3, (n_terminal + 1) // 2)
    for i in range(span + 1):
        g.add_node("J%d" % i, "junction", i * 70.0, 0.0)
        g.add_node("K%d" % i, "junction", i * 70.0, 120.0)
    for i in range(span):
        g.add_edge("J%d" % i, "J%d" % (i + 1), edge_capacity)
        g.add_edge("K%d" % i, "K%d" % (i + 1), edge_capacity)
    for i in range(0, span + 1, max(1, span // 2)):
        g.add_edge("J%d" % i, "K%d" % i, edge_capacity)          # cross links
    g.add_edge("J%d" % span, "K%d" % span, edge_capacity)

    # terminal stands hang off the J road, alternating pier sides
    for s in range(n_terminal):
        j = "J%d" % min(span, s // 2)
        side = 1.0 if s % 2 == 0 else -1.0
        g.add_node("T%d" % s, "stand", (s // 2) * 70.0 + 18.0, side * 46.0)
        g.add_edge(j, "T%d" % s, max(2.0, edge_capacity * 0.5))

    # remote stands hang off the K road
    for s in range(n_remote):
        k = "K%d" % min(span, s // 2)
        g.add_node("M%d" % s, "stand", (s // 2) * 70.0 + 18.0, 120.0 + 44.0)
        g.add_edge(k, "M%d" % s, max(2.0, edge_capacity * 0.5))

    # depots and staging areas attach to opposite ends of both roads
    for d in range(n_depot):
        anchor = "J0" if d % 2 == 0 else "K%d" % span
        node = g.nodes[anchor]
        g.add_node("D%d" % d, "depot", node.x - 55.0, node.y - 30.0)
        g.add_edge(anchor, "D%d" % d, edge_capacity)
    for s in range(n_staging):
        anchor = "K0" if s % 2 == 0 else "J%d" % span
        node = g.nodes[anchor]
        g.add_node("G%d" % s, "staging", node.x + 40.0, node.y + 34.0)
        g.add_edge(anchor, "G%d" % s, edge_capacity)

    # a holding position keeps pushback traffic off the service road
    g.add_node("H0", "holding", (span * 70.0) / 2.0, -80.0)
    g.add_edge("J%d" % max(0, span // 2), "H0", edge_capacity)
    return g


def corridor_apron(n_terminal: int, n_remote: int, edge_capacity: float = 6.0,
                   congestion_level: str = "medium", beta: float = 1.6
                   ) -> ApronGraph:
    """Scenario C: three terminal satellites behind one shared service corridor.

    Depots, staging G0, the remote apron and the holding point lie west of a
    single 300 m WEST-EAST link; the three satellites and staging G1 lie east
    of it.  Almost every equipment movement between depots and terminal stands
    therefore shares that link, so congestion couples movements that would use
    separate roads on layouts A and B, and no alternative path exists.  The
    corridor capacity is two thirds of the backbone capacity; stand links use
    the same rule as layouts A and B.
    """
    g = ApronGraph(alpha=CONGESTION_ALPHA.get(congestion_level, 0.60), beta=beta)
    for name, kind, x, y in [("WEST", "junction", -60.0, 0.0),
                             ("EAST", "junction", 60.0, 0.0),
                             ("D0", "depot", -180.0, -70.0),
                             ("D1", "depot", -180.0, 70.0),
                             ("G0", "staging", -90.0, 80.0),
                             ("G1", "staging", 110.0, 70.0),
                             ("RJ", "junction", -60.0, -190.0),
                             ("H0", "holding", -150.0, -220.0)]:
        g.add_node(name, kind, x, y)
    g.add_edge("WEST", "EAST", edge_capacity * 2.0 / 3.0)
    for a, b in [("WEST", "D0"), ("WEST", "D1"), ("WEST", "G0"), ("EAST", "G1"),
                 ("WEST", "RJ"), ("RJ", "H0")]:
        g.add_edge(a, b, edge_capacity)
    centres = [(260.0, 180.0), (340.0, 0.0), (260.0, -180.0)]
    for j, (x, y) in enumerate(centres):
        g.add_node("S%d" % j, "junction", x, y)
        g.add_edge("EAST", "S%d" % j, edge_capacity)
    for k in range(n_terminal):
        j, slot = k % 3, k // 3
        count = (n_terminal + 2 - j) // 3
        x, y = centres[j]
        ang = -1.1 + 2.2 * slot / max(1, count - 1)
        g.add_node("T%d" % k, "stand", x + 65.0 * math.cos(ang), y + 65.0 * math.sin(ang))
        g.add_edge("S%d" % j, "T%d" % k, max(2.0, edge_capacity * 0.5))
    for k in range(n_remote):
        g.add_node("M%d" % k, "stand", -60.0 + (k - (n_remote - 1) / 2.0) * 42.0, -255.0)
        g.add_edge("RJ", "M%d" % k, max(2.0, edge_capacity * 0.5))
    return g


def geometry_inspired_apron(n_terminal: int, n_remote: int,
                            edge_capacity: float = 6.0,
                            congestion_level: str = "medium") -> ApronGraph:
    """Scenario B: abstracted satellite-pier geometry.

    The layout is an abstraction of a common two-satellite pier arrangement.
    It is used only as a *structural* benchmark: no claim is made that its
    traversal times, service policies or procedures reproduce those of any
    real airport, and no proprietary airport data is used anywhere in this
    study.
    """
    g = ApronGraph(alpha=CONGESTION_ALPHA.get(congestion_level, 0.60))
    g.add_node("CORE", "junction", 0.0, 0.0)
    sats = 2
    per = max(2, n_terminal // sats)
    for s in range(sats):
        ang = math.pi * (0.25 + 0.5 * s)
        cx, cy = 190.0 * math.cos(ang), 190.0 * math.sin(ang)
        g.add_node("S%d" % s, "junction", cx, cy)
        g.add_edge("CORE", "S%d" % s, edge_capacity)
        for k in range(per):
            th = ang + (k - per / 2.0) * 0.34
            sid = "T%d" % (s * per + k)
            if int(sid[1:]) >= n_terminal:
                break
            g.add_node(sid, "stand", cx + 62.0 * math.cos(th), cy + 62.0 * math.sin(th))
            g.add_edge("S%d" % s, sid, max(2.0, edge_capacity * 0.5))
    if sats == 2:
        g.add_edge("S0", "S1", edge_capacity)               # tangential taxi road
    g.add_node("RJ", "junction", 0.0, -180.0)
    g.add_edge("CORE", "RJ", edge_capacity)
    for s in range(n_remote):
        g.add_node("M%d" % s, "stand", (s - n_remote / 2.0) * 58.0, -250.0)
        g.add_edge("RJ", "M%d" % s, max(2.0, edge_capacity * 0.5))
    g.add_node("D0", "depot", -150.0, -90.0)
    g.add_edge("CORE", "D0", edge_capacity)
    g.add_edge("RJ", "D0", edge_capacity)
    g.add_node("D1", "depot", 150.0, -90.0)
    g.add_edge("CORE", "D1", edge_capacity)
    g.add_node("G0", "staging", 0.0, 90.0)
    g.add_edge("CORE", "G0", edge_capacity)
    g.add_node("H0", "holding", -90.0, -250.0)
    g.add_edge("RJ", "H0", edge_capacity)
    return g
