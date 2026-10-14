"""Strongly connected component decomposition and condensation ordering.

The structural dependency graph of a turnaround system is *not* acyclic.  Two
mechanisms close cycles in practice:

    route reservations   <->  predicted apron congestion
    resource schedules   <->  predicted service availability

so the propagation engine cannot simply sweep a topological order.  We
therefore decompose G^struct into strongly connected components K_1..K_m,
contract each one, and propagate over the resulting condensation DAG G^SCC.

The decomposition is performed on the *structural* graph rather than on the
current binding graph.  That is deliberate and conservative: the binding graph
changes whenever an argmax switches, so an SCC decomposition derived from it
would have to be recomputed on every criticality switch, whereas the
structural decomposition is stable for as long as the model topology is.

Tarjan's algorithm is implemented iteratively because turnaround graphs at
bank scale reach recursion depths that exceed the interpreter limit.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple


def tarjan_scc(adj: Dict[str, List[str]]) -> List[List[str]]:
    """Iterative Tarjan.  Returns components in reverse topological order."""
    index: Dict[str, int] = {}
    low: Dict[str, int] = {}
    on: Dict[str, bool] = {}
    stack: List[str] = []
    comps: List[List[str]] = []
    counter = [0]

    for root in adj:
        if root in index:
            continue
        work: List[Tuple[str, int]] = [(root, 0)]
        while work:
            v, pi = work[-1]
            if pi == 0:
                index[v] = low[v] = counter[0]
                counter[0] += 1
                stack.append(v)
                on[v] = True
            recurse = False
            nbrs = adj.get(v, ())
            for i in range(pi, len(nbrs)):
                w = nbrs[i]
                if w not in index:
                    work[-1] = (v, i + 1)
                    work.append((w, 0))
                    recurse = True
                    break
                elif on.get(w):
                    low[v] = min(low[v], index[w])
            if recurse:
                continue
            work[-1] = (v, len(nbrs))
            if low[v] == index[v]:
                comp: List[str] = []
                while True:
                    w = stack.pop()
                    on[w] = False
                    comp.append(w)
                    if w == v:
                        break
                comps.append(comp)
            work.pop()
            if work:
                p = work[-1][0]
                low[p] = min(low[p], low[v])
    return comps


class Condensation:
    """The condensation DAG G^SCC together with its topological index."""

    def __init__(self, comps: Sequence[Sequence[str]], adj: Dict[str, List[str]]):
        self.members: List[List[str]] = [list(c) for c in comps]
        self.of: Dict[str, int] = {}
        for i, comp in enumerate(self.members):
            for v in comp:
                self.of[v] = i
        # cyclic if it has more than one member or carries a self-loop
        self.cyclic: List[bool] = []
        for i, comp in enumerate(self.members):
            cyc = len(comp) > 1
            if not cyc:
                v = comp[0]
                cyc = any(w == v for w in adj.get(v, ()))
            self.cyclic.append(cyc)
        # Tarjan emits components in reverse topological order
        n = len(self.members)
        self.index: Dict[int, int] = dict((c, n - 1 - c) for c in range(n))

    # -- queries -----------------------------------------------------------
    def topo_index(self, vid: str) -> int:
        return self.index.get(self.of.get(vid, 0), 0)

    def component_of(self, vid: str) -> int:
        return self.of.get(vid, -1)

    def is_cyclic(self, cid: int) -> bool:
        return bool(0 <= cid < len(self.cyclic) and self.cyclic[cid])

    @property
    def n_components(self) -> int:
        return len(self.members)

    @property
    def n_cyclic(self) -> int:
        return sum(1 for c in self.cyclic if c)

    @property
    def largest_cyclic(self) -> int:
        sizes = [len(m) for m, c in zip(self.members, self.cyclic) if c]
        return max(sizes) if sizes else 0
