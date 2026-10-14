"""The sparse action-sensitivity stream S_x.

Two logically distinct message streams are maintained by Algorithm 1.  The
*state* stream propagates changes to predicted physical and operational state,
x -> x'.  The *sensitivity* stream propagates changes to control influence,

    s_x^u = d x / d u,

with the chain rule over each local factor:

    s_y^u = sum_j (d f / d x_j) s_{x_j}^u.

Numerical state changes therefore do not necessarily require sensitivity
recomputation; sensitivities are recomputed when the local active set changes
or when the local derivative depends continuously on the modified state.

Truncation inside cyclic components
-----------------------------------
Inside a cyclic SCC the recursion above is a power iteration that diverges
whenever the component's linearised loop gain exceeds one (this happens, for
example, when a long GSE queue couples completion times back through crew
workload).  We therefore propagate sensitivities on the condensation DAG only:
contributions arriving from inputs *inside* the same component are dropped, so
the reported S_x is the feed-forward first-order term.  This is an explicit
approximation and is reported as such.  State values remain exact, because the
state fixed point is solved rather than truncated.
"""

from __future__ import annotations

from typing import Dict, Optional

from .variables import HANDLE_KINDS

SENS_TOL = 1e-9
SENS_EQ_TOL = 1e-6


def factor_sensitivity(g, f, out, in_cycle: bool = False) -> Dict[str, float]:
    """s_y^u = sum_j (df/dx_j) s_{x_j}^u over the sparse support."""
    acc: Dict[str, float] = {}
    jac = f.jacobian(g)
    if not jac:
        return {out.vid: 1.0} if out.kind in HANDLE_KINDS else {}
    own = g.scc_of(out.vid) if in_cycle else -2
    for vid, w in jac.items():
        src = g.var.get(vid)
        if src is None or abs(w) < 1e-12:
            continue
        if in_cycle and g.scc_of(vid) == own:
            continue                    # intra-component feedback truncated
        for u, s in src.sens.items():
            nv = acc.get(u, 0.0) + w * s
            if abs(nv) > SENS_TOL:
                acc[u] = nv
            elif u in acc:
                del acc[u]
    if out.kind in HANDLE_KINDS:
        acc[out.vid] = 1.0              # own exogenous control handle
    return acc


def sens_differs(a: Dict[str, float], b: Dict[str, float]) -> bool:
    if len(a) != len(b):
        return True
    for k, v in a.items():
        if abs(b.get(k, 0.0) - v) > SENS_EQ_TOL:
            return True
    return False


def recompute_all(g) -> None:
    """Reference (non-incremental) sensitivity solve over the whole graph."""
    g.ensure_scc()
    for v in g.var.values():
        v.sens = {}
    order = sorted(g.fac.values(), key=lambda f: g.topo_index(f.output))
    for _ in range(2):
        for f in order:
            out = g.var.get(f.output)
            if out is None:
                continue
            cyc = g.is_cyclic(g.scc_of(out.vid))
            out.sens = factor_sensitivity(g, f, out, in_cycle=cyc)


def influence_ancestors(g, target: str, use_influence: bool = True,
                        max_depth: int = 8):
    """Anc_{G^inf}(z), truncated at a bounded causal path length.

    With ``use_influence=False`` the same traversal runs on G^struct, which is
    what the C1 ablation (structural-neighbourhood pruning) uses.
    """
    seen = set()
    stack = [(target, 0)]
    while stack:
        vid, d = stack.pop()
        if vid in seen:
            continue
        seen.add(vid)
        if d >= max_depth:
            continue
        v = g.var.get(vid)
        if v is None or v.producer is None:
            continue
        f = g.fac.get(v.producer)
        if f is None:
            continue
        preds = f.jacobian(g).keys() if use_influence else f.inputs
        for p in preds:
            if p in g.var and p not in seen:
                stack.append((p, d + 1))
    return seen


def structural_expansion(g, region, kappa: int = 1):
    """N_kappa^struct(B): expand a causal region by kappa structural hops."""
    cur = set(region)
    for _ in range(max(0, kappa)):
        nxt = set(cur)
        for vid in sorted(cur):
            v = g.var.get(vid)
            if v is None:
                continue
            if v.producer and v.producer in g.fac:
                nxt.update(x for x in g.fac[v.producer].inputs if x in g.var)
            for fid in v.consumers:
                f = g.fac.get(fid)
                if f is not None and f.output in g.var:
                    nxt.add(f.output)
        cur = nxt
    return cur


def absorption_margins(g, target: str) -> Dict[str, float]:
    """M_j for every nonbinding input of the factor producing ``target``."""
    v = g.var.get(target)
    if v is None or v.producer is None:
        return {}
    f = g.fac.get(v.producer)
    if f is None:
        return {}
    out: Dict[str, float] = {}
    for vid in f.inputs:
        if vid in f.active or vid not in g.var:
            continue
        out[vid] = f.margin(g, vid)
    return out
