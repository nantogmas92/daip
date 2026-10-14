"""Experiment campaign driver.

Every cell of every study is described by a ``Cell`` record, executed in a
worker process, and written to Parquet.  Paired comparisons rely on common
random numbers: a given ``seed`` fixes the entire apron realisation, so two
methods evaluated at the same seed experienced an identical operation.

Studies
-------
S1  Algorithm 1 propagation: A0 / AS / A1 / A2 at fixed representation.
S2  Algorithm 2 candidate quality: recall, precision and CRR against the
    exhaustively enumerated action universe on small instances.
S3  Candidate-space computational value: CP-SAT held fixed, C0 vs C1 vs C2.
S4  Representation fidelity: R0..R6 with Algorithm 3 and CP-SAT held fixed.
S5  Operating regimes: the best architectures under congestion, utilisation,
    turnaround tightness and bank structure.
S6  Robustness: ETA noise, service-duration noise, disturbance rate.
S7  Scalability: decision latency against the number of simultaneous flights.
SB  Baselines B0..B5.

Usage
-----
    python experiments/runner.py all --seeds 20 --workers 6
    python experiments/runner.py S4 --seeds 30
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

os.environ.setdefault("PYTHONHASHSEED", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from airport.flights import AirportConfig                      # noqa: E402
from algorithms.control import ClosedLoopController, ControllerConfig  # noqa: E402
from experiments.metrics import ExperimentLogger               # noqa: E402
from simulation.disturbances import build_plan                 # noqa: E402
from simulation.event_engine import TurnaroundSimulator        # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "results")


# ---------------------------------------------------------------------------


@dataclass
class Cell:
    study: str
    method: str = ""
    representation: str = "R6"
    propagation: str = "A2"
    candidates: str = "C2"
    validation: str = "V1"
    backend: str = "cpsat"
    intervene: bool = True
    urgency_only: bool = False
    seed: int = 0
    # environment
    n_flights: int = 20
    horizon: float = 150.0
    bank_structure: str = "moderate"
    turnaround_tightness: str = "moderate"
    gse_utilisation: str = "medium"
    crew_utilisation: str = "medium"
    congestion_level: str = "medium"
    safety_profile: str = "default"
    layout: str = "stylised"
    eta_noise: float = 0.05
    duration_noise: float = 0.10
    disturbances: str = "mixed"
    disturbance_rate: float = 1.0
    workload_gain: float = 0.0
    # instrumentation
    measure_universe: bool = False
    recall_every: int = 0
    regime: str = "base"

    def label(self) -> str:
        return "%s|%s|%s|%s|%s|%s|%s" % (
            self.study, self.method or "-", self.representation,
            self.propagation, self.candidates, self.validation, self.backend)


def build_run(cell: Cell):
    """Construct the simulator and controller for one cell.

    Shared by the campaign and by ``experiments/replay_run.py``, so a traced
    replay is built by exactly the code that produced the reported row.
    """
    acfg = AirportConfig(
        n_flights=cell.n_flights, horizon=cell.horizon, layout=cell.layout,
        bank_structure=cell.bank_structure,
        turnaround_tightness=cell.turnaround_tightness,
        gse_utilisation=cell.gse_utilisation,
        crew_utilisation=cell.crew_utilisation,
        congestion_level=cell.congestion_level,
        safety_profile=cell.safety_profile,
        eta_noise=cell.eta_noise, duration_noise=cell.duration_noise,
        workload_gain=cell.workload_gain,
        seed=cell.seed, regime=cell.regime)
    plan = build_plan(cell.seed, cell.horizon, cell.disturbances,
                      rate=cell.disturbance_rate)
    sim = TurnaroundSimulator(acfg, plan)
    ccfg = ControllerConfig(
        representation=cell.representation, propagation=cell.propagation,
        candidates=cell.candidates, validation=cell.validation,
        backend=cell.backend, intervene=cell.intervene,
        urgency_only=cell.urgency_only,
        measure_universe=cell.measure_universe,
        recall_every=cell.recall_every)
    return sim, ClosedLoopController(sim, ccfg, cell.label())


def run_cell(cell: Cell, trace=None) -> Dict[str, Any]:
    sim, ctrl = build_run(cell)
    ctrl.trace = trace
    t0 = time.time()          # as before: construction is not timed
    try:
        summary = ctrl.run()
        status = "ok"
    except Exception:                             # pragma: no cover
        summary = ctrl.summary()
        status = "error: " + traceback.format_exc(limit=2).replace("\n", " ")
    row: Dict[str, Any] = dict(asdict(cell))
    row.update(summary)
    row["wall_s"] = time.time() - t0
    row["status"] = status
    row["scn_tasks"] = len(sim.scn.tasks)
    row["scn_gse"] = len(sim.scn.fleet)
    row["scn_crews"] = len(sim.scn.crews)
    row["scn_stands"] = len(sim.scn.stands)
    row["complete"] = summary.get("ready_flights", 0) >= len(sim.scn.flights)
    return row


def _worker(cell: Cell) -> Dict[str, Any]:
    os.environ.setdefault("PYTHONHASHSEED", "0")
    return run_cell(cell)


# ---------------------------------------------------------------------------
# study definitions
# ---------------------------------------------------------------------------


def study_S1(seeds: Sequence[int]) -> List[Cell]:
    """Propagation mode ablation at fixed representation and scale."""
    out = []
    for s in seeds:
        for n in (12, 24, 36):
            for prop in ("A0", "AS", "A1", "A2"):
                out.append(Cell("S1", method=prop, representation="R6",
                                propagation=prop, candidates="C2",
                                n_flights=n, horizon=60.0 + 5.0 * n, seed=s))
    return out


def study_S2(seeds: Sequence[int]) -> List[Cell]:
    """Candidate quality against the exhaustive universe (small instances)."""
    out = []
    for s in seeds:
        for cand in ("C0", "C1", "C2"):
            out.append(Cell("S2", method=cand, representation="R6",
                            propagation="A2", candidates=cand,
                            n_flights=10, horizon=110.0, seed=s,
                            measure_universe=True, recall_every=4))
    return out


def study_S3(seeds: Sequence[int]) -> List[Cell]:
    """Candidate-space value with the solver held fixed."""
    out = []
    for s in seeds:
        for n in (12, 24, 36):
            for cand in ("C0", "C1", "C2"):
                out.append(Cell("S3", method=cand, representation="R6",
                                propagation="A2", candidates=cand,
                                backend="cpsat", n_flights=n,
                                horizon=60.0 + 5.0 * n, seed=s,
                                measure_universe=True))
    return out


def study_S4(seeds: Sequence[int]) -> List[Cell]:
    """The central representation experiment: R0..R6, Algorithm 3 fixed.

    Propagation and candidate modes track the ladder definition: R6 is the
    level that *adds* the causal machinery, so levels below it run value-only
    propagation and structural candidate generation.  The selection backend is
    CP-SAT throughout, which is what separates the representation effect from
    the optimiser effect.
    """
    out = []
    for s in seeds:
        out.append(Cell("S4", method="none", representation="R6",
                        propagation="A1", candidates="C1", intervene=False,
                        seed=s))
        for r in ("R0", "R1", "R2", "R3", "R4", "R5", "R6"):
            lvl = int(r[1])
            out.append(Cell("S4", method=r, representation=r,
                            propagation="A2" if lvl >= 6 else "A1",
                            candidates="C2" if lvl >= 6 else "C1",
                            backend="cpsat", seed=s))
    return out


def study_S5(seeds: Sequence[int]) -> List[Cell]:
    """Operating regimes: where representation fidelity pays."""
    regimes = {
        "congestion_low": dict(congestion_level="low"),
        "congestion_high": dict(congestion_level="high"),
        "gse_low": dict(gse_utilisation="low"),
        "gse_high": dict(gse_utilisation="high"),
        "crew_high": dict(crew_utilisation="high"),
        "tight": dict(turnaround_tightness="tight"),
        "loose": dict(turnaround_tightness="loose"),
        "clustered": dict(bank_structure="clustered"),
        "uniform": dict(bank_structure="uniform"),
        "strict_safety": dict(safety_profile="strict"),
    }
    out = []
    for s in seeds:
        for name, kw in regimes.items():
            for r in ("R0", "R2", "R4", "R6"):
                lvl = int(r[1])
                out.append(Cell("S5", method=r, representation=r,
                                propagation="A2" if lvl >= 6 else "A1",
                                candidates="C2" if lvl >= 6 else "C1",
                                regime=name, seed=s, **kw))
    return out


def study_S6(seeds: Sequence[int]) -> List[Cell]:
    """Robustness to information quality and disturbance intensity."""
    out = []
    for s in seeds:
        for noise in (0.02, 0.10, 0.20):
            for r in ("R2", "R6"):
                lvl = int(r[1])
                out.append(Cell("S6", method="%s_eta%.2f" % (r, noise),
                                representation=r,
                                propagation="A2" if lvl >= 6 else "A1",
                                candidates="C2" if lvl >= 6 else "C1",
                                eta_noise=noise, duration_noise=noise + 0.05,
                                regime="noise%.2f" % noise, seed=s))
        for rate in (0.5, 1.0, 2.0):
            for r in ("R2", "R6"):
                lvl = int(r[1])
                out.append(Cell("S6", method="%s_rate%.1f" % (r, rate),
                                representation=r,
                                propagation="A2" if lvl >= 6 else "A1",
                                candidates="C2" if lvl >= 6 else "C1",
                                disturbance_rate=rate,
                                regime="rate%.1f" % rate, seed=s))
    return out


def study_S7(seeds: Sequence[int]) -> List[Cell]:
    """Scalability of the decision latency."""
    out = []
    for s in seeds:
        for n in (8, 16, 24, 32, 40, 48):
            for method, prop, cand in (("full", "A0", "C0"),
                                       ("struct", "AS", "C1"),
                                       ("causal", "A2", "C2")):
                out.append(Cell("S7", method=method, representation="R6",
                                propagation=prop, candidates=cand,
                                n_flights=n, horizon=60.0 + 5.0 * n,
                                seed=s, measure_universe=(cand != "C0")))
    return out


def study_SB(seeds: Sequence[int]) -> List[Cell]:
    """Baselines B0..B5 under one common environment."""
    out = []
    for s in seeds:
        out.append(Cell("SB", method="B0_static", representation="R6",
                        propagation="A1", candidates="C1", intervene=False, seed=s))
        out.append(Cell("SB", method="B1_global", representation="R6",
                        propagation="A0", candidates="C0", seed=s))
        # B2 must be able to *act*: the point of the baseline is that it
        # prioritises the most delayed flight without knowing which resource is
        # binding, not that it is blind to resources altogether.  It therefore
        # runs the same cross-flight representation as the proposed method but
        # selects by urgency instead of by causal leverage.
        out.append(Cell("SB", method="B2_urgency", representation="R5",
                        propagation="A1", candidates="C1",
                        urgency_only=True, seed=s))
        out.append(Cell("SB", method="B3_fullspace", representation="R6",
                        propagation="A2", candidates="C0", seed=s))
        out.append(Cell("SB", method="B4_proposed", representation="R6",
                        propagation="A2", candidates="C2", seed=s))
        out.append(Cell("SB", method="B5_greedy", representation="R6",
                        propagation="A2", candidates="C2",
                        backend="greedy", seed=s))
    return out


def study_SV(seeds: Sequence[int]) -> List[Cell]:
    """Validation ablation V0 vs V1."""
    out = []
    for s in seeds:
        for v in ("V0", "V1"):
            out.append(Cell("SV", method=v, representation="R6",
                            propagation="A2", candidates="C2", validation=v,
                            seed=s))
    return out


def _on_geometry(fn, name):
    """Scenario B: the same study on the satellite-pier (geometry) layout."""
    from dataclasses import replace

    def study(seeds):
        return [replace(c, study=name, layout="geometry") for c in fn(seeds)]
    return study


STUDIES = {
    "S4G": _on_geometry(study_S4, "S4G"),
    "SBG": _on_geometry(study_SB, "SBG"),
    "S7G": _on_geometry(study_S7, "S7G"),
    "S1": study_S1, "S2": study_S2, "S3": study_S3, "S4": study_S4,
    "S5": study_S5, "S6": study_S6, "S7": study_S7, "SB": study_SB,
    "SV": study_SV,
}


# ---------------------------------------------------------------------------


def run_study(name: str, seeds: Sequence[int], workers: int,
              out_root: str = RESULTS) -> str:
    cells = STUDIES[name](seeds)
    print("[%s] %d cells x %d seeds -> %d runs"
          % (name, len(cells) // max(1, len(seeds)), len(seeds), len(cells)))
    rows: List[Dict[str, Any]] = []
    t0 = time.time()
    if workers > 1:
        import multiprocessing as mp
        with mp.Pool(workers) as pool:
            for i, row in enumerate(pool.imap_unordered(_worker, cells, chunksize=1)):
                rows.append(row)
                if (i + 1) % 20 == 0:
                    print("  %4d/%d  %.0fs" % (i + 1, len(cells), time.time() - t0),
                          flush=True)
    else:
        for i, c in enumerate(cells):
            rows.append(run_cell(c))
            if (i + 1) % 10 == 0:
                print("  %4d/%d  %.0fs" % (i + 1, len(cells), time.time() - t0),
                      flush=True)
    logger = ExperimentLogger(out_root, name)
    fn = logger.write_summary(rows)
    bad = [r for r in rows if r.get("status") != "ok"]
    print("[%s] done in %.0fs -> %s  (%d errors)"
          % (name, time.time() - t0, fn, len(bad)))
    for r in bad[:3]:
        print("   !", r["label"] if "label" in r else r.get("method"), r["status"][:160])
    return fn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("study", help="study name, or 'all'")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default=RESULTS)
    args = ap.parse_args()
    seeds = list(range(args.offset, args.offset + args.seeds))
    names = list(STUDIES) if args.study == "all" else args.study.split(",")
    for n in names:
        run_study(n, seeds, args.workers, args.out)


if __name__ == "__main__":
    main()
