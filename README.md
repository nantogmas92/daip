# DAIP: dependency-aware intervention planning for aircraft turnaround recovery

Source code, input data and run summaries for the paper

> S. Nantogma, W. Ran, S. Zhang. *Dependency-aware intervention planning for
> coordinated aircraft turnaround recovery using explicit delay knowledge.*
> Submitted to *Advanced Engineering Informatics*.

Dependency-aware intervention planning (DAIP) recovers active aircraft
turnarounds that share equipment, crews, service positions and apron roads. It
maintains an **operational dependency model**: a max-plus timing model that
records, for every predicted time, the inputs that currently determine it (its
**binding provenance**). When a readiness target is violated, DAIP traces the
binding dependencies backwards and searches them and their one-step structural
ring for interventions. It evaluates the candidates counterfactually in the
same model and coordinates interacting actions across aircraft. Each bundle is
validated jointly before it is dispatched with a decision record.

This repository contains:

- the engine;
- the six comparison methods;
- the closed-loop simulator;
- the scripts of every study in the paper;
- the public-data inputs of the Charlotte Douglas (CLT) case study;
- the figure and analysis scripts.

The run summaries behind every reported number are distributed with the
repository (see [Data](#data)). **All evidence comes from simulation:** run
outputs are measured from the simulator and are not airport observations.

## Contents

```
engine/            DAIP engine, simulator and closed-loop harness (Python package root)
  core/            typed variables, timing operators, dependency graph, condensation, journal
  airport/         flights, service templates, equipment, crews, stands, apron roads, observation layer
  simulation/      discrete-event execution, dispatch, disturbance families, outcome accounting
  algorithms/      Algorithms 1-3, counterfactual operators, interaction terms, validation, completion
  optimization/    selection backends: beam (primary), greedy, CP-SAT, static top-k
  experiments/     revision_harness.py: the paired closed-loop harness used by every study
  tests/           regression and integration tests (37)
studies_main/      primary DAIP configuration, comparators and CLT case study
  baselines/       dispatcher rule, bipartite re-matching, CP-SAT re-optimisation, Bao et al., Krauth et al.
  abl/             DAIP primary runs, representation ladder, component ablation, selection backends
  clt/             CLT bank construction, scenario, calibration, comparisons, figures
  data/            raw public inputs (small files), derived CLT inputs, MANIFEST.csv of sources
  runs/            run summaries (from the data archive)
  results/         analysis outputs
studies_earlier/   earlier DAIP configuration: robustness, information, burden, scale, saved states, audits
  runs/ runs_fairness/ runs_p6ext/ timing/ audits/ states/   run summaries and audits (data archive)
  results/         numbers.json and tables/ written by analyze.py
figures/           manuscript figures written by the figure scripts
run_example.py     one paired scenario: no repair, dispatcher rule and DAIP
```

## Installation

Python 3.10 or later. The study runs used Python 3.13.0, and this release was
also checked with Python 3.14.

```bash
pip install -r requirements.txt          # lower bounds
pip install -r requirements-lock.txt     # exact versions of the study runs (Python 3.13)
```

The dependencies are numpy, pandas, scipy, matplotlib, pyarrow, ortools
(CP-SAT) and pytest. Nothing needs to be installed from this repository: every
script adds `engine/` to the import path. Run all commands from the repository
root.

```bash
cd engine && python -m pytest tests -q && cd ..      # 37 tests
python run_example.py --layout geometry --seed 1000
```

`run_example.py` runs no repair, the dispatcher rule and DAIP on one paired
scenario of the main comparison. Its results equal the stored summaries of
that scenario.

## Terminology: paper and code

| Paper | Code |
|---|---|
| operational dependency model, timing operators | `core.graph` (factor graph), `core.factors` (operators) |
| binding inputs / binding provenance | binding sets of the operators (`core.factors`), sensitivities (`core.sensitivities`) |
| Algorithm 1, state updates and counterfactual evaluation | `algorithms/algorithm1_propagation.py`, `algorithms/counterfactual.py` |
| Algorithm 2, binding-guided candidate search | `algorithms/algorithm2_candidates.py` |
| Algorithm 3, coordinated selection, validation and completion | `algorithms/algorithm3_selection.py`, `interaction.py`, `validation.py`, `completion.py`, `optimization/` |
| DAIP | method / arm `proposed` |
| No repair, dispatcher rule, bipartite re-matching | `none`, `dispatch`, `matching` |
| Bao et al., Krauth et al. (common-interface adaptations) | `bao`, `krauth` (`studies_main/baselines/literature.py`) |
| CP-SAT re-optimisation | `reopt` (`studies_main/baselines/cpr.py`) |
| Structural search, unscreened search, task-urgency search | arms `structural`, `global`, `urgency` |
| Without ring / completion / joint validation | arms `no_ring`, `no_completion`, `no_validation` |
| Representation ladder R0-R6 | arms `R0` ... `R6` of study `P2L` |
| Layout A, Layout B | raw layout labels `geometry`, `corridor` |

The raw label `stylised` belongs to a third layout that the paper does not use.
Tables written by `studies_earlier/analyze.py` label the raw layouts
`stylised`, `geometry` and `corridor` as A, B and C. In those tables, B and C
are the paper's layouts A and B.

## Configurations

- **Primary configuration** (paper Table 5):
  - beam selection;
  - Q = 130 planning evaluations per decision, of which up to 107 are for
    single-action screening and 8 for pair evaluations;
  - beam width 8;
  - interaction cutoff 0.25 min;
  - backward-trace depth 40;
  - B = 4;
  - minimum gain 0.5 min.

  This is DAIP in the main comparison, at level R6 of the representation
  ladder, in the selection-backend study and in both CLT experiments
  (`studies_main`). Two `studies_main` runs use greedy selection with the same
  budget: ladder levels R0-R5 and the base of the component ablation.
- **Earlier configuration**: the same model and search, with exact CP-SAT
  selection and Q = 194 for every method. Used for the robustness,
  information-quality, intervention-burden, scale and saved-state studies
  (`studies_earlier`).
- **Comparator budgets:**
  - in the main comparison, the operating and information conditions and the
    CLT tight-turn experiment, the comparators ran under the earlier protocol
    with Q = 194;
  - in the original-schedule CLT experiment, every method had Q = 130.

Every run summary records its full specification (`spec`) and the SHA-256
digest of the engine's runtime modules (`source_sha256`).

## Data

The run summaries are distributed as archives that unpack into this folder:

| Archive | Contents | Needed for |
|---|---|---|
| `daip-run-summaries.zip` | the folders `studies_main/runs/`, `studies_earlier/runs/`, `runs_fairness/`, `runs_p6ext/`, `timing/` and `audits/`. One JSON summary per run, plus the run logs that the analysis reads (studies P1, P2, P5 and CLT_TIGHT) | every analysis and figure script |
| `daip-saved-states.zip` | `studies_earlier/states/`: the decision states saved by study FS | re-running the saved-state audits (`frozen_audit.py`, `region_check.py`, `consistency_audit.py`, `fairness_check.py recall`) |

```bash
unzip daip-run-summaries.zip      # from the repository root
unzip daip-saved-states.zip       # optional
```

The saved states are Python pickles of the planning model. As with any
pickle, load them only from this archive.

On Windows, run-file paths can exceed the 260-character limit when the
repository sits in a deeply nested folder. Unpack it close to the drive root
(for example `C:\daip`) or enable long paths.

The analyses below read only these files. The summaries are kept: a study
runner skips every run whose summary exists. To re-run a study from scratch,
move its run folder away first.

## Reproducing the paper

Each step is listed with the script that writes the result from the saved
runs (seconds to minutes). The study runners re-run the simulations; all
studies together took about 120 CPU-hours.

| Paper | From saved runs | Re-run the simulations |
|---|---|---|
| Fig. 5 layouts; Fig. 11a share of the graph evaluated | `python studies_main/figures_extra.py` | `python studies_earlier/campaign.py P5 --scale 2` |
| Sec. 6.1, Fig. 6, main comparison and abstract percentages | `python studies_main/figures_main.py`; `python studies_main/baselines/run_native.py analyze` | `python studies_main/abl/run_daip_beam.py run`; `python studies_main/baselines/run_native.py run`; no repair: `campaign.py P1 --scale 2` |
| Sec. 6.2, Fig. 7a, representation ladder | `python studies_main/abl/ladder_figure.py`; `run_p2_ladder.py analyze` | `python studies_main/abl/run_p2_ladder.py run` (R6 from `run_daip_beam.py`) |
| Sec. 6.3, Fig. 7b, component ablation | `python studies_main/abl/ablation_figure.py` | `python studies_main/abl/run_p1_abl.py run` |
| Sec. 6.3, Fig. 8, selection backends | `python studies_main/abl/selection_figure.py` | `python studies_main/abl/run_selection.py run` |
| Sec. 6.4, saved-state reference and single-action audit | `python studies_earlier/analyze.py FS` | `campaign.py FS --scale 2`; `frozen_audit.py`; `region_check.py`; `fairness_check.py recall` |
| Sec. 6.5, Figs. 9-10, regimes and information conditions | `python studies_main/figures_main.py` | `campaign.py P3C,P3RC,P4C,P7 --scale 2`; `python studies_main/baselines/run_native_stress.py run`; `fairness_check.py run` |
| Sec. 6.6, Table 6, burden and lead time | `python studies_main/audit_reported.py` | `campaign.py P6 --scale 2` (Layout A); `python studies_earlier/p6ext_check.py run` (Layout B) |
| Sec. 6.7, Fig. 11b, timing, equivalence and consistency audits | `python studies_main/figures_main.py`; `python studies_earlier/analyze.py` | `campaign.py TIME` (alone, on an idle machine); `equivalence_audit.py`; `consistency_audit.py` |
| Sec. 7, Fig. 12, CLT apron | `python studies_main/figures_main.py` | input: `python studies_main/clt/build_bank.py` |
| Sec. 7.2, CLT calibration and validation | `studies_main/results/clt_calibration2.md`, `clt_validation.md` | `python studies_main/clt/run_clt.py calibrate2`; `run_clt.py validate` |
| Sec. 7, Table 7, Fig. 13, original-schedule CLT case | `python studies_main/clt/clt_figures.py` | `python studies_main/clt/run_clt_timecap.py run` (DAIP from `run_daip_beam.py`) |
| Sec. 7, Table 8, controlled tight-turn experiment | `python studies_main/clt/run_tight_bank.py analyze` | `python studies_main/clt/run_tight_bank.py run` |
| Every number quoted in the text of Sections 6-7 | `python studies_main/audit_reported.py` | |

`studies_main/audit_reported.py` recomputes the numbers quoted in Sections 6
and 7 from the saved runs and writes `studies_main/results/reported_numbers.md`.
`studies_earlier/analyze.py` regenerates `results/numbers.json` and
`results/tables/` of the earlier configuration. Both were checked on this
release:

- the figure scripts reproduce the submitted Figures 6 to 13 pixel for pixel;
- `analyze.py` reproduces every table and `numbers.json` exactly.

The study runners take `--workers N` (one process per run). Runs are paired
across methods by seed and by stable task identity. `PYTHONHASHSEED` is set to
0 by every runner.

## Provenance and determinism

The engine of this release has digest
`3ba9a7c5e83350c253c511335bf03763a1b33ec50d7a582226cc3cf1bfc90e60`
(`experiments.revision_harness.source_digest()`). It produced DAIP's primary
runs. The other stored runs record the engine state that produced them:

| Digest | Runs |
|---|---|
| `3ba9a7c5` (this release) | DAIP in N1, P2L (R6), CLT__nocap, CLTH__nocap, CLT_TIGHT |
| `dc1a1ffe` | ABL1; P2L R0-R5; CLT comparators |
| `37eea8b0` | SEL |
| `197ab66d` | the comparators of N1 and N2; CLT_TIGHT comparators and CLT calibration; all of `studies_earlier` |

Runs recorded under the earlier digests are reproduced by this release's
engine, as the following check shows.

**Reproduction check.** We re-ran 26 stored runs with this release's engine.
They spanned all four digests and all six methods, in the main comparison and
in the ladder, ablation, selection, regime, burden and CLT studies. The
weighted delay, on-time readiness, 95th-percentile lateness, executed actions
and accepted bundles of 25 of them were identical to the stored summaries. The
remaining run used the earlier configuration's time-limited CP-SAT selection;
it reproduced the same delay with three instead of four executed actions.

**Timing.** Results that depend on wall-clock time can vary with hardware and
load:

- CP-SAT solves are time-limited, so a binding limit can change a decision;
- measured search times depend on the machine.

All other runs are deterministic given `PYTHONHASHSEED=0` and the package
versions. The paper's times were measured on an AMD Ryzen 7 5700G (8 cores,
32 GB).

The CLT tight-turn experiment was frozen before any method was compared
(`studies_main/runs/CLT_TIGHT/manifest.json`). The manifest records the
settings and a fingerprint of every scenario, and `run_tight_bank.py` checks
each scenario against it.

## Data sources and licences

- **Code:** MIT licence (`LICENSE`).
- **Run summaries, audits and derived tables:** CC BY 4.0.
- **CLT case study inputs** (sources, retrieval dates and checksums in
  `studies_main/data/MANIFEST.csv`):
  - BTS Reporting Carrier On-Time Performance, July 2025, and the Schedule
    B-43 aircraft inventory: U.S. government public data.
    - The 32 MB on-time archive is not redistributed. Download it from the URL
      in `MANIFEST.csv` to `studies_main/data/raw/BTS_ontime_2025_07.zip` to
      rebuild the banks with `clt/build_bank.py`.
    - The derived banks (`data/work/clt_bank_2025_07.csv`,
      `clt_bank_days.csv`, `clt_turnarounds_2025_07.csv`) are included.
  - OpenStreetMap extract of the CLT apron (`data/raw/OSM_CLT_apron_overpass.json`)
    and the stand coordinates derived from it (`data/work/clt_stands_xy.json`):
    © OpenStreetMap contributors, available under the Open Database Licence (ODbL).
  - Manufacturers' airport planning manuals (CRJ900, E175): cited in the paper
    and listed in `MANIFEST.csv`, not redistributed. The service durations
    taken from them are encoded in `studies_main/clt/case.py`.

## Citation

Please cite the paper; `CITATION.cff` contains the metadata.

## Contact

Sulemana Nantogma, College of Information Engineering, Yango University,
Fuzhou, China. nantogmas@ygu.edu.cn
