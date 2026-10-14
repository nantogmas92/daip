# DAIP engine

The engine implements the operational dependency model, Algorithms 1-3 of the
paper, the closed-loop turnaround simulator and the harness that runs every
study. Scripts add this folder to the import path, so its subpackages are
imported as `core`, `airport`, `algorithms`, `simulation`, `optimization` and
`experiments`.

## Modules

| Folder | Contents |
|---|---|
| `core/` | typed state variables; max-plus timing operators with binding sets, including capacity and compatibility operators (`factors.py`); the dependency graph with journal, rollback and sandbox (`graph.py`); condensation into strongly connected components (`scc.py`); sparse action sensitivities (`sensitivities.py`); typed events, messages and decision records (`messages.py`) |
| `airport/` | flights and readiness targets (`flights.py`, `calibration.py`: stand-feasible minimum turnaround); service templates (`turnaround_tasks.py`); ground support equipment, crews, stands and service positions; compatibility profiles (`safety_constraints.py`); apron road networks (`apron_graph.py`: layout `geometry` = paper Layout A, `corridor` = Layout B); the controller's observation layer; the adapter from the physical state to the dependency model (`adapter.py`) |
| `simulation/` | discrete-event execution of the turnarounds (`event_engine.py`), service execution and movement, disturbance families (`disturbances.py`), deterministic replay |
| `algorithms/` | Algorithm 1, incremental maintenance (`algorithm1_propagation.py`); counterfactual operators (`counterfactual.py`); violated targets (`violations.py`); Algorithm 2, binding-guided candidate search (`algorithm2_candidates.py`); pair interactions (`interaction.py`); Algorithm 3, selection (`algorithm3_selection.py`), joint validation (`validation.py`) and completion (`completion.py`); the planning oracle that counts evaluations against the budget Q (`literature_recovery.py`) |
| `optimization/` | selection backends: beam search (`beam.py`, primary) and static top-k, greedy marginal selection, CP-SAT (`cpsat.py`) and an optional MILP reference |
| `experiments/` | `revision_harness.py`: the paired closed-loop harness (scenario specification `Spec`, task-keyed noise, common replanning gate, counted planning evaluations, per-epoch and per-action logs, run summary, `source_digest()`); `schedule_audit.py`: independent audit of realised schedules |
| `tests/` | regression and integration tests: `python -m pytest tests -q` from this folder |

The code predates the paper's terminology in places:

- the *factor graph* and its *factors* are the operational dependency model
  and its timing operators;
- *causal* or *influence* metadata is the binding provenance;
- `perho` is the internal name of an adapted comparator that the paper does
  not report.

`experiments/runner.py`, `experiments/trace.py`, `experiments/metrics.py`,
`algorithms/control.py` and the command-line part of
`experiments/compare_literature.py` belong to an earlier experiment driver.
They are kept because the harness and tests import parts of them, and they do
not produce the paper's results.

## Running a scenario

```python
import sys; sys.path.insert(0, "engine")
from experiments import revision_harness as H

spec = H.Spec(study="demo", arm="proposed", seed=1000, layout="geometry", method="proposed")
summary, controller = H.run(spec)
print(summary["delay_w_per_flight"], summary["on_time_rate"], summary["interventions"])
```

The defaults of `Spec` are the primary configuration:

- beam selection, `query_limit=130`, `budget=4`, `eps_gain=0.5`;
- `eps_lambda=0.25`, `max_depth=40`;
- 8 pair evaluations (`PAIR_BUDGET`);
- a 20-flight bank over 150 min.

`method` selects the policy:

- `proposed` is DAIP and `none` is no repair;
- the comparators of `studies_main/baselines` are registered with
  `native.install()`.

The fields `representation`, `propagation`, `candidates`, `validation`,
`completion`, `backend`, `ring` and `urgency` define the ladder and the
ablations.

## Integrity digest

`source_digest()` hashes every runtime module (the folders `airport`,
`algorithms`, `core`, `optimization` and `simulation`, plus
`experiments/revision_harness.py`, `compare_literature.py` and
`schedule_audit.py`). Every run summary stores it, so a summary can be traced
to an exact engine state. The digest of this release is
`3ba9a7c5e83350c253c511335bf03763a1b33ec50d7a582226cc3cf1bfc90e60`.
Changing any runtime module, including a comment, changes the digest.

## Scope

The incremental-versus-global agreement checked by the tests and audits holds
inside the deterministic, acyclic planning model. It does not establish field
accuracy, operational safety, global optimality or complete candidate
coverage.
