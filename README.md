# AdaptivePager

A learning-augmented page-replacement simulator that compares FIFO, LRU,
Optimal, and a lightweight Decision Tree eviction policy under shifting
memory-access patterns.

## Overview

AdaptivePager is a small, reproducible systems and machine-learning experiment
about virtual-memory page replacement. It asks how classical policies and a
learned eviction policy behave when the page-access pattern changes partway
through execution. The implementation is intentionally compact and readable so
that each policy and experimental step can be explained in a short Operating
Systems walkthrough.

## Classical Policies

- **FIFO** evicts the page that entered memory earliest.
- **LRU** evicts the resident page whose most recent access is oldest.
- **Optimal (Belady)** evicts the page whose next use is furthest in the future,
  or a page that is never used again. Because it requires future knowledge, it
  is an offline theoretical benchmark rather than a deployable policy.

## Learned Policy

The learned policy uses a small scikit-learn `DecisionTreeClassifier`. During
training, every resident page at an eviction point becomes a candidate example,
and Belady identifies the ideal victim to provide the target label. The model
sees only two past-observable features:

1. **Recency:** accesses elapsed since the candidate page was last referenced.
2. **Recent access frequency:** references to the candidate in the previous 20
   accesses.

During testing, Belady is not consulted. The tree scores each resident page and
evicts the candidate with the highest predicted probability, using deterministic
tie-breaking.

> Belady makes decisions using future page references, while the learned model
> must infer useful eviction behaviour using only past access history.

The feature function accepts only the prefix observed before the current
decision. Next-use distance, future frequency, remaining-trace statistics, and
other future information never enter the model inputs.

## Workload Shift

Each default trace contains 400 accesses and shifts at index 200:

- **First half:** a small working set with strong sequential/local behaviour,
  hot-page repeats, and occasional nearby noise.
- **Second half:** a wider page range with changing short-lived burst sets and
  global random accesses.

The second phase remains a plausible synthetic workload, but historical access
patterns are less stable than in the first phase.

## Experimental Method

- Ten independent training traces use seeds `10042`–`10051`.
- Seven unseen evaluation traces use seeds `42`–`48`.
- Seed `42` is retained as the primary demonstration trace.
- Every policy receives the same evaluation trace and the same default four
  frames for each run.
- Measurements are calculated before the shift, after the shift, and overall.

The fixed seeds make the experiment reproducible while keeping training and
test data separate.

## Metrics

- **Hits:** references satisfied by a page already in a frame.
- **Page faults:** references requiring a page to be loaded.
- **Hit ratio:** hits divided by total accesses in the measured phase.

## Key Findings

Optimal performs best because it knows future page references. In the default
experiment, LRU and FIFO both outperform the current learned model overall, and
the learned model performs worse after the workload shift. This is a valid
experimental result rather than a project failure.

The learned policy works only from past and present access history, while
Belady's Optimal policy has perfect knowledge of future references. This
information asymmetry is a major reason for the large performance gap. The
learned feature set is also intentionally limited to recency and frequency, and
the workload shift reduces the reliability of historical patterns. In addition,
the learned policy can create frame states during inference that differ from the
Belady-generated states represented in its training examples.

Claims are limited to these synthetic traces; the results are not evidence that
one heuristic will dominate on real production workloads.

## Primary Result

The reproduced seed-42 result is:

| Policy | Before Faults | After Faults | Overall Faults | Overall Hit Ratio |
| --- | ---: | ---: | ---: | ---: |
| FIFO | 34 | 115 | 149 | 0.6275 |
| LRU | 32 | 115 | 147 | 0.6325 |
| Optimal | 17 | 77 | 94 | 0.7650 |
| Learned | 35 | 124 | 159 | 0.6025 |

Across all seven evaluation runs, the mean overall page-fault counts are:

- Optimal: `97.71`
- LRU: `157.86`
- FIFO: `160.14`
- Learned: `175.71`

## Core Code Overview

### `policies.py`

Implements FIFO, LRU, and Optimal (Belady) page replacement. FIFO removes the
oldest resident page, LRU removes the least recently used page, and Optimal
removes the page whose next use is furthest in the future. The file records
hits, faults, evictions, and frame states for every access. It provides the
classical baselines for comparison. Belady's victim-selection function is also
reused during training to generate ideal eviction labels.

### `features.py`

Generates the two past-observable features used by the learned policy: recency
and recent access frequency. Recency measures how long ago a page was last
referenced. Recent frequency counts how often the page appeared in the recent
history window. Only the observed prefix of the trace is accepted. This
explicitly prevents future-information leakage.

### `learned.py`

Implements the Decision Tree-based page-replacement policy. During training,
Belady identifies the ideal victim and provides the target label. The model
itself sees only recency and recent-frequency features. During testing, Belady
is no longer used. The model scores each resident page and evicts the candidate
with the highest predicted probability.

### `workload.py`

Generates reproducible two-phase synthetic page-reference workloads. The first
half uses a small working set with strong locality. The second half expands to a
wider page range with changing bursts and random accesses. This deliberate
shift tests how the policies react when the workload changes. Seeds make the
workloads reproducible.

### `experiment.py`

Coordinates training, testing, evaluation, result export, and figure
generation. It trains the learned model on separate synthetic traces. It
evaluates FIFO, LRU, Optimal, and Learned on identical unseen traces. Metrics
are computed before the shift, after the shift, and overall. Results are saved
as detailed and aggregate CSV files and plotted as figures.

## Installation

Python 3.10 or newer is recommended. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

On Debian or Ubuntu, install the distribution's `python3-venv` package first if
the standard library reports that `ensurepip` is unavailable.

## Running the Experiment

```bash
python3 -m src.adaptivepager.experiment
```

This trains the model, evaluates all four policies, regenerates the CSV files
and figures, and prints the primary trace results. Optional flags include
`--frames`, `--length`, `--seed`, `--test-runs`, `--training-traces`, and
`--output-dir`.

## Running Tests

```bash
python3 -m pytest -q
```

The tests cover known classical-policy fault counts, invalid frame counts,
past-only feature behaviour, Belady labels, learned inference, and workload
reproducibility.

## Repository Structure

```text
AdaptivePager/
├── src/
│   └── adaptivepager/
│       ├── policies.py
│       ├── features.py
│       ├── learned.py
│       ├── workload.py
│       ├── experiment.py
│       └── plots.py
├── tests/
├── results/
│   ├── policy_results.csv
│   ├── aggregate_results.csv
│   └── figures/
├── README.md
├── requirements.txt
└── LICENSE
```

## Results and Figures

- `results/policy_results.csv` contains per-seed, per-policy metrics for both
  phases and the complete trace.
- `results/aggregate_results.csv` contains means and sample standard deviations
  across the seven evaluation seeds.
- `results/figures/page_faults_by_phase.png` compares page faults before and
  after the shift for the primary trace.
- `results/figures/hit_ratio_by_phase.png` compares the corresponding hit
  ratios.

All committed measurements and plots are generated by the repository code, not
entered manually. Running the default experiment command reproduces them.

## Limitations

- The learned policy has only two historical features.
- Belady uses future information while learned inference cannot.
- The workload is synthetic rather than a production memory trace.
- Decision Tree performance depends on the training distribution.
- Training-state mismatch can occur because Belady creates the frame states
  used for training, while the learned model creates its own states during
  inference.
- This is a simulator, not a kernel implementation.

## AI Assistance Disclosure

AI coding assistance was used for project scaffolding, implementation support,
debugging, documentation, and review. All code was reviewed for understanding
and correctness. Experiments were executed from the repository implementation,
and the final interpretation of results remains the author's responsibility.

## License

AdaptivePager is available under the MIT License.
