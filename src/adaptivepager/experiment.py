"""Run the complete AdaptivePager experiment from one module command."""

import argparse
from pathlib import Path
from typing import Callable

import pandas as pd

from .learned import LearnedPager
from .plots import create_figures
from .policies import SimulationResult, simulate_fifo, simulate_lru, simulate_optimal
from .workload import generate_shift_workload


def phase_row(
    result: SimulationResult, seed: int, phase: str, start: int, end: int
) -> dict[str, int | float | str]:
    events = result.events[start:end]
    hits = sum(event.hit for event in events)
    accesses = len(events)
    return {
        "policy": result.policy,
        "seed": seed,
        "phase": phase,
        "accesses": accesses,
        "hits": hits,
        "faults": accesses - hits,
        "hit_ratio": hits / accesses if accesses else 0.0,
    }


def evaluate_result(
    result: SimulationResult, seed: int, shift_index: int
) -> list[dict[str, int | float | str]]:
    length = len(result.events)
    return [
        phase_row(result, seed, "before_shift", 0, shift_index),
        phase_row(result, seed, "after_shift", shift_index, length),
        phase_row(result, seed, "overall", 0, length),
    ]


def run_experiment(
    frame_count: int = 4,
    trace_length: int = 400,
    primary_seed: int = 42,
    test_runs: int = 7,
    training_traces: int = 10,
    output_dir: Path = Path("results"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train once on independent traces, then evaluate fixed unseen seeds."""
    if test_runs <= 0 or training_traces <= 0:
        raise ValueError("test_runs and training_traces must be positive")

    # The large offset makes train/test seed separation obvious and reproducible.
    training_seeds = [primary_seed + 10_000 + i for i in range(training_traces)]
    test_seeds = [primary_seed + i for i in range(test_runs)]
    training_sequences = [
        generate_shift_workload(trace_length, seed).trace for seed in training_seeds
    ]

    learned = LearnedPager(frame_count=frame_count, random_state=primary_seed)
    training_data = learned.fit(training_sequences)

    classical: list[Callable[[list[int], int], SimulationResult]] = [
        simulate_fifo,
        simulate_lru,
        simulate_optimal,
    ]
    rows: list[dict[str, int | float | str]] = []
    for seed in test_seeds:
        workload = generate_shift_workload(trace_length, seed)
        trace = list(workload.trace)
        results = [policy(trace, frame_count) for policy in classical]
        results.append(learned.simulate(trace))
        for result in results:
            rows.extend(evaluate_result(result, seed, workload.shift_index))

    detailed = pd.DataFrame(rows)
    aggregate = (
        detailed.groupby(["policy", "phase"], sort=False)
        .agg(
            runs=("seed", "count"),
            mean_hits=("hits", "mean"),
            std_hits=("hits", "std"),
            mean_faults=("faults", "mean"),
            std_faults=("faults", "std"),
            mean_hit_ratio=("hit_ratio", "mean"),
            std_hit_ratio=("hit_ratio", "std"),
        )
        .reset_index()
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    detailed.to_csv(output_dir / "policy_results.csv", index=False)
    aggregate.to_csv(output_dir / "aggregate_results.csv", index=False)
    primary = detailed[detailed["seed"] == primary_seed]
    figure_files = create_figures(primary, output_dir / "figures")

    print(
        f"Trained DecisionTreeClassifier on {len(training_data.labels):,} "
        f"candidate examples from seeds {training_seeds[0]}-{training_seeds[-1]}."
    )
    print(f"Evaluated seeds {test_seeds}; primary demonstration seed: {primary_seed}.")
    print("\nPrimary trace results:")
    print(
        primary[["policy", "phase", "hits", "faults", "hit_ratio"]]
        .to_string(index=False, float_format=lambda value: f"{value:.3f}")
    )
    print(f"\nWrote results to {output_dir.resolve()}")
    print("Figures: " + ", ".join(path.name for path in figure_files))
    return detailed, aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--length", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-runs", type=int, default=7)
    parser.add_argument("--training-traces", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run_experiment(
        frame_count=arguments.frames,
        trace_length=arguments.length,
        primary_seed=arguments.seed,
        test_runs=arguments.test_runs,
        training_traces=arguments.training_traces,
        output_dir=arguments.output_dir,
    )
