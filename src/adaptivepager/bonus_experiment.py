"""Run the optional explanation-confidence audit for learned evictions."""

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .explanations import ExplanationRecord, generate_explanation
from .learned import LearnedPager
from .workload import generate_shift_workload


DETAIL_COLUMNS = [
    "seed",
    "access_index",
    "phase",
    "requested_page",
    "frames_before",
    "chosen_victim",
    "ideal_victim",
    "ideal_victims",
    "recency",
    "recent_frequency",
    "confidence",
    "correct",
    "explanation",
]


def belady_optimal_victims(
    residents: Sequence[int], trace: Sequence[int], current_index: int
) -> tuple[int, ...]:
    """Return every resident tied for Belady's maximum next-use distance."""
    if not residents:
        raise ValueError("residents must not be empty")

    next_uses: dict[int, float] = {}
    for resident in residents:
        next_use = float("inf")
        for future_index in range(current_index + 1, len(trace)):
            if trace[future_index] == resident:
                next_use = float(future_index)
                break
        next_uses[resident] = next_use

    maximum_next_use = max(next_uses.values())
    return tuple(
        resident
        for resident in residents
        if next_uses[resident] == maximum_next_use
    )


def collect_explanation_records(
    pager: LearnedPager,
    trace: Sequence[int],
    seed: int,
    shift_index: int,
) -> list[ExplanationRecord]:
    """Replay learned inference and audit each actual eviction after selection."""
    trace_list = list(trace)
    frames: list[int] = []
    last_seen: dict[int, int] = {}
    records: list[ExplanationRecord] = []

    for index, page in enumerate(trace_list):
        if page not in frames:
            if len(frames) == pager.frame_count:
                frames_before = tuple(frames)
                observed_prefix = trace_list[:index]

                # The model choice is finalized using past-only inputs.
                decision = pager.choose_victim(
                    frames_before,
                    index,
                    last_seen,
                    observed_prefix,
                )
                chosen_victim = decision.victim
                recency, recent_frequency = decision.chosen_features
                confidence = decision.confidence

                # Only now may future-aware Belady evaluate that fixed choice.
                # All pages tied at the maximum next-use distance are optimal.
                # This audit never affects frames, scores, or tie-breaking.
                ideal_victims = belady_optimal_victims(
                    frames_before, trace_list, index
                )
                ideal_victim = ideal_victims[0]
                correct = chosen_victim in ideal_victims
                records.append(
                    ExplanationRecord(
                        seed=seed,
                        access_index=index,
                        phase=(
                            "before_shift" if index < shift_index else "after_shift"
                        ),
                        requested_page=page,
                        frames_before=frames_before,
                        chosen_victim=chosen_victim,
                        ideal_victim=ideal_victim,
                        ideal_victims=ideal_victims,
                        recency=int(recency),
                        recent_frequency=int(recent_frequency),
                        confidence=confidence,
                        correct=correct,
                        explanation=generate_explanation(
                            chosen_victim,
                            int(recency),
                            int(recent_frequency),
                            confidence,
                            pager.window_size,
                        ),
                    )
                )

                removed_page = frames.pop(decision.victim_index)
                if removed_page != chosen_victim:
                    raise AssertionError("recorded victim differs from simulation victim")
            frames.append(page)
        last_seen[page] = index

    return records


def confidence_summary(details: pd.DataFrame) -> pd.DataFrame:
    """Summarize correctness and confidence by phase and overall."""
    rows: list[dict[str, int | float | str]] = []
    for phase in ("before_shift", "after_shift", "overall"):
        subset = details if phase == "overall" else details[details["phase"] == phase]
        correct = subset[subset["correct"]]
        incorrect = subset[~subset["correct"]]
        rows.append(
            {
                "phase": phase,
                "total_decisions": len(subset),
                "correct_decisions": len(correct),
                "incorrect_decisions": len(incorrect),
                "decision_accuracy": subset["correct"].mean(),
                "mean_confidence_overall": subset["confidence"].mean(),
                "mean_confidence_correct": correct["confidence"].mean(),
                "mean_confidence_incorrect": incorrect["confidence"].mean(),
            }
        )
    return pd.DataFrame(rows)


def confidence_bins(details: pd.DataFrame) -> pd.DataFrame:
    """Compare average model probability with observed accuracy in fixed bins."""
    labels = ["0.00-0.50", "0.50-0.70", "0.70-0.90", "0.90-1.00"]
    working = details.copy()
    working["confidence_bin"] = pd.cut(
        working["confidence"],
        bins=[0.0, 0.5, 0.7, 0.9, 1.0],
        labels=labels,
        include_lowest=True,
    )
    return (
        working.groupby("confidence_bin", observed=True)
        .agg(
            decisions=("correct", "size"),
            mean_confidence=("confidence", "mean"),
            fraction_correct=("correct", "mean"),
        )
        .reset_index()
    )


def create_confidence_figure(details: pd.DataFrame, output_file: Path) -> None:
    """Plot mean selected-victim confidence by actual correctness."""
    means = details.groupby("correct")["confidence"].mean()
    values = [means.get(True, float("nan")), means.get(False, float("nan"))]
    figure, axis = plt.subplots(figsize=(6.8, 4.6))
    bars = axis.bar(
        ["Correct", "Incorrect"], values, color=["#2e7d32", "#c62828"], width=0.6
    )
    axis.set_title("Learned-policy confidence vs. Belady correctness")
    axis.set_xlabel("Learned eviction decision")
    axis.set_ylabel("Mean predicted probability for selected victim")
    axis.set_ylim(0, 1)
    axis.grid(axis="y", alpha=0.25)
    axis.bar_label(bars, fmt="%.3f", padding=3)
    figure.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_file, dpi=160)
    plt.close(figure)


def run_bonus_experiment(
    frame_count: int = 4,
    trace_length: int = 400,
    primary_seed: int = 42,
    test_runs: int = 7,
    training_traces: int = 10,
    output_dir: Path = Path("results"),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train as in the base experiment and audit unseen learned-policy runs."""
    training_seeds = [primary_seed + 10_000 + i for i in range(training_traces)]
    test_seeds = [primary_seed + i for i in range(test_runs)]
    training_sequences = [
        generate_shift_workload(trace_length, seed).trace for seed in training_seeds
    ]
    pager = LearnedPager(frame_count=frame_count, random_state=primary_seed)
    pager.fit(training_sequences)

    records: list[ExplanationRecord] = []
    for seed in test_seeds:
        workload = generate_shift_workload(trace_length, seed)
        records.extend(
            collect_explanation_records(
                pager, workload.trace, seed, workload.shift_index
            )
        )

    details = pd.DataFrame(
        [record.as_csv_row() for record in records], columns=DETAIL_COLUMNS
    )
    summary = confidence_summary(details)
    bins = confidence_bins(details)

    output_dir.mkdir(parents=True, exist_ok=True)
    details.to_csv(output_dir / "explanation_confidence.csv", index=False)
    summary.to_csv(output_dir / "explanation_confidence_summary.csv", index=False)
    bins.to_csv(output_dir / "explanation_confidence_bins.csv", index=False)
    figure_file = output_dir / "figures" / "confidence_vs_correctness.png"
    create_confidence_figure(details, figure_file)

    print(f"Evaluated {len(details):,} learned eviction decisions.")
    print("\nConfidence and correctness summary:")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nConfidence bins:")
    print(bins.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nWrote bonus results to {output_dir.resolve()}")
    print(f"Figure: {figure_file.name}")
    return details, summary, bins


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
    run_bonus_experiment(
        frame_count=arguments.frames,
        trace_length=arguments.length,
        primary_seed=arguments.seed,
        test_runs=arguments.test_runs,
        training_traces=arguments.training_traces,
        output_dir=arguments.output_dir,
    )
