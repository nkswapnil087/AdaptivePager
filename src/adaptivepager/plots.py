"""Small, reproducible figures for the primary workload."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


POLICY_ORDER = ["FIFO", "LRU", "Optimal", "Learned"]
PHASE_ORDER = ["before_shift", "after_shift"]


def create_figures(primary_results: pd.DataFrame, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    subset = primary_results[primary_results["phase"].isin(PHASE_ORDER)]
    files: list[Path] = []

    for metric, ylabel, filename in (
        ("faults", "Page faults", "page_faults_by_phase.png"),
        ("hit_ratio", "Hit ratio", "hit_ratio_by_phase.png"),
    ):
        pivot = subset.pivot(index="policy", columns="phase", values=metric)
        pivot = pivot.reindex(index=POLICY_ORDER, columns=PHASE_ORDER)
        axis = pivot.plot(kind="bar", figsize=(9.2, 4.8), rot=0, width=0.75)
        axis.set_title(f"{ylabel} before and after the workload shift")
        axis.set_xlabel("Replacement policy")
        axis.set_ylabel(ylabel)
        axis.legend(
            ["Before shift", "After shift"],
            title="Phase",
            loc="upper left",
            bbox_to_anchor=(1.01, 1),
        )
        axis.grid(axis="y", alpha=0.25)
        if metric == "hit_ratio":
            axis.set_ylim(0, 1)
        plt.tight_layout()
        output_file = output_dir / filename
        plt.savefig(output_file, dpi=160)
        plt.close()
        files.append(output_file)
    return files
