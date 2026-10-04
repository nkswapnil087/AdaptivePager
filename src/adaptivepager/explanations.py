"""Rule-based records and explanations for learned eviction decisions."""

from dataclasses import asdict, dataclass
import json


@dataclass(frozen=True)
class ExplanationRecord:
    """One actual full-frame eviction made by the learned policy."""

    seed: int
    access_index: int
    phase: str
    requested_page: int
    frames_before: tuple[int, ...]
    chosen_victim: int
    ideal_victim: int
    ideal_victims: tuple[int, ...]
    recency: int
    recent_frequency: int
    confidence: float
    correct: bool
    explanation: str

    def as_csv_row(self) -> dict[str, object]:
        row = asdict(self)
        row["frames_before"] = json.dumps(self.frames_before)
        row["ideal_victims"] = json.dumps(self.ideal_victims)
        return row


def generate_explanation(
    chosen_victim: int,
    recency: int,
    recent_frequency: int,
    confidence: float,
    window_size: int,
) -> str:
    """Describe observable inputs and the model score without inventing reasoning."""
    recency_unit = "reference" if recency == 1 else "references"
    frequency_unit = "time" if recent_frequency == 1 else "times"
    return (
        f"The learned policy selected page {chosen_victim} because it received "
        "the highest predicted eviction score among the current resident pages. "
        f"It was last accessed {recency} {recency_unit} ago and appeared "
        f"{recent_frequency} {frequency_unit} in the recent "
        f"{window_size}-reference window. "
        f"Model confidence: {confidence:.1%}."
    )
