"""Past-observable features for candidate eviction pages."""

from collections import Counter
from typing import Sequence

FEATURE_NAMES = ("recency", "recent_frequency")


def candidate_features(
    page: int,
    current_index: int,
    last_seen: dict[int, int],
    past_trace: Sequence[int],
    window_size: int,
) -> list[float]:
    """Calculate recency and frequency without reading future references.

    ``past_trace`` must end immediately before ``current_index``. Keeping this
    boundary in the API makes the no-future-information rule visible.
    """
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if len(past_trace) != current_index:
        raise ValueError("past_trace must contain exactly the observed prefix")
    if page not in last_seen:
        raise ValueError("candidate page has not appeared in the observed prefix")

    window = past_trace[max(0, current_index - window_size) : current_index]
    frequency = Counter(window)[page]
    recency = current_index - last_seen[page]
    return [float(recency), float(frequency)]
