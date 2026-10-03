"""Decision-tree page replacement trained from Belady-labelled examples."""

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from .features import FEATURE_NAMES, candidate_features
from .policies import AccessEvent, SimulationResult, belady_victim


@dataclass(frozen=True)
class TrainingData:
    features: np.ndarray
    labels: np.ndarray


def build_training_data(
    traces: Iterable[Sequence[int]], frame_count: int, window_size: int = 20
) -> TrainingData:
    """Create candidate examples using past-only features and Belady labels.

    Future references are consulted *only* by ``belady_victim`` to create the
    supervised target. They never enter the feature matrix.
    """
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    feature_rows: list[list[float]] = []
    labels: list[int] = []

    for original_trace in traces:
        trace = list(original_trace)
        frames: list[int] = []
        last_seen: dict[int, int] = {}

        for index, page in enumerate(trace):
            if page not in frames:
                if len(frames) == frame_count:
                    # This is the sole future-aware operation in ML training.
                    ideal_victim = belady_victim(frames, trace, index)
                    observed_prefix = trace[:index]
                    for candidate in frames:
                        feature_rows.append(
                            candidate_features(
                                candidate,
                                index,
                                last_seen,
                                observed_prefix,
                                window_size,
                            )
                        )
                        labels.append(int(candidate == ideal_victim))
                    frames.remove(ideal_victim)
                frames.append(page)
            last_seen[page] = index

    if not feature_rows:
        raise ValueError("training traces did not produce any eviction examples")
    return TrainingData(np.asarray(feature_rows), np.asarray(labels))


class LearnedPager:
    """A small decision tree that scores each currently resident page."""

    def __init__(
        self, frame_count: int, window_size: int = 20, random_state: int = 42
    ) -> None:
        if frame_count <= 0:
            raise ValueError("frame_count must be positive")
        self.frame_count = frame_count
        self.window_size = window_size
        self.classifier = DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=8, random_state=random_state
        )
        self.is_fitted = False

    def fit(self, traces: Iterable[Sequence[int]]) -> TrainingData:
        training_data = build_training_data(
            traces, self.frame_count, self.window_size
        )
        self.classifier.fit(training_data.features, training_data.labels)
        self.is_fitted = True
        return training_data

    def simulate(self, trace: Sequence[int]) -> SimulationResult:
        """Run inference using only the prefix observed before each decision."""
        if not self.is_fitted:
            raise RuntimeError("fit must be called before simulate")

        trace_list = list(trace)
        frames: list[int] = []
        last_seen: dict[int, int] = {}
        events: list[AccessEvent] = []

        for index, page in enumerate(trace_list):
            hit = page in frames
            evicted = None
            if not hit:
                if len(frames) == self.frame_count:
                    observed_prefix = trace_list[:index]
                    feature_rows = [
                        candidate_features(
                            candidate,
                            index,
                            last_seen,
                            observed_prefix,
                            self.window_size,
                        )
                        for candidate in frames
                    ]
                    positive_column = list(self.classifier.classes_).index(1)
                    scores = self.classifier.predict_proba(feature_rows)[
                        :, positive_column
                    ]
                    # Ties prefer greater recency, then the earlier frame.
                    victim_index = max(
                        range(len(frames)),
                        key=lambda position: (
                            scores[position], feature_rows[position][0], -position
                        ),
                    )
                    evicted = frames.pop(victim_index)
                frames.append(page)
            last_seen[page] = index
            events.append(AccessEvent(index, page, hit, evicted, tuple(frames)))

        return SimulationResult("Learned", tuple(events))

    @property
    def feature_names(self) -> tuple[str, ...]:
        return FEATURE_NAMES
