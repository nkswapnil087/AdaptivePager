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


@dataclass(frozen=True)
class EvictionDecision:
    """The model scores and selected index for one full-frame page fault."""

    victim_index: int
    victim: int
    feature_rows: tuple[tuple[float, ...], ...]
    scores: tuple[float, ...]

    @property
    def confidence(self) -> float:
        """Probability of class 1 for the candidate actually selected."""
        return self.scores[self.victim_index]

    @property
    def chosen_features(self) -> tuple[float, ...]:
        return self.feature_rows[self.victim_index]


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

    def choose_victim(
        self,
        frames: Sequence[int],
        current_index: int,
        last_seen: dict[int, int],
        observed_prefix: Sequence[int],
    ) -> EvictionDecision:
        """Score resident pages and reproduce the learned victim selection."""
        if not self.is_fitted:
            raise RuntimeError("fit must be called before choosing a victim")
        if len(frames) != self.frame_count:
            raise ValueError("victim selection requires a full frame set")

        feature_rows = [
            candidate_features(
                candidate,
                current_index,
                last_seen,
                observed_prefix,
                self.window_size,
            )
            for candidate in frames
        ]
        positive_column = list(self.classifier.classes_).index(1)
        probabilities = self.classifier.predict_proba(feature_rows)[
            :, positive_column
        ]

        # Ties prefer greater recency, then the earlier frame. This is the same
        # deterministic rule used by the original learned-policy simulation.
        victim_index = max(
            range(len(frames)),
            key=lambda position: (
                probabilities[position], feature_rows[position][0], -position
            ),
        )
        return EvictionDecision(
            victim_index=victim_index,
            victim=frames[victim_index],
            feature_rows=tuple(tuple(row) for row in feature_rows),
            scores=tuple(float(score) for score in probabilities),
        )

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
                    decision = self.choose_victim(
                        frames,
                        index,
                        last_seen,
                        observed_prefix,
                    )
                    evicted = frames.pop(decision.victim_index)
                frames.append(page)
            last_seen[page] = index
            events.append(AccessEvent(index, page, hit, evicted, tuple(frames)))

        return SimulationResult("Learned", tuple(events))

    @property
    def feature_names(self) -> tuple[str, ...]:
        return FEATURE_NAMES
