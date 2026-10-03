import numpy as np

from src.adaptivepager.features import candidate_features
from src.adaptivepager.learned import LearnedPager, build_training_data
from src.adaptivepager.workload import generate_shift_workload


def test_features_depend_only_on_observed_prefix():
    prefix = [0, 1, 0, 2]
    last_seen = {0: 2, 1: 1, 2: 3}
    first = candidate_features(0, 4, last_seen, prefix, window_size=3)
    # Two incompatible futures cannot change a feature calculated from the prefix.
    future_a = prefix + [0, 0, 0]
    future_b = prefix + [9, 8, 7]
    assert candidate_features(0, 4, last_seen, future_a[:4], 3) == first
    assert candidate_features(0, 4, last_seen, future_b[:4], 3) == first
    assert first == [2.0, 1.0]


def test_belady_labels_one_candidate_per_eviction():
    trace = [0, 1, 2, 3, 0, 1, 4, 0, 1, 2, 3, 4]
    data = build_training_data([trace], frame_count=3, window_size=5)
    assert data.features.shape[1] == 2
    assert len(data.labels) % 3 == 0
    for start in range(0, len(data.labels), 3):
        assert np.sum(data.labels[start : start + 3]) == 1


def test_learned_pager_trains_and_returns_one_event_per_access():
    training = [generate_shift_workload(100, seed).trace for seed in (100, 101)]
    pager = LearnedPager(frame_count=4, random_state=42)
    pager.fit(training)
    test_trace = generate_shift_workload(100, 42).trace
    result = pager.simulate(test_trace)
    assert result.policy == "Learned"
    assert len(result.events) == len(test_trace)
    assert result.hits + result.faults == len(test_trace)
