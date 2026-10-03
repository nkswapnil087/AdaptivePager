import pytest

from src.adaptivepager.policies import (
    simulate_fifo,
    simulate_lru,
    simulate_optimal,
)


REFERENCE_STRING = [7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2]


@pytest.mark.parametrize(
    ("simulator", "expected_faults"),
    [
        (simulate_fifo, 10),
        (simulate_lru, 9),
        (simulate_optimal, 7),
    ],
)
def test_known_reference_string(simulator, expected_faults):
    result = simulator(REFERENCE_STRING, frame_count=3)
    assert result.faults == expected_faults
    assert result.hits == len(REFERENCE_STRING) - expected_faults


@pytest.mark.parametrize("simulator", [simulate_fifo, simulate_lru, simulate_optimal])
def test_rejects_zero_frames(simulator):
    with pytest.raises(ValueError):
        simulator([1, 2, 3], frame_count=0)


def test_empty_trace_metrics_are_well_defined():
    result = simulate_fifo([], frame_count=3)
    assert result.hits == 0
    assert result.faults == 0
    assert result.hit_ratio == 0.0
