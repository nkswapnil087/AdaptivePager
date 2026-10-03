from src.adaptivepager.workload import generate_shift_workload


def test_workload_is_reproducible_and_records_midpoint():
    first = generate_shift_workload(length=200, seed=42)
    second = generate_shift_workload(length=200, seed=42)
    assert first == second
    assert first.shift_index == 100


def test_second_phase_uses_a_wider_page_range():
    workload = generate_shift_workload(length=400, seed=42)
    before = set(workload.trace[: workload.shift_index])
    after = set(workload.trace[workload.shift_index :])
    assert len(before) <= 6
    assert len(after) > len(before)
