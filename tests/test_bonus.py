import src.adaptivepager.bonus_experiment as bonus_experiment
from src.adaptivepager.learned import LearnedPager
from src.adaptivepager.workload import generate_shift_workload


def trained_pager_and_workload():
    training = [generate_shift_workload(120, seed).trace for seed in (100, 101, 102)]
    pager = LearnedPager(frame_count=4, random_state=42)
    pager.fit(training)
    return pager, generate_shift_workload(120, seed=42)


def test_bonus_records_real_probabilities_explanations_and_correctness():
    pager, workload = trained_pager_and_workload()
    records = bonus_experiment.collect_explanation_records(
        pager, workload.trace, workload.seed, workload.shift_index
    )

    assert records
    for record in records:
        assert 0.0 <= record.confidence <= 1.0
        assert record.explanation.strip()
        assert record.correct == (record.chosen_victim in record.ideal_victims)


def test_all_equally_optimal_belady_victims_are_accepted():
    residents = (1, 2, 3)
    # None of the residents appears after the current request, so all are tied
    # at infinite next-use distance and are valid Optimal victims.
    trace = [1, 2, 3, 4, 9, 9]
    assert bonus_experiment.belady_optimal_victims(residents, trace, 3) == residents


def test_bonus_uses_the_same_victims_as_base_learned_simulation():
    pager, workload = trained_pager_and_workload()
    base_result = pager.simulate(workload.trace)
    records = bonus_experiment.collect_explanation_records(
        pager, workload.trace, workload.seed, workload.shift_index
    )
    base_evictions = {
        event.index: event.evicted
        for event in base_result.events
        if event.evicted is not None
    }

    assert {record.access_index: record.chosen_victim for record in records} == (
        base_evictions
    )


def test_belady_audit_cannot_change_learned_choices(monkeypatch):
    pager, workload = trained_pager_and_workload()
    original = bonus_experiment.collect_explanation_records(
        pager, workload.trace, workload.seed, workload.shift_index
    )

    monkeypatch.setattr(
        bonus_experiment,
        "belady_optimal_victims",
        lambda residents, trace, current_index: tuple(residents),
    )
    altered_audit = bonus_experiment.collect_explanation_records(
        pager, workload.trace, workload.seed, workload.shift_index
    )

    assert [record.chosen_victim for record in altered_audit] == [
        record.chosen_victim for record in original
    ]
