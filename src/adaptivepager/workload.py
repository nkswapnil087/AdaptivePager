"""Reproducible two-phase synthetic page-reference workloads."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ShiftWorkload:
    trace: tuple[int, ...]
    shift_index: int
    seed: int


def generate_shift_workload(length: int = 400, seed: int = 42) -> ShiftWorkload:
    """Generate a locality-heavy phase followed by a wider, bursty phase.

    Phase A mostly advances through a four-page working set, with hot-page
    repeats and occasional nearby noise. Phase B switches among short-lived
    five-page burst sets selected from sixteen pages, plus global random reads.
    """
    if length < 20:
        raise ValueError("length must be at least 20")
    rng = np.random.default_rng(seed)
    shift_index = length // 2

    phase_a: list[int] = []
    local_pages = np.array([0, 1, 2, 3])
    cursor = int(rng.integers(0, len(local_pages)))
    for _ in range(shift_index):
        choice = rng.random()
        if choice < 0.72:
            page = int(local_pages[cursor])
            cursor = (cursor + 1) % len(local_pages)
        elif choice < 0.94:
            page = int(rng.choice(local_pages[:2]))
        else:
            page = int(rng.choice([4, 5]))
        phase_a.append(page)

    phase_b: list[int] = []
    wide_pages = np.arange(16)
    burst_pages = rng.choice(wide_pages, size=5, replace=False)
    burst_remaining = 0
    for _ in range(length - shift_index):
        if burst_remaining == 0:
            burst_pages = rng.choice(wide_pages, size=5, replace=False)
            burst_remaining = int(rng.integers(10, 25))
        if rng.random() < 0.68:
            page = int(rng.choice(burst_pages))
        else:
            page = int(rng.choice(wide_pages))
        phase_b.append(page)
        burst_remaining -= 1

    return ShiftWorkload(tuple(phase_a + phase_b), shift_index, seed)
