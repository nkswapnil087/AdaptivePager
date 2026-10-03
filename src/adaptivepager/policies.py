"""Readable implementations of the three classical replacement policies."""

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class AccessEvent:
    """The state immediately after one page reference."""

    index: int
    page: int
    hit: bool
    evicted: int | None
    frames: tuple[int, ...]


@dataclass(frozen=True)
class SimulationResult:
    """Results and an optional step-by-step trace from one policy run."""

    policy: str
    events: tuple[AccessEvent, ...]

    @property
    def hits(self) -> int:
        return sum(event.hit for event in self.events)

    @property
    def faults(self) -> int:
        return len(self.events) - self.hits

    @property
    def hit_ratio(self) -> float:
        return self.hits / len(self.events) if self.events else 0.0


def _validate_frame_count(frame_count: int) -> None:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")


def simulate_fifo(trace: Sequence[int], frame_count: int) -> SimulationResult:
    """Evict the page that has been resident for the longest time."""
    _validate_frame_count(frame_count)
    frames: list[int] = []
    events: list[AccessEvent] = []

    for index, page in enumerate(trace):
        hit = page in frames
        evicted = None
        if not hit:
            if len(frames) == frame_count:
                evicted = frames.pop(0)
            frames.append(page)
        events.append(AccessEvent(index, page, hit, evicted, tuple(frames)))

    return SimulationResult("FIFO", tuple(events))


def simulate_lru(trace: Sequence[int], frame_count: int) -> SimulationResult:
    """Evict the resident page whose last reference is oldest."""
    _validate_frame_count(frame_count)
    frames: list[int] = []
    last_used: dict[int, int] = {}
    events: list[AccessEvent] = []

    for index, page in enumerate(trace):
        hit = page in frames
        evicted = None
        if not hit:
            if len(frames) == frame_count:
                # Frames are unique, so each timestamp is also unique here.
                evicted = min(frames, key=lambda resident: last_used[resident])
                frames.remove(evicted)
            frames.append(page)
        last_used[page] = index
        events.append(AccessEvent(index, page, hit, evicted, tuple(frames)))

    return SimulationResult("LRU", tuple(events))


def belady_victim(
    residents: Sequence[int], trace: Sequence[int], current_index: int
) -> int:
    """Return Belady's victim; only training labels and Optimal call this.

    Ties are deterministic because ``max`` keeps the first resident in frame
    order. A page that is never referenced again has infinite next-use distance.
    """
    if not residents:
        raise ValueError("residents must not be empty")

    def next_use(page: int) -> float:
        for future_index in range(current_index + 1, len(trace)):
            if trace[future_index] == page:
                return float(future_index)
        return float("inf")

    return max(residents, key=next_use)


def simulate_optimal(trace: Sequence[int], frame_count: int) -> SimulationResult:
    """Evict the page used furthest in the future (offline lower bound)."""
    _validate_frame_count(frame_count)
    # A list gives deterministic tie-breaking and clear frame-order output.
    trace_list = list(trace)
    frames: list[int] = []
    events: list[AccessEvent] = []

    for index, page in enumerate(trace_list):
        hit = page in frames
        evicted = None
        if not hit:
            if len(frames) == frame_count:
                evicted = belady_victim(frames, trace_list, index)
                frames.remove(evicted)
            frames.append(page)
        events.append(AccessEvent(index, page, hit, evicted, tuple(frames)))

    return SimulationResult("Optimal", tuple(events))
