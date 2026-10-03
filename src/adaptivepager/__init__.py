"""AdaptivePager: small page-replacement simulations and experiments."""

from .policies import SimulationResult, simulate_fifo, simulate_lru, simulate_optimal
from .workload import ShiftWorkload, generate_shift_workload

__all__ = [
    "ShiftWorkload",
    "SimulationResult",
    "generate_shift_workload",
    "simulate_fifo",
    "simulate_lru",
    "simulate_optimal",
]
