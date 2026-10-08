"""btc (semiclassical subset) - Boundary Time Crystal QFI on a cluster.

Trimmed copy of the main ``btc`` package, carrying only what the semiclassical
N-convergence campaign needs:

    sde         SDE stepper for the collective dynamics
    operators   collective spin operators (QuTiP; needed only by the quantum QFI)
    qfi         quantum and semiclassical quantum Fisher information

Importing this package needs only NumPy + SciPy (+ tqdm for the progress bar);
the QuTiP-backed quantum QFI functions defer their ``operators`` import to call
time, so the semiclassical route runs without QuTiP installed.
"""

from .sde import (
    run_collective_trajectories,
    decay_matrix_dicke,
)
from .qfi import (
    collective_qfi_omega,
    collective_qfi_powersums,
    qfi_from_powersums,
    quantum_qfi_omega,
    quantum_monitored_qfi_omega,
)

__all__ = [
    "run_collective_trajectories",
    "decay_matrix_dicke",
    "collective_qfi_omega",
    "collective_qfi_powersums",
    "qfi_from_powersums",
    "quantum_qfi_omega",
    "quantum_monitored_qfi_omega",
]
