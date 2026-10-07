"""btc (semiclassical subset) – Boundary Time Crystal QFI on a cluster.

This is a trimmed copy of the main ``btc`` package, carrying only what the
semiclassical N-convergence run needs:

    sde         Stratonovich SDE steppers (drift / diffusive collective schemes)
    operators   collective spin operators (QuTiP; needed only by the quantum QFI)
    qfi         quantum and semiclassical quantum Fisher information

Importing this package requires only NumPy + SciPy (+ tqdm for the progress
bar).  The QuTiP-backed quantum QFI functions still work if QuTiP is installed
— their ``operators`` import is deferred to call time — but the diffusive
semiclassical route used by ``run_N_convergence.py`` never touches it.
"""

from .sde import (
    run_collective_trajectories,
    decay_matrix_dicke,
)
from .qfi import (
    semiclassical_qfi_omega,
    collective_qfi_omega,
    collective_qfi_powersums,
    qfi_from_powersums,
    quantum_qfi_omega,
    quantum_monitored_qfi_omega,
    quantum_monitored_qfi_omega_mcwf,
)

__all__ = [
    "run_collective_trajectories",
    "decay_matrix_dicke",
    "semiclassical_qfi_omega",
    "collective_qfi_omega",
    "collective_qfi_powersums",
    "qfi_from_powersums",
    "quantum_qfi_omega",
    "quantum_monitored_qfi_omega",
    "quantum_monitored_qfi_omega_mcwf",
]
