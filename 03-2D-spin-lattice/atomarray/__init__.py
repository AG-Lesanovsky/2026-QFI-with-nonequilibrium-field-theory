"""array – semiclassical + exact QFI for driven atomic arrays.

Truncated-Wigner (Mink & Fleischhauer, arXiv:2305.19829) dynamics of a driven,
radiatively coupled 2-D atomic array and the quantum Fisher information for
sensing the Rabi frequency Ω.

    geometry    lattice positions and the Green's-tensor rate matrices Γ, J
    sde         pole-safe Cartesian rotation SDE stepper (drive + Γ/J collective)
    operators   sparse many-body operators for the MCWF reference (QuTiP-free)
    qfi         semiclassical QFI power sums + statevector MCWF monitored QFI

The semiclassical path needs only NumPy + SciPy; the MCWF reference additionally
uses SciPy sparse (no QuTiP).
"""

from .geometry import square_lattice, rate_matrices, factorize_decay_matrix
from .sde import run_array_trajectories, sample_cone_cartesian, array_step
from .qfi import (
    array_qfi_omega,
    array_qfi_powersums,
    qfi_from_powersums,
    array_monitored_qfi_mcwf,
    array_monitored_qfi_exact,
)

# Circular polarisation in the array (x–y) plane, as in Sect. 6.1 of the paper.
EP_CIRCULAR_XY = (1.0, 1.0j, 0.0)

__all__ = [
    "square_lattice",
    "rate_matrices",
    "factorize_decay_matrix",
    "run_array_trajectories",
    "sample_cone_cartesian",
    "array_step",
    "array_qfi_omega",
    "array_qfi_powersums",
    "qfi_from_powersums",
    "array_monitored_qfi_mcwf",
    "array_monitored_qfi_exact",
    "EP_CIRCULAR_XY",
]
