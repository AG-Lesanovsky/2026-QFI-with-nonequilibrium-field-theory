"""dho – global QFI of a single damped/pumped bosonic mode.

Three estimators of the global quantum Fisher information for sensing the pump
rate γ↑ of  H = ε a†a,  L₁ = √γ↑ a†,  L₂ = √γ↓ a, plus the driver that compares
them.  Plotting lives in the notebook, not here, so the figures stay editable
without touching the package:

    states           Fock-space initial states + Wigner phase-space samples
    analytical       exact closed-form (Keldysh) QFI  — the ground truth
    exact_numerical  deformed-Liouvillian finite-difference QFI (sparse superop)
    semiclassical    Keldysh stochastic-trajectory QFI + finite-sample error band
    meanfield        exact vs semiclassical ⟨a(t)⟩ diagnostic (SDE drift/noise check)
    compare          one-parameter-set driver bundling the three estimators

The semiclassical + analytical paths need only NumPy/SciPy; the exact-numerical
path additionally uses SciPy sparse.  No QuTiP.
"""

from .states import thermal_state, coherent_state, sample_wigner_alpha
from .analytical import qfi_keldysh, qfi_long_time, steady_state_occupation
from .exact_numerical import (
    deformed_liouvillian,
    global_qfi_deformed,
    fock_truncation,
)
from .semiclassical import semiclassical_qfi_trajectories, qfi_from_powersums
from .meanfield import exact_mean_a, sc_trajectories_alpha
from .compare import run_comparison

__all__ = [
    "thermal_state",
    "coherent_state",
    "sample_wigner_alpha",
    "qfi_keldysh",
    "qfi_long_time",
    "steady_state_occupation",
    "deformed_liouvillian",
    "global_qfi_deformed",
    "fock_truncation",
    "semiclassical_qfi_trajectories",
    "qfi_from_powersums",
    "exact_mean_a",
    "sc_trajectories_alpha",
    "run_comparison",
]
