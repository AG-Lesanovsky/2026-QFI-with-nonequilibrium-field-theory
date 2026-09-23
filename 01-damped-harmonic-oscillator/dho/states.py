"""Initial states for the single-mode damped/pumped oscillator.

A single bosonic mode with pump L₁ = √γ↑ a† and decay L₂ = √γ↓ a relaxes to a
thermal state with mean occupation n_ss = γ↑/(γ↓ − γ↑).  These helpers build the
Fock-space density matrices used by the exact (deformed-Liouvillian) engine and
the Wigner phase-space samples used to seed the semiclassical trajectories.
"""

from __future__ import annotations

import numpy as np
from scipy.special import factorial


def thermal_state(N: int, n_0: float) -> np.ndarray:
    """Thermal (mixed) state ρ with mean occupation ``n_0``, truncated to ``N`` Fock levels.

    Bose–Einstein weights p_n = (n_0/(n_0+1))ⁿ / (n_0+1); ``n_0 = 0`` returns the
    vacuum |0⟩⟨0|.  Renormalised after truncation.
    """
    if n_0 == 0:
        rho = np.zeros((N, N), dtype=complex)
        rho[0, 0] = 1.0
        return rho
    probs = np.array([(1.0 / (n_0 + 1)) * (n_0 / (n_0 + 1)) ** n for n in range(N)])
    probs /= probs.sum()  # renormalise due to Fock-space truncation
    return np.diag(probs).astype(complex)


def coherent_state(N: int, n_0: float) -> np.ndarray:
    """Coherent state |α⟩⟨α| with mean occupation n_0 = |α|², truncated to ``N`` levels.

    Fock amplitudes ψ_n = e^{−n_0/2} n_0^{n/2}/√(n!); ``n_0 = 0`` returns the
    vacuum |0⟩⟨0|.  Renormalised after truncation.
    """
    if n_0 == 0:
        rho = np.zeros((N, N), dtype=complex)
        rho[0, 0] = 1.0
        return rho
    ns = np.arange(N)
    psi = np.exp(-n_0 / 2) * (n_0 ** (ns / 2)) / np.sqrt(factorial(ns, exact=False))
    psi /= np.linalg.norm(psi)  # renormalise due to Fock-space truncation
    return np.outer(psi, psi.conj()).astype(complex)


def sample_wigner_alpha(n_traj: int, n_0: float, rng: np.random.Generator) -> np.ndarray:
    """Draw ``n_traj`` initial fields α_c(0) from the Wigner distribution of the state.

    A state with mean occupation ``n_0`` has a Gaussian Wigner function with
    ⟨|α|²⟩_W = n_0 + ½, so each quadrature ~ N(0, (n_0+½)/2).  ``n_0 = 0`` gives
    the vacuum zero-point fluctuations.  Matches the thermal/coherent variance
    used by the Fock-space initial states above.
    """
    sigma_per_quad = np.sqrt((n_0 + 0.5) / 2)
    return sigma_per_quad * (
        rng.standard_normal(n_traj) + 1j * rng.standard_normal(n_traj)
    )
