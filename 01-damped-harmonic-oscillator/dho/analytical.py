"""Exact analytical global QFI from the Keldysh (real-time) approach.

For the linear (Gaussian) model H = ε a†a, L₁ = √γ↑ a†, L₂ = √γ↓ a the global QFI
for estimating the pump rate γ↑ is known in closed form.  These formulas are the
ground truth the numerical (deformed-Liouvillian) and semiclassical estimators
are benchmarked against.
"""

from __future__ import annotations

import numpy as np


def steady_state_occupation(gamma_up: float, gamma_down: float) -> float:
    """Steady-state mean occupation n_ss = γ↑/(γ↓ − γ↑) (stable for γ↓ > γ↑)."""
    return gamma_up / (gamma_down - gamma_up)


def qfi_long_time(gamma_up: float, gamma_down: float, t):
    """Long-time (linear-in-t) limit of the global QFI, I/t → γ↓/(γ↑ Δ), Δ = γ↓ − γ↑."""
    return (gamma_down / gamma_up) * (t / (gamma_down - gamma_up))


def qfi_keldysh(gamma_up: float, gamma_down: float, n_0: float, t):
    """Global QFI I(γ↑, t) at all times for initial mean occupation ``n_0``.

    I(γ↑,t) = γ↓/(γ↑Δ)·t + (n_ss − n_0)/(Δ γ↑)·(e^{−Δt} − 1),  Δ = γ↓ − γ↑,
    with the transient set by the relaxation of ⟨n⟩ from n_0 towards n_ss.
    """
    if gamma_up == gamma_down:
        raise ValueError("gamma_up and gamma_down must be different for the Keldysh formula.")
    n_ss = steady_state_occupation(gamma_up, gamma_down)
    Delta = gamma_down - gamma_up
    return (
        qfi_long_time(gamma_up, gamma_down, t)
        + (n_ss - n_0) / (Delta * gamma_up) * (np.exp(-Delta * t) - 1)
    )
