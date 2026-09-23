"""Mean-field diagnostic: exact ⟨a(t)⟩ vs semiclassical ⟨α_c(t)⟩.

Because the equation of motion for ⟨a⟩ is linear,

    d⟨a⟩/dt = (−iε − ½Δ)⟨a⟩,   Δ = γ↓ − γ↑,

the exact quantum result is analytic, ⟨a(t)⟩ = α₀ e^{(−iε − Δ/2)t}, and the
semiclassical SDE has the same deterministic drift — so the trajectory mean
converges to it.  This sanity-checks the drift/noise of the SDE stepper used by
:mod:`dho.semiclassical` (individual trajectories scatter around the mean).
"""

from __future__ import annotations

import numpy as np


def exact_mean_a(alpha_0: complex, epsilon: float, gamma_up: float, gamma_down: float,
                 times: np.ndarray) -> np.ndarray:
    """Exact ⟨a(t)⟩ = α₀·exp((−iε − Δ/2)t), Δ = γ↓ − γ↑ (any state with ⟨a(0)⟩ = α₀)."""
    Delta = gamma_down - gamma_up
    return alpha_0 * np.exp((-1j * epsilon - 0.5 * Delta) * times)


def sc_trajectories_alpha(
    N_traj: int = 300,
    epsilon: float = 1.0,
    gamma_up: float = 0.1,
    gamma_down: float = 1.0,
    tmax: float = 15.0,
    nt: int = 600,
    alpha_0: complex = 1.0,
    seed: int = 0,
) -> tuple:
    """Euler–Maruyama trajectories of the classical field α_c from a coherent state |α₀⟩.

    SDE (Itô): dα_c = (−iε − ½Δ)α_c dt + ½(dW↑ − dW↓), dW_i ~ CN(0, 2γ_i dt).
    Returns ``(times, trajs)`` with ``trajs`` of shape ``(N_traj, nt)``.
    """
    rng = np.random.default_rng(seed)
    dt = tmax / (nt - 1)
    times = np.linspace(0.0, tmax, nt)
    Delta = gamma_down - gamma_up

    drift = -1j * epsilon - 0.5 * Delta
    phi = np.exp(drift * dt)     # exact one-step drift propagator (unconditionally stable)
    std_up = np.sqrt(gamma_up * dt)
    std_down = np.sqrt(gamma_down * dt)

    alpha = np.full(N_traj, alpha_0, dtype=complex)
    trajs = np.zeros((N_traj, nt), dtype=complex)
    trajs[:, 0] = alpha

    for n in range(1, nt):
        dW_up = (rng.standard_normal(N_traj) + 1j * rng.standard_normal(N_traj)) * std_up
        dW_dn = (rng.standard_normal(N_traj) + 1j * rng.standard_normal(N_traj)) * std_down
        alpha = phi * alpha + 0.5 * (dW_up - dW_dn)
        trajs[:, n] = alpha

    return times, trajs
