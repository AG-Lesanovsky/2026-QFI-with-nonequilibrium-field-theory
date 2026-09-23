"""Semiclassical global QFI from Keldysh stochastic (saddle-point) trajectories.

The classical field α_c obeys the additive-noise Itô SDE

    dα_c = (−iε − ½Δ) α_c dt + ½(dW↑ − dW↓),   Δ = γ↓ − γ↑,

with ⟨dW_i* dW_j⟩ = 2γ_i δ_{ij} dt.  The γ↑-derivative of the classical action is
the Itô integral  X = ∂_{γ↑}𝒮 · (−2γ↑) = Σ_n Im[α_c(t_n)·dW↑_n], and the global
QFI is  I(γ↑,t) = 4·Var[∂_{γ↑}𝒮] = Var(X)/γ↑².

Because the noise is additive, Itô ≡ Stratonovich, so a plain Euler–Maruyama
step suffices.  The QFI is a variance estimator; its finite-sample uncertainty
(the shaded band in the comparison plot) is obtained from additive power sums
s₁..s₄ of X across trajectories via the 4th-moment formula.
"""

from __future__ import annotations

import numpy as np

from .states import sample_wigner_alpha


def qfi_from_powersums(count: int, s1, s2, s3, s4, gamma_up: float):
    """Reduce additive power sums of X to ``(qfi, qfi_stderr)`` with prefactor 1/γ↑².

    ``s_p = Σ X^p`` over ``count`` trajectories.  QFI = Var(X)/γ↑² (unbiased
    sample variance); the standard error of that variance estimate uses the 2nd
    and 4th central moments.  Power sums are additive, so partial batches may be
    summed before calling.
    """
    n = count
    pref = 1.0 / gamma_up ** 2
    mean = s1 / n
    m2 = s2 / n - mean * mean
    var = n / (n - 1) * m2                                  # unbiased Var(X)
    m4 = s4 / n - 4.0 * mean * (s3 / n) + 6.0 * mean * mean * (s2 / n) - 3.0 * mean ** 4
    var_of_var = (m4 - (n - 3) / (n - 1) * m2 * m2) / n     # Var of the variance estimator
    qfi = pref * var
    qfi_stderr = pref * np.sqrt(np.maximum(var_of_var, 0.0))
    return qfi, qfi_stderr


def semiclassical_qfi_trajectories(
    N_traj: int = 10000,
    epsilon: float = 1.0,
    gamma_up: float = 0.1,
    gamma_down: float = 1.0,
    tmax: float = 1000.0,
    nt: int = 3000,
    n_0: float = 0.0,
    seed: int = 42,
    progress: bool = False,
) -> dict:
    """Time-resolved global QFI I(γ↑, t) from ``N_traj`` semiclassical trajectories.

    Euler–Maruyama integration of the additive-noise SDE with Wigner-sampled
    initial fields α_c(0) ~ CN(0, n_0 + ½).  The Itô accumulator
    X = Σ_n Im[α_c(t_n)·dW↑_n] uses the left-point value α_c(t_n) (independent of
    dW↑_n), so the Itô isometry applies.  ``qfi[k]`` and ``qfi_stderr[k]`` are the
    QFI and its finite-sample error at ``times[k]`` (both 0 at t=0).

    Returns ``{times, qfi, qfi_stderr, n_traj}``.
    """
    if N_traj < 2:
        raise ValueError("Need at least 2 trajectories to estimate a variance.")

    rng = np.random.default_rng(seed)
    dt = tmax / (nt - 1)
    times = np.linspace(0.0, tmax, nt)
    Delta = gamma_down - gamma_up

    # ∂_t α_c = (−iε − Δ/2) α_c;  Re(drift) = −Δ/2 < 0 ⟹ relaxation for γ↓ > γ↑.
    # Exponential-Euler drift: the exact one-step propagator |phi| = e^{−Δdt/2} < 1
    # is unconditionally stable (plain Euler 1+drift·dt can exceed 1 for ε·dt ≳ Δ),
    # and matches plain Euler–Maruyama to O(dt²) for the additive noise.
    drift = -1j * epsilon - 0.5 * Delta
    phi = np.exp(drift * dt)
    std_up = np.sqrt(gamma_up * dt)     # std of Re[dW↑], Im[dW↑]
    std_down = np.sqrt(gamma_down * dt)

    alpha = sample_wigner_alpha(N_traj, n_0, rng)

    running_X = np.zeros(N_traj)        # X = Σ_n Im[α_c(t_n)·dW↑_n]
    s1 = np.zeros(nt); s2 = np.zeros(nt)
    s3 = np.zeros(nt); s4 = np.zeros(nt)   # index 0 stays 0 (X = 0 at t = 0)

    step_iter = range(1, nt)
    if progress:
        from tqdm.auto import tqdm
        step_iter = tqdm(step_iter, desc="SC trajectories", leave=False)

    for n in step_iter:
        # Left-point snapshot α_c(t_{n-1}), independent of this step's noise.
        a_re = alpha.real
        a_im = alpha.imag

        dW_up_re = rng.standard_normal(N_traj) * std_up
        dW_up_im = rng.standard_normal(N_traj) * std_up
        dW_dn_re = rng.standard_normal(N_traj) * std_down
        dW_dn_im = rng.standard_normal(N_traj) * std_down

        # Im[(a_re + i a_im)(dW_re + i dW_im)] = a_re·dW_im + a_im·dW_re.
        running_X += a_re * dW_up_im + a_im * dW_up_re

        X = running_X
        X2 = X * X
        s1[n] = X.sum(); s2[n] = X2.sum()
        s3[n] = (X2 * X).sum(); s4[n] = (X2 * X2).sum()

        dW_up = dW_up_re + 1j * dW_up_im
        dW_dn = dW_dn_re + 1j * dW_dn_im
        alpha = phi * alpha + 0.5 * (dW_up - dW_dn)

    qfi = np.zeros(nt)
    qfi_stderr = np.zeros(nt)
    qfi[1:], qfi_stderr[1:] = qfi_from_powersums(
        N_traj, s1[1:], s2[1:], s3[1:], s4[1:], gamma_up
    )
    return {"times": times, "qfi": qfi, "qfi_stderr": qfi_stderr, "n_traj": N_traj}
