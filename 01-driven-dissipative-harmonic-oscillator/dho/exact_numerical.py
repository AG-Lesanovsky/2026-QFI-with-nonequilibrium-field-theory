"""Exact numerical global QFI via the deformed (two-copy) master equation.

The global QFI for estimating γ↑ is read off the generalized fidelity amplitude
F_δ(t) = Tr[ϱ_δ(t)], where ϱ_δ is the cross-overlap operator evolved by the
deformed Liouvillian coupling the dynamics at γ↑ ± δ/2:

    ϱ̇_δ = −i(H₊ϱ_δ − ϱ_δ H₋)
           + Σⱼ (L_{j,+} ϱ_δ L_{j,−}† − ½ L_{j,+}†L_{j,+} ϱ_δ − ½ ϱ_δ L_{j,−}†L_{j,−}).

For small δ,  I(γ↑, t) ≈ 8[1 − Re F_δ(t)]/δ².  The Liouvillian is built as a
sparse superoperator (column-stacking vec, order='F') and propagated with
``scipy.sparse.linalg.expm_multiply`` so moderate Fock truncations stay cheap.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply

from .states import thermal_state, coherent_state
from .analytical import steady_state_occupation


def _annihilation(N: int) -> sp.csr_matrix:
    """Sparse annihilation operator a on ``N`` Fock levels (a|n⟩ = √n |n−1⟩)."""
    return sp.diags(np.sqrt(np.arange(1, N)), offsets=1, format="csr", dtype=complex)


def fock_truncation(n_0: float, n_ss: float, tol: float = 1e-4, min_dim: int = 20,
                    max_dim: int = 400) -> int:
    """Suggest a Fock-space dimension covering the initial and steady-state occupations.

    A thermal state with mean m has geometric tail P(n≥N) = (m/(m+1))ᴺ, so keeping
    it below ``tol`` needs N ≥ ln(1/tol)/ln((m+1)/m) with m = max(n_0, n_ss).  This
    is the honest size: a crude N ≈ 6·m truncates ~1% of the QFI once the state
    *sits* at high occupation (large n_ss, e.g. γ↑/γ↓ → 1), where the pump a†
    keeps populating the top levels.  Clamped to [``min_dim``, ``max_dim``]; the
    deformed Liouvillian is N²-dimensional, so ``max_dim`` guards infeasible builds.
    """
    m = max(n_0, n_ss, 1e-9)
    N = int(np.ceil(np.log(1.0 / tol) / np.log1p(1.0 / m)))
    return int(np.clip(N, min_dim, max_dim))


def deformed_liouvillian(N: int, epsilon: float, gamma_up: float, gamma_down: float,
                         delta: float) -> sp.csr_matrix:
    """Sparse deformed Liouvillian A(δ) acting on vec(X) (column-stacking, order='F').

    X is the cross-overlap operator between the dynamics at γ↑ ± δ/2.  Requires
    γ↑ ± δ/2 ≥ 0.
    """
    g_plus = gamma_up + 0.5 * delta
    g_minus = gamma_up - 0.5 * delta
    if g_plus < 0 or g_minus < 0:
        raise ValueError("Choose delta so that gamma_up +/- delta/2 remain non-negative.")

    a = _annihilation(N)
    adag = a.conj().T.tocsr()
    Id = sp.identity(N, dtype=complex, format="csr")

    H = epsilon * (adag @ a)          # H₊ = H₋ = H
    L1_plus = np.sqrt(g_plus) * adag
    L1_minus = np.sqrt(g_minus) * adag
    L2_plus = np.sqrt(gamma_down) * a
    L2_minus = np.sqrt(gamma_down) * a

    def one_channel(Lp, Lm):
        C_plus = (Lp.conj().T @ Lp)
        C_minus = (Lm.conj().T @ Lm)
        jump = sp.kron(Lm.conj(), Lp, format="csr")
        loss_left = 0.5 * sp.kron(Id, C_plus, format="csr")
        loss_right = 0.5 * sp.kron(C_minus.T, Id, format="csr")
        return jump - loss_left - loss_right

    coherent = -1j * (sp.kron(Id, H, format="csr") - sp.kron(H.T, Id, format="csr"))
    A = coherent + one_channel(L1_plus, L1_minus) + one_channel(L2_plus, L2_minus)
    return A.tocsr()


def global_qfi_deformed(N: int = 25, epsilon: float = 1.0, gamma_up: float = 0.4,
                        gamma_down: float = 1.0, tmax: float = 10.0, nt: int = 200,
                        delta: float = 1e-5, n_0: float = 0.0, state_type: str = "thermal",
                        parameter_dependent_initial_state: bool = False) -> dict:
    """Global QFI I(γ↑, t) by finite-difference deformed evolution.

    Parameters mirror the physics knobs; ``N`` is the Fock truncation (use
    :func:`fock_truncation` to size it), ``delta`` the finite-difference step.
    ``parameter_dependent_initial_state=True`` seeds with the γ↑ ± δ/2 steady
    states (thermal only).  Returns ``{times, qfi, fidelity}``.
    """
    times = np.linspace(0.0, tmax, nt)
    A = deformed_liouvillian(N, epsilon, gamma_up, gamma_down, delta)

    if parameter_dependent_initial_state and state_type == "thermal":
        # Initial states depend on the target parameter.
        n_ss_plus = steady_state_occupation(gamma_up + 0.5 * delta, gamma_down)
        n_ss_minus = steady_state_occupation(gamma_up - 0.5 * delta, gamma_down)
        rho0_plus = thermal_state(N, n_ss_plus)
        rho0_minus = thermal_state(N, n_ss_minus)
        # Diagonal ρ: the initial generalized overlap is √ρ₊ · √ρ₋.
        rho0 = np.sqrt(rho0_plus) @ np.sqrt(rho0_minus)
    elif state_type == "thermal":
        rho0 = thermal_state(N, n_0)
    elif state_type == "coherent":
        rho0 = coherent_state(N, n_0)
    else:
        raise ValueError(f"Unknown state_type '{state_type}'. Choose 'thermal' or 'coherent'.")

    vec_rho0 = rho0.reshape(-1, order="F")
    vec_t = expm_multiply(A, vec_rho0, start=times[0], stop=times[-1], num=nt)

    # F_δ(t) = Tr[X(t)];  the trace picks the vec entries at positions n(N+1).
    trace_idx = np.arange(N) * (N + 1)
    fidelity = vec_t[:, trace_idx].sum(axis=1).real

    qfi = np.maximum(8.0 * (1.0 - fidelity) / (delta ** 2), 0.0)
    return {"times": times, "qfi": qfi, "fidelity": fidelity}
