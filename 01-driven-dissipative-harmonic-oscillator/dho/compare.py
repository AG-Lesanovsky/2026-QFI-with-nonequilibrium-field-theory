"""One-parameter-set driver: semiclassical vs exact-numerical vs exact-analytical QFI.

Bundles the three estimators of :mod:`dho` for a single physical parameter set so
the notebook stays a thin wrapper.  The exact-numerical (deformed-Liouvillian)
curve needs a Fock truncation N ≫ max(n_0, n_ss); when the auto-sized N would
exceed ``max_fock`` it is skipped (returned as ``None``) rather than attempting an
infeasible dense build.
"""

from __future__ import annotations

import numpy as np

from .analytical import qfi_keldysh, steady_state_occupation
from .exact_numerical import global_qfi_deformed, fock_truncation
from .semiclassical import semiclassical_qfi_trajectories


def run_comparison(
    *,
    label: str | None = None,
    gamma_up: float,
    gamma_down: float = 1.0,
    epsilon: float = 1.0,
    n_0: float = 1.0,
    tmax: float | None = None,
    dt_sc: float = 0.05,
    N_traj: int = 8000,
    seed: int = 42,
    N_fock: int | None = None,
    max_fock: int = 120,
    delta: float = 1e-5,
    nt_exact: int = 400,
    nt_analytic: int = 600,
    progress: bool = False,
) -> dict:
    """Compute the three QFI curves for one parameter set.

    ``tmax`` defaults to 20/Δ (≈ 20 relaxation times, enough to reach the
    linear-in-t plateau).  The semiclassical step count follows from ``dt_sc``;
    the exact-numerical Fock dimension is auto-sized unless ``N_fock`` is given.
    Returns a panel dict consumed by the plotting cells of the notebook.
    """
    Delta = gamma_down - gamma_up
    if Delta <= 0:
        raise ValueError("Require gamma_down > gamma_up for a stable (relaxing) mode.")
    if tmax is None:
        tmax = 100.0 / Delta
    if label is None:
        label = rf"$\gamma_\uparrow/\gamma_\downarrow={gamma_up / gamma_down:.2g}$, $n_0={n_0:g}$"

    n_ss = steady_state_occupation(gamma_up, gamma_down)

    # ── Semiclassical (with finite-sample error band) ─────────────────────────
    nt_sc = int(round(tmax / dt_sc)) + 1
    sc = semiclassical_qfi_trajectories(
        N_traj=N_traj, epsilon=epsilon, gamma_up=gamma_up, gamma_down=gamma_down,
        tmax=tmax, nt=nt_sc, n_0=n_0, seed=seed, progress=progress,
    )

    # ── Exact numerical (deformed Liouvillian), skipped if truncation too large ─
    # Size honestly first (fock_truncation clamps only to its own max_dim), then
    # skip the N²-dimensional build when the required N exceeds max_fock.
    N = N_fock if N_fock is not None else fock_truncation(n_0, n_ss)
    if N <= max_fock:
        exact = global_qfi_deformed(
            N=N, epsilon=epsilon, gamma_up=gamma_up, gamma_down=gamma_down,
            tmax=tmax, nt=nt_exact, delta=delta, n_0=n_0, state_type="thermal",
        )
        exact["N"] = N
    else:
        print(f"[{label}] exact-numerical skipped: needs Fock N={N} > max_fock={max_fock} "
              f"(n_ss={n_ss:.1f}); raise max_fock to include it.")
        exact = None

    # ── Exact analytical (Keldysh), log-spaced from the SC step to span the axis ─
    t_an = np.geomspace(dt_sc, tmax, nt_analytic)
    analytic = {"times": t_an, "qfi": qfi_keldysh(gamma_up, gamma_down, n_0, t_an)}

    return {
        "label": label,
        "gamma_up": gamma_up, "gamma_down": gamma_down,
        "epsilon": epsilon, "n_0": n_0, "n_ss": n_ss, "tmax": tmax,
        "sc": sc, "exact": exact, "analytic": analytic,
    }
