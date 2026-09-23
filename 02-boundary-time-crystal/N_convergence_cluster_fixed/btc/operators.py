"""Collective spin operators, Hamiltonian, collapse operators, and initial state.

All operators are built for total spin S = N/2 using QuTiP's jmat convention,
where basis index 0 corresponds to m = +S (fully polarised up).
"""

import numpy as np
import qutip as qt

from .sde import bloch_direction


def make_operators(S: float) -> dict:
    """Return the collective spin operators for total spin S.

    Returns
    -------
    dict with keys 'Sx', 'Sy', 'Sz', 'Sp', 'Sm'
    """
    return {
        "Sx": qt.jmat(S, "x"),
        "Sy": qt.jmat(S, "y"),
        "Sz": qt.jmat(S, "z"),
        "Sp": qt.jmat(S, "+"),
        "Sm": qt.jmat(S, "-"),
    }


def make_hamiltonian(ops: dict, omega: float) -> qt.Qobj:
    """H = omega * Sx.

    Parameters
    ----------
    ops :   output of make_operators
    omega : coefficient in front of Sx  (can encode any rescaling)
    """
    return omega * ops["Sx"]


def make_collapse_ops(ops: dict, kappa: float) -> list:
    """Lindblad collapse operators for collective decay.

    L = sqrt(kappa) * S_-
    """
    return [np.sqrt(kappa) * ops["Sm"]]


def make_observables(ops: dict, S: float) -> list:
    """Normalised spin observables [Sx/S, Sy/S, Sz/S] for mesolve e_ops."""
    return [ops["Sx"] / S, ops["Sy"] / S, ops["Sz"] / S]


def make_initial_state(S: float, state="up") -> qt.Qobj:
    """Density matrix for a spin-coherent state |S, n̂⟩ along Bloch direction n̂.

    The quantum counterpart of the DTWA initial sampling: the same ``state`` spec
    selects the coherent state pointing along the Bloch unit vector
    n̂ = (⟨σ_x⟩, ⟨σ_y⟩, ⟨σ_z⟩), built by rotating the fully polarised state
    |S, m=+S⟩ to direction (θ, φ),

        |S, n̂⟩ = e^{-i φ J_z} e^{-i θ J_y} |S, m=+S⟩,

    so that ⟨S⟩/S = n̂.  This matches :func:`~btc.sde.sample_initial_conditions_dtwa`
    (whose ensemble mean is the same n̂), enabling like-for-like quantum vs.
    semiclassical comparisons.

    Parameters
    ----------
    S     : total spin quantum number (S = N/2)
    state : coherent-state Bloch direction — a name (``"up"``, ``"down"``,
            ``"+x"``, ``"-x"``, ``"+y"``, ``"-y"``) or a length-3 array-like
            Bloch vector n̂.  Default ``"up"`` is |S, m=+S⟩.

    Returns
    -------
    rho0 : (dim, dim) density matrix of the coherent state
    """
    dim = int(2 * S + 1)
    psi_up = qt.basis(dim, 0)   # index 0 ↔ m = +S in QuTiP's jmat convention

    n_hat = bloch_direction(state)
    if np.isclose(n_hat[2], 1.0):
        return qt.ket2dm(psi_up)   # already |S, m=+S⟩ (avoids 0/0 in φ)

    theta = np.arccos(np.clip(n_hat[2], -1.0, 1.0))
    phi = np.arctan2(n_hat[1], n_hat[0])
    Jy = qt.jmat(S, "y")
    Jz = qt.jmat(S, "z")
    # Rotate the up state to direction (θ, φ): ⟨S⟩/S = n̂.
    rot = (-1j * phi * Jz).expm() * (-1j * theta * Jy).expm()
    return qt.ket2dm(rot * psi_up)
