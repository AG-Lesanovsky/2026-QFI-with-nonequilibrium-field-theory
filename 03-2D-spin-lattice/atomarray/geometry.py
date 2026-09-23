"""Atom positions and the free-space radiative rate matrices Gamma_mn, J_mn.

Coordinates are in units of the transition wavelength lambda_e, so the phase argument
is x = k_e r_mn = 2 pi |r_m - r_n| / lambda_e.  The dissipative matrix Gamma and the
coherent dipole-dipole matrix J follow the free-space Green's tensor
(arXiv:2305.19829); Gamma_nn = Gamma0, J_nn = 0.
"""

from __future__ import annotations

import numpy as np

TWO_PI = 2.0 * np.pi


def square_lattice(L: int, a: float) -> np.ndarray:
    """LxL square lattice in the x-y plane, spacing a*lambda_e.

    Returns positions of shape (L*L, 3) (z = 0), row-major in (ix, iy).
    """
    idx = np.arange(int(L))
    ix, iy = np.meshgrid(idx, idx, indexing="ij")
    r = np.stack([ix.ravel() * a, iy.ravel() * a, np.zeros(L * L)], axis=1)
    return r.astype(float)


def rate_matrices(
    positions: np.ndarray,
    polarization,
    Gamma0: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Radiative rate matrices (Gamma, J) for atoms at ``positions``.

    Parameters
    ----------
    positions    : (N, 3) atom positions in lambda_e units
    polarization : length-3 (possibly complex) dipole unit vector e_p
    Gamma0       : single-atom decay rate (diagonal of Gamma)

    Returns
    -------
    (Gamma, J) : (N, N) real symmetric matrices; Gamma_nn = Gamma0, J_nn = 0.
    """
    r = np.asarray(positions, dtype=float)
    N = r.shape[0]
    ep = np.asarray(polarization, dtype=complex)
    ep = ep / np.linalg.norm(ep)

    rij = r[:, None, :] - r[None, :, :]          # (N, N, 3)
    dist = np.linalg.norm(rij, axis=2)           # (N, N)
    iu, ju = np.triu_indices(N, k=1)             # unique unordered pairs

    d = dist[iu, ju]
    er = rij[iu, ju] / d[:, None]                # unit separation vectors
    x = TWO_PI * d                               # k_e r_mn
    pdot = np.abs(er @ ep) ** 2                  # |e_p · e_r|^2

    sinx, cosx = np.sin(x), np.cos(x)
    a1 = 1.0 - pdot
    a2 = 1.0 - 3.0 * pdot

    J_pair = -0.75 * Gamma0 * (
        a1 * cosx / x - a2 * (sinx / x**2 + cosx / x**3)
    )
    G_pair = 1.5 * Gamma0 * (
        a1 * sinx / x + a2 * (cosx / x**2 - sinx / x**3)
    )

    Gamma = np.zeros((N, N))
    J = np.zeros((N, N))
    Gamma[iu, ju] = Gamma[ju, iu] = G_pair
    J[iu, ju] = J[ju, iu] = J_pair
    np.fill_diagonal(Gamma, float(Gamma0))      
    return Gamma, J


def factorize_decay_matrix(Gamma: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Factor the PSD decay matrix Gamma = G G^T, keeping positive eigenmodes only.

    Returns G of shape (N, R) whose columns are the collective noise /
    jump channels (R = number of eigenvalues above tol·max).
    """
    Gamma = np.asarray(Gamma, dtype=float)
    evals, evecs = np.linalg.eigh(Gamma)
    evals = np.maximum(evals, 0.0)
    if evals.max() <= 0.0:
        return np.zeros((Gamma.shape[0], 0))
    keep = evals > tol * evals.max()
    return evecs[:, keep] * np.sqrt(evals[keep])[None, :]
