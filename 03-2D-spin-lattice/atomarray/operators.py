"""Sparse many-body operators for the exact/MCWF reference of the driven array.

Builds the N-atom (2ᴺ-dimensional) Hamiltonian, collapse operators and drive
generator of the master equation (Mink & Fleischhauer Eqs. 44/45) directly from
per-atom Pauli operators, so the statevector MCWF in ``qfi.py`` can integrate it.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

# Single-spin Pauli / ladder operators (basis order |0⟩=↑, |1⟩=↓ → σ_z = diag(+1,−1)).
_SX = sp.csr_matrix(np.array([[0, 1], [1, 0]], dtype=complex))
_SY = sp.csr_matrix(np.array([[0, -1j], [1j, 0]], dtype=complex))
_SZ = sp.csr_matrix(np.array([[1, 0], [0, -1]], dtype=complex))
_SM = sp.csr_matrix(np.array([[0, 0], [1, 0]], dtype=complex))   # σ⁻ = |↓⟩⟨↑|


def _embed(op: sp.spmatrix, n: int, N: int) -> sp.csr_matrix:
    """Embed a single-site operator on atom ``n`` into the N-atom Hilbert space."""
    left = sp.identity(2 ** n, dtype=complex, format="csr")
    right = sp.identity(2 ** (N - n - 1), dtype=complex, format="csr")
    return sp.kron(sp.kron(left, op), right, format="csr")


def local_ops(N: int) -> dict:
    """Per-atom operators as lists over atoms: ``sx, sy, sz, sm`` (each length N)."""
    return {
        "sx": [_embed(_SX, n, N) for n in range(N)],
        "sy": [_embed(_SY, n, N) for n in range(N)],
        "sz": [_embed(_SZ, n, N) for n in range(N)],
        "sm": [_embed(_SM, n, N) for n in range(N)],
    }


def build_operators(
    Gamma: np.ndarray,
    J: np.ndarray,
    Omega: float,
    Delta: float = 0.0,
    ops: dict | None = None,
) -> dict:
    """Assemble (H, dH, c_ops, psi0) for the array master equation.

    H = −Ω Σ_n σ^x_n − (Δ/2) Σ_n σ^z_n + Σ_{m≠n} J_mn σ⁺_m σ⁻_n,
    collapse operators L_c = Σ_n G[n,c] σ⁻_n from Γ = G Gᵀ, and ∂_Ω H = −Σ_n σ^x_n.
    ``psi0`` is the collective ground state |↓…↓⟩.
    """
    from .geometry import factorize_decay_matrix

    N = Gamma.shape[0]
    if ops is None:
        ops = local_ops(N)
    sx, sz, sm = ops["sx"], ops["sz"], ops["sm"]
    dim = 2 ** N

    Mx = sum(sx)                                   # Σ_n σ^x_n
    dH = -Mx
    H = -Omega * Mx
    if Delta != 0.0:
        H = H - 0.5 * Delta * sum(sz)
    # Coherent dipole–dipole Σ_{m≠n} J_mn σ⁺_m σ⁻_n (J_nn = 0).
    sp_ops = [c.conj().T.tocsr() for c in sm]
    for m in range(N):
        for n in range(N):
            if m != n and J[m, n] != 0.0:
                H = H + J[m, n] * (sp_ops[m] @ sm[n])

    G = factorize_decay_matrix(Gamma)
    c_ops = [
        sum(G[n, c] * sm[n] for n in range(N)).tocsr()
        for c in range(G.shape[1])
    ]

    psi0 = np.zeros(dim, dtype=complex)
    psi0[-1] = 1.0                                 # |↓…↓⟩ (all atoms in ground state)

    return {"H": H.tocsr(), "dH": dH.tocsr(), "c_ops": c_ops, "psi0": psi0, "N": N}
