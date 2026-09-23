"""Quantum Fisher information for sensing the Rabi frequency Ω of a driven array.

Semiclassical (truncated-Wigner)
--------------------------------
Per trajectory the action derivative is ∂_Ω 𝒮 = ∫₀ᵀ M_x dt with M_x = Σ_n s^x_n
(the total x-magnetisation symbol), and F_sc(T) = 4·Var_traj(∂_Ω 𝒮).

Quantum monitored (reference)
-----------------------------
Gammelmark–Mølmer QFI of the continuously monitored master equation, estimated by
statevector MCWF (quantum-jump) trajectories — the only route that scales to a
4×4 array (2ᴺ statevector; the 4ᴺ density matrix is infeasible).  With the drive
generator ∂_Ω H = −Σ_n σ^x_n the two are directly comparable: F_sc(T) ≈ F_mon(T).

The two are in fact the same functional: propagating the Gammelmark–Mølmer
hierarchy and tracing gives

    F_mon(T) = 4[ 2∫₀ᵀdt ∫₀ᵗds Re⟨M̂_x(t)M̂_x(s)⟩ − (∫₀ᵀ⟨M̂_x⟩dt)² ]
             = 4·Var_sym[ ∫₀ᵀ M̂_x dt ],

i.e. the variance of the time-integrated x-magnetisation with *symmetrised*
quantum two-time correlations, so F_sc is the same expression with the TWA
correlator in place of the quantum one — the only source of their deviation.
"""

from __future__ import annotations

import os
import queue as _queue
from concurrent.futures import ProcessPoolExecutor, as_completed, wait

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply

from .sde import sample_cone_cartesian, array_step
from .geometry import rate_matrices, factorize_decay_matrix


# ── Semiclassical QFI ─────────────────────────────────────────────────────────

def _sc_chunk(N, Omega, Delta, Gamma, J, G, dt, n_time, store_idx,
              n_traj, seed, n_hat):
    """Integrate ``n_traj`` array realisations, streaming power sums of X = ∂_Ω 𝒮.

    Returns ``(n_traj, s1, s2, s3, s4)`` with s_p = Σ_k X_k^p at the stored times.
    """
    rng = np.random.default_rng(seed)
    s = sample_cone_cartesian(N * n_traj, rng, n_hat).reshape(N, n_traj, 3)

    n_out = n_time if store_idx is None else len(store_idx)
    s1 = np.zeros(n_out); s2 = np.zeros(n_out)
    s3 = np.zeros(n_out); s4 = np.zeros(n_out)

    Mx_prev = s[..., 0].sum(axis=0)                # Σ_n s^x_n per realisation
    dS = np.zeros(n_traj)                          # running ∫ M_x dt (trapezoid)
    half_dt = 0.5 * dt

    if store_idx is None:
        for n in range(1, n_time):
            s = array_step(s, dt, Omega, Delta, Gamma, J, G, rng)
            Mx = s[..., 0].sum(axis=0)
            dS += half_dt * (Mx_prev + Mx)
            Mx_prev = Mx
            d2 = dS * dS
            s1[n] = dS.sum(); s2[n] = d2.sum()
            s3[n] = (d2 * dS).sum(); s4[n] = (d2 * d2).sum()
    else:
        k = 1
        nxt = store_idx[k] if n_out > 1 else -1
        for n in range(1, n_time):
            s = array_step(s, dt, Omega, Delta, Gamma, J, G, rng)
            Mx = s[..., 0].sum(axis=0)
            dS += half_dt * (Mx_prev + Mx)
            Mx_prev = Mx
            if n == nxt:
                d2 = dS * dS
                s1[k] = dS.sum(); s2[k] = d2.sum()
                s3[k] = (d2 * dS).sum(); s4[k] = (d2 * d2).sum()
                k += 1
                if k < n_out:
                    nxt = store_idx[k]
    return n_traj, s1, s2, s3, s4


def qfi_from_powersums(count, s1, s2, s3, s4):
    """Reduce additive power sums of X = ∂_Ω 𝒮 to (qfi, qfi_stderr).

    Additive across independent batches/tasks, so partial sums may be summed
    before calling.  QFI = 4·Var(X); its error bar uses the 4th moment.
    """
    n = count
    mean = s1 / n
    qfi = 4.0 * (s2 - n * mean * mean) / (n - 1)
    m2 = s2 / n - mean * mean
    m4 = s4 / n - 4.0 * mean * (s3 / n) + 6.0 * mean * mean * (s2 / n) - 3.0 * mean**4
    var_S2 = (m4 - (n - 3) / (n - 1) * m2 * m2) / n
    return qfi, 4.0 * np.sqrt(np.maximum(var_S2, 0.0))


def _store_grid(times, n_store):
    n_time = len(times)
    if n_store is not None and n_store < n_time:
        if n_store < 2:
            raise ValueError("`n_store` must be at least 2.")
        idx = np.unique(np.linspace(0, n_time - 1, n_store).round().astype(np.int64))
        return idx, times[idx]
    return None, times


def array_qfi_powersums(
    positions,
    polarization,
    Omega: float,
    times: np.ndarray,
    *,
    batch_start: int,
    n_batches: int,
    batch_size: int = 20,
    master_seed: int = 42,
    Delta: float = 0.0,
    Gamma0: float = 1.0,
    n_hat=(0.0, 0.0, -1.0),
    n_store: int | None = None,
    n_jobs: int = 1,
    progress: bool = False,
) -> dict:
    """Semiclassical QFI power sums over globally-indexed trajectory batches.

    Each batch is ``batch_size`` trajectories seeded by
    ``SeedSequence(master_seed, spawn_key=(global_index,))``, so disjoint slices
    (any task/machine) never share trajectories and their power sums sum to a
    single run over their union — the checkpointable unit for cluster runs.
    Returns ``count, s1..s4, times`` and the echoed batch identity.
    """
    positions = np.asarray(positions, dtype=float)
    N = positions.shape[0]
    times = np.asarray(times, dtype=float)
    n_time = len(times)
    dt = float(times[1] - times[0])
    store_idx, store_times = _store_grid(times, n_store)
    n_out = len(store_times)

    Gamma, J = rate_matrices(positions, polarization, Gamma0)
    G = factorize_decay_matrix(Gamma)

    def _args(g):
        cs = np.random.SeedSequence(master_seed, spawn_key=(g,))
        return (N, Omega, Delta, Gamma, J, G, dt, n_time, store_idx,
                batch_size, cs, n_hat)

    globals_ = range(batch_start, batch_start + n_batches)
    parts: dict[int, tuple] = {}
    if n_jobs is None or n_jobs == 1:
        g_iter = globals_
        if progress:
            from tqdm.auto import tqdm
            g_iter = tqdm(g_iter, total=n_batches, desc=f"powersums N={N}", unit="batch")
        for g in g_iter:
            parts[g] = _sc_chunk(*_args(g))
    else:
        if n_jobs < 0:
            n_jobs = os.cpu_count() or 1
        pbar = None
        if progress:
            from tqdm.auto import tqdm
            pbar = tqdm(total=n_batches * batch_size, desc=f"powersums N={N}", unit="traj")
        with ProcessPoolExecutor(max_workers=min(n_jobs, n_batches)) as ex:
            futures = {ex.submit(_sc_chunk, *_args(g)): g for g in globals_}
            for fut in as_completed(futures):
                parts[futures[fut]] = fut.result()
                if pbar is not None:
                    pbar.update(batch_size)
        if pbar is not None:
            pbar.close()

    s1 = np.zeros(n_out); s2 = np.zeros(n_out)
    s3 = np.zeros(n_out); s4 = np.zeros(n_out)
    count = 0
    for g in range(batch_start, batch_start + n_batches):   # canonical order
        c, p1, p2, p3, p4 = parts[g]
        count += c; s1 += p1; s2 += p2; s3 += p3; s4 += p4

    return {"count": count, "s1": s1, "s2": s2, "s3": s3, "s4": s4,
            "times": store_times, "batch_start": batch_start,
            "n_batches": n_batches, "batch_size": batch_size,
            "master_seed": master_seed, "N": N}


def array_qfi_omega(
    positions,
    polarization,
    Omega: float,
    times: np.ndarray,
    n_traj: int,
    seed: int = 42,
    Delta: float = 0.0,
    Gamma0: float = 1.0,
    n_hat=(0.0, 0.0, -1.0),
    n_store: int | None = None,
    batch_size: int = 20,
    n_jobs: int = 1,
    progress: bool = False,
) -> dict:
    """Semiclassical F_sc(t) for ``n_traj`` trajectories (convenience wrapper).

    Returns ``qfi, qfi_stderr, times, n_traj, N``.  F_sc = 4·Var(∂_Ω 𝒮); compare
    directly to :func:`array_monitored_qfi_mcwf` (same drive generator).
    """
    n_batches = (n_traj + batch_size - 1) // batch_size
    ps = array_qfi_powersums(
        positions, polarization, Omega, times,
        batch_start=0, n_batches=n_batches, batch_size=batch_size,
        master_seed=seed, Delta=Delta, Gamma0=Gamma0, n_hat=n_hat,
        n_store=n_store, n_jobs=n_jobs, progress=progress,
    )
    qfi, err = qfi_from_powersums(ps["count"], ps["s1"], ps["s2"], ps["s3"], ps["s4"])
    return {"qfi": qfi, "qfi_stderr": err, "times": ps["times"],
            "n_traj": ps["count"], "N": ps["N"]}


# ── Quantum monitored QFI: exact density-matrix hierarchy (small L) ───────────

def array_monitored_qfi_exact(
    positions,
    polarization,
    Omega: float,
    times: np.ndarray,
    Delta: float = 0.0,
    Gamma0: float = 1.0,
    progress: bool = False,
) -> dict:
    """Exact Gammelmark–Mølmer monitored QFI (no MC / no dt bias).

    Propagates the (M⁰, M¹, M²) density-matrix hierarchy of the two-sided map
    with a single time-independent block Liouvillian, then
    F_mon = 4(−Re Tr M² − |Tr M¹|²).  Feasible only for very small arrays (the
    superoperator is 4ᴺ-dimensional); use :func:`array_monitored_qfi_mcwf` for a
    3×3–4×4 array.  Returns ``qfi, qfi_stderr(=0), times, n_traj(=0), N``.
    """
    from .operators import build_operators

    times = np.asarray(times, dtype=float)
    n_time = len(times)
    Gamma, J = rate_matrices(positions, polarization, Gamma0)
    ops = build_operators(Gamma, J, Omega, Delta)
    H, dH, c_ops, psi0, N = ops["H"], ops["dH"], ops["c_ops"], ops["psi0"], ops["N"]
    dim = 2 ** N
    Id = sp.identity(dim, dtype=complex, format="csc")
    Hc = H.tocsc()

    Lio = -1j * (sp.kron(Id, Hc) - sp.kron(Hc.T, Id))
    for c in c_ops:
        cc = c.tocsc(); cdc = cc.conj().T @ cc
        Lio = Lio + sp.kron(cc.conj(), cc) - 0.5 * sp.kron(Id, cdc) - 0.5 * sp.kron(cdc.conj(), Id)
    Xl = sp.kron(Id, dH.tocsc())                       # ∂_Ω H, ket side
    d2 = dim * dim
    Lbig = sp.bmat([[Lio, None, None],
                    [-1j * Xl, Lio, None],
                    [None, -2j * Xl, Lio]], format="csc")

    vecI = np.zeros(d2, dtype=complex); vecI[::dim + 1] = 1.0
    y = np.zeros(3 * d2, dtype=complex)
    y[:d2] = np.outer(psi0, psi0.conj()).flatten(order="F")

    qfi = np.zeros(n_time)
    step_iter = range(1, n_time)
    if progress:
        from tqdm.auto import tqdm
        step_iter = tqdm(step_iter, desc=f"exact monitored QFI (N={N})")
    for k in step_iter:
        y = expm_multiply(Lbig * (times[k] - times[k - 1]), y)
        m1 = y[d2:2 * d2]; m2 = y[2 * d2:]
        qfi[k] = 4.0 * (-(vecI @ m2).real - abs(vecI @ m1) ** 2)

    return {"qfi": qfi, "qfi_stderr": np.zeros(n_time),
            "times": times, "n_traj": 0, "N": N}


# ── Quantum monitored QFI via statevector MCWF ────────────────────────────────

_MCWF_BLOCK_BYTES = 256 * 2 ** 20      # target dense working set per process


def _traj_block_size(dim, n_traj, traj_block=None):
    """Number of records to propagate together, bounding the dense working set.

    One step keeps ~10 arrays of shape ``(dim, block)`` live at once (ψ, ξ, the
    stacked no-jump buffer and the Taylor temporaries), i.e. ~10·dim·16 B per
    record — 10 MB apiece at N = 16.  Blocking caps that at ``_MCWF_BLOCK_BYTES``
    instead of letting it grow with ``n_traj``; the floor keeps the sparse-matrix
    streaming in ``Mdt @ ·`` amortised over enough columns to stay flop-bound.
    """
    if traj_block is not None:
        return int(np.clip(traj_block, 1, n_traj))
    block = _MCWF_BLOCK_BYTES // (10 * dim * 16)
    return int(np.clip(block, min(8, n_traj), n_traj))


def _monitored_qfi_from_powersums(count, sxx, sxx2, sim, sim2, sxim):
    """Reduce additive MCWF power sums to (qfi, qfi_stderr).

    The Gammelmark–Mølmer QFI of the joint system+record state is

        F_mon = 4( E[⟨ξ|ξ⟩] − ( E[Im⟨ψ|ξ⟩] )² ),

    i.e. the mean of Im⟨ψ|ξ⟩ is squared *after* the ensemble average — matching
    ``|Tr M¹|²`` of :func:`array_monitored_qfi_exact`.  Averaging a per-record
    ``4(⟨ξ|ξ⟩ − (Im⟨ψ|ξ⟩)²)`` instead subtracts E[(Im)²] ≥ (E[Im])² and biases
    F_mon low by 4·Var(Im⟨ψ|ξ⟩), which grows with T (≈ −4 % at Γ₀T = 10, −7 % at
    Γ₀T = 40 for a 2×2 array at Ω = 2Γ₀).

    The sums are over records and hence additive across chunks/workers: partial
    sums may be added before calling.  (E[Im])² uses the unbiased B² − Var(B)/n
    and the error bar is the delta method applied to F(A, B) = 4(A − B²).
    """
    n = count
    A = sxx / n
    Bm = sim / n
    var_A = np.maximum(sxx2 / n - A * A, 0.0)
    var_B = np.maximum(sim2 / n - Bm * Bm, 0.0)
    cov_AB = sxim / n - A * Bm
    var_B_unb = var_B * n / max(n - 1, 1)
    qfi = 4.0 * (A - Bm * Bm + var_B_unb / n)
    var_F = var_A + 4.0 * Bm * Bm * var_B - 4.0 * Bm * cov_AB
    return qfi, 4.0 * np.sqrt(np.maximum(var_F, 0.0) / n)


def _mcwf_chunk(H, c_ops, dH, psi0, times, n_traj, seed, progress=False,
                traj_block=None, progress_q=None):
    """MCWF monitored-QFI power sums over ``n_traj`` records (sparse statevector).

    Along each quantum-jump record the (unnormalised) conditional state ψ̃ and its
    derivative ∂_Ω ψ̃ are propagated; with ξ = ∂_Ω ψ̃/‖ψ̃‖ the monitored QFI is
    F_mon = 4(E[⟨ξ|ξ⟩] − (E[Im⟨ψ|ξ⟩])²).  Both averages are ensemble averages, so
    this returns the power sums ``(n_traj, sxx, sxx2, sim, sim2, sxim)`` of
    x = ⟨ξ|ξ⟩ and y = Im⟨ψ|ξ⟩ — additive across chunks — which
    :func:`_monitored_qfi_from_powersums` reduces.  Collapse operators are
    ω-independent, so a jump sends both ψ, ξ → c_k·(·).

    Jump timing.  One uniform r is drawn per step; a jump is declared when the
    no-jump survival ‖ψ̃(dt)‖² falls below r, and the crossing time is then
    recovered *inside* the step by inverting the (linearly interpolated) survival,
    u = (1 − r)/(1 − ‖ψ̃(dt)‖²).  The record is propagated to u·dt, collapsed
    there (with the channel weights evaluated at the crossing), and propagated
    over the remaining (1 − u)·dt.  Applying the collapse at the *start* of the
    step instead — the textbook first-order recipe — leaves an O(dt) bias of
    ≈ −0.5 % at Γ₀dt = 2.5·10⁻³ and ≈ −2.7 % at Γ₀dt = 10⁻²; interpolating makes
    it O(dt²), so a much coarser dt is affordable.  A second jump inside one step
    is neglected, which is O(dt²) as well.

    s1..s5 are sums over records, so the records are propagated in blocks of
    ``traj_block`` columns (see :func:`_traj_block_size`) and accumulated: peak
    memory is set by the block, not by ``n_traj``.  The RNG stream is consumed
    block by block, so a run is reproducible for a given ``(seed, traj_block)``
    but not across different block sizes — the records stay i.i.d. either way.

    ``progress``  drives a local tqdm (single-process use).  ``progress_q`` is a
    queue on which completed record-steps are posted for a parent process to
    render instead — a worker's own progress is invisible to the parent, so
    without it a pool of k workers can only tick k times, all at the very end.
    """
    rng = np.random.default_rng(seed)
    dim = H.shape[0]
    dt = float(times[1] - times[0])

    A = (-1j * H - 0.5 * sum((c.conj().T @ c) for c in c_ops)).tocsr()   # −i H_eff
    B = (-1j * dH).tocsr()                                               # ket-side source
    # Block generator [[A,0],[B,A]] propagates [ψ; ξ] over one no-jump step.  dt is
    # fixed and dt·‖Mblk‖ ≪ 1, so a fixed-order Taylor series (Horner, TAYLOR_ORDER
    # matvecs) is accurate to well below the O(dt²) MCWF unravelling bias and far
    # cheaper than an adaptive expm — the difference between a feasible and an
    # infeasible 4×4 reference.
    Mdt = (sp.bmat([[A, None], [B, A]], format="csr") * dt).tocsr()
    TAYLOR_ORDER = 4

    def _nojump(Y, u=None):
        """exp(M·u·dt)·Y by Taylor series; ``u`` is a per-column step fraction.

        The j'th Taylor term of exp(M·u·dt) is uʲ·(M·dt)ʲY/j!, so the same
        recursion serves the full step (``u=None``) and the two partial steps
        that straddle a jump.
        """
        acc = Y.copy()
        term = Y
        uj = None if u is None else np.ones_like(u)
        for j in range(1, TAYLOR_ORDER + 1):
            term = (Mdt @ term) / j
            if u is None:
                acc += term
            else:
                uj = uj * u
                acc += term * uj
        return acc

    def _renorm(Y):
        """Split [ψ̃; ξ̃] and divide both by ‖ψ̃‖ (keeps ξ = ∂_Ω ψ̃/‖ψ̃‖)."""
        psi, xi = Y[:dim], Y[dim:]
        inv = 1.0 / np.linalg.norm(psi, axis=0, keepdims=True)
        return psi * inv, xi * inv

    n_time = len(times)
    block = _traj_block_size(dim, n_traj, traj_block)
    sxx = np.zeros(n_time); sxx2 = np.zeros(n_time)
    sim = np.zeros(n_time); sim2 = np.zeros(n_time); sxim = np.zeros(n_time)

    pbar = None
    if progress:
        from tqdm.auto import tqdm
        pbar = tqdm(total=n_traj * (n_time - 1), unit="rec·step",
                    desc=f"MCWF monitored QFI (block={block})")
    # Post upstream ~100 times per block: frequent enough for a smooth bar, rare
    # enough that the queue round-trip stays far below the cost of a step.
    report_every = max(1, (n_time - 1) // 100)
    posted = 0

    for start in range(0, n_traj, block):
        m = min(block, n_traj - start)
        psi = np.tile(psi0.astype(complex)[:, None], (1, m))
        psi /= np.linalg.norm(psi, axis=0, keepdims=True)
        xi = np.zeros((dim, m), dtype=complex)

        for n in range(1, n_time):
            r = rng.random(m)
            # No-jump candidate: propagate stacked [ψ; ξ] by the block generator.
            Y0 = np.vstack([psi, xi])
            Ynj = _nojump(Y0)
            psi_nj = Ynj[:dim]

            nrm2 = np.einsum("ij,ij->j", psi_nj.conj(), psi_nj).real
            psi_new, xi_new = _renorm(Ynj)

            jcols = np.where(nrm2 < r)[0]
            if jcols.size:
                # Crossing time inside the step: ‖ψ̃(u·dt)‖² = r, survival linearised.
                dp = np.maximum(1.0 - nrm2[jcols], np.finfo(float).tiny)
                u = np.clip((1.0 - r[jcols]) / dp, 0.0, 1.0)
                psi_a, xi_a = _renorm(_nojump(Y0[:, jcols], u))

                if len(c_ops) == 1:
                    k_of = np.zeros(jcols.size, dtype=int)
                else:
                    w = np.empty((len(c_ops), jcols.size))
                    for k, c in enumerate(c_ops):
                        cp = c @ psi_a
                        w[k] = np.einsum("ij,ij->j", cp.conj(), cp).real
                    cw = np.cumsum(w, axis=0)
                    pick = rng.random(jcols.size) * cw[-1]
                    k_of = (pick[None, :] > cw).sum(axis=0)

                Yj = np.empty((2 * dim, jcols.size), dtype=complex)
                for k, c in enumerate(c_ops):
                    sel = k_of == k
                    if not sel.any():
                        continue
                    cp = c @ psi_a[:, sel]
                    cx = c @ xi_a[:, sel]
                    nn = 1.0 / np.linalg.norm(cp, axis=0, keepdims=True)
                    Yj[:dim, sel] = cp * nn
                    Yj[dim:, sel] = cx * nn

                # Rest of the step after the collapse (a second jump is O(dt²)).
                psi_c, xi_c = _renorm(_nojump(Yj, 1.0 - u))
                psi_new[:, jcols] = psi_c
                xi_new[:, jcols] = xi_c

            psi, xi = psi_new, xi_new
            xx = np.einsum("ij,ij->j", xi.conj(), xi).real
            yy = np.einsum("ij,ij->j", psi.conj(), xi).imag
            sxx[n] += xx.sum(); sxx2[n] += (xx * xx).sum()
            sim[n] += yy.sum(); sim2[n] += (yy * yy).sum(); sxim[n] += (xx * yy).sum()

            if pbar is not None:
                pbar.update(m)
            elif progress_q is not None:
                posted += m
                if n % report_every == 0:
                    progress_q.put(posted)
                    posted = 0

    if pbar is not None:
        pbar.close()
    if progress_q is not None and posted:
        progress_q.put(posted)

    return n_traj, sxx, sxx2, sim, sim2, sxim


def array_monitored_qfi_mcwf(
    positions,
    polarization,
    Omega: float,
    times: np.ndarray,
    n_traj: int = 500,
    seed: int = 42,
    Delta: float = 0.0,
    Gamma0: float = 1.0,
    n_jobs: int = 1,
    traj_block: int | None = None,
    progress: bool = False,
) -> dict:
    """MCWF estimate of the monitored QFI F_mon(t) for the driven array.

    Reference for the semiclassical :func:`array_qfi_omega` (same drive generator
    ∂_Ω H = −Σ_n σ^x_n, so compare F_mon ≈ F_sc directly).  Feasible up to a 4×4
    array.  Returns ``qfi, qfi_stderr, times, n_traj, N``.

    F_mon is a *variance*, so its estimator converges like √((κ−1)/n) with an
    effective kurtosis κ ≈ 10–16 here (against κ ≈ 3 for the semiclassical F_sc):
    ~10⁴ records give ≈ 3 % at Γ₀T = 10 and ~10⁵ are needed for 1 %.  Bounded
    one-time observables such as ⟨Ŝᶻ⟩ converge far faster — trajectory counts
    quoted for those (e.g. 10³ in arXiv:2305.19829) do not carry over.

    Each worker propagates its records in blocks of ``traj_block`` columns
    (default: auto, ~256 MB of dense state per process), so peak memory is set by
    the block size and the 2ᴺ operators rather than by ``n_traj``.
    """
    from .operators import build_operators

    times = np.asarray(times, dtype=float)
    n_time = len(times)
    Gamma, J = rate_matrices(positions, polarization, Gamma0)
    ops = build_operators(Gamma, J, Omega, Delta)
    H, dH, c_ops, psi0, N = ops["H"], ops["dH"], ops["c_ops"], ops["psi0"], ops["N"]

    if n_jobs is None or n_jobs == 1:
        count, sxx, sxx2, sim, sim2, sxim = _mcwf_chunk(
            H, c_ops, dH, psi0, times, n_traj, seed, progress, traj_block)
    else:
        if n_jobs < 0:
            n_jobs = os.cpu_count() or 1
        n_workers = min(n_jobs, n_traj)
        counts = [n_traj // n_workers + (1 if i < n_traj % n_workers else 0)
                  for i in range(n_workers)]
        child_seeds = np.random.SeedSequence(seed).spawn(n_workers)
        sxx = np.zeros(n_time); sxx2 = np.zeros(n_time)
        sim = np.zeros(n_time); sim2 = np.zeros(n_time); sxim = np.zeros(n_time)
        count = 0
        # Workers post completed record-steps on a shared queue that the parent
        # drains while it waits, so the bar advances continuously.  Ticking once
        # per finished future instead would leave it empty until the very end,
        # since every worker gets an equal slice and they all land together.
        manager = pbar = queue_ = None
        if progress:
            import multiprocessing
            from tqdm.auto import tqdm
            manager = multiprocessing.Manager()
            queue_ = manager.Queue()
            pbar = tqdm(total=n_traj * (n_time - 1), unit="rec·step",
                        desc=f"MCWF monitored QFI ({n_workers} workers)")

        def _drain():
            if pbar is None:
                return
            while True:
                try:
                    pbar.update(queue_.get_nowait())
                except _queue.Empty:
                    return

        try:
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                futures = [ex.submit(_mcwf_chunk, H, c_ops, dH, psi0, times, c, cs,
                                     False, traj_block, queue_)
                           for c, cs in zip(counts, child_seeds)]
                index = {f: i for i, f in enumerate(futures)}
                parts: dict[int, tuple] = {}
                pending = set(futures)
                while pending:
                    finished, pending = wait(pending, timeout=0.25)
                    _drain()
                    for fut in finished:
                        parts[index[fut]] = fut.result()
                _drain()
            # Sum in submission order: float addition is not associative, so
            # accumulating as workers happen to finish would make a run depend on
            # scheduling.  (Same convention as `array_qfi_powersums`.)
            for i in range(len(futures)):
                c, a1, a2, b1, b2, ab = parts[i]
                count += c
                sxx += a1; sxx2 += a2; sim += b1; sim2 += b2; sxim += ab
        finally:
            if pbar is not None:
                pbar.close()
            if manager is not None:
                manager.shutdown()

    qfi, err = _monitored_qfi_from_powersums(count, sxx, sxx2, sim, sim2, sxim)
    return {"qfi": qfi, "qfi_stderr": err,
            "times": times, "n_traj": count, "N": N}
