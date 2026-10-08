"""Semiclassical and quantum Fisher information for sensing the Rabi frequency ω
with a BTC model.
"""

import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply

from .sde import (
    angles_to_spin,
    collective_noise_factor,
    make_collective_stepper,
    _sample_collective_initial,
)
# NOTE: make_operators / make_initial_state (from .operators) pull in
# QuTiP and are only needed by the quantum QFI functions.  They are imported
# lazily inside those functions so the semiclassical path (this script's use
# case) runs on a bare NumPy/SciPy stack, without QuTiP installed.


# -- Collective (BTC) semiclassical QFI ----------------------------------------

def _collective_qfi_chunk(
    N: int,
    omega: float,
    delta: float,
    kappa: float,
    dt: float,
    n_time: int,
    store_idx: np.ndarray | None,
    n_traj: int,
    seed,
    sampling: str,
    state: str,
    theta0: float,
    phi0: float,
    spread: float,
    Gamma: np.ndarray | None = None,
    progress: bool = False,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Integrate ``n_traj`` collective realisations of spin observables,
    accumulate powersums of these observables to compute the QFI.

    Parameters
    ----------
    N, omega, delta, kappa, dt, n_time, store_idx : as in :func:`collective_qfi_omega`
    n_traj : number of trajectories
    seed   : seed for this chunk's RNG
    sampling, state, theta0, phi0, spread, Gamma : as in
             :func:`collective_qfi_omega`
    """
    rng = np.random.default_rng(seed)
    S = N / 2.0

    G = collective_noise_factor(N, kappa, Gamma)
    step = make_collective_stepper(omega, delta, G)

    theta, phi = _sample_collective_initial(
        N * n_traj, rng, sampling, state, theta0, phi0, spread, S
    )
    theta = theta.reshape(N, n_traj)
    phi = phi.reshape(N, n_traj)

    # Cartesian state (N, n_traj, 3) for the pole-safe rotation solver.
    s = np.stack(angles_to_spin(theta, phi), axis=-1)

    # Output length: full grid, or only the requested sub-sampled points.
    n_out = n_time if store_idx is None else len(store_idx)

    s1 = np.zeros(n_out) # 1st and 2nd moment needed for QFI
    s2 = np.zeros(n_out)
    s3 = np.zeros(n_out) # 3rd and 4th moment needed for stderr 
    s4 = np.zeros(n_out)

    mx_prev = s[..., 0].mean(axis=0)   # (n_traj,)
    dS = np.zeros(n_traj)           # trapezoidal integral of x component
    half_dt = 0.5 * dt

    step_iter = range(1, n_time)
    if progress:
        from tqdm.auto import tqdm

        step_iter = tqdm(step_iter, desc="collective QFI")

    if store_idx is None:
        # Record every integration step (memory O(n_time)).
        for n in step_iter:
            s = step(s, dt, rng)
            mx = s[..., 0].mean(axis=0)
            dS -= half_dt * (mx_prev + mx)   
            mx_prev = mx
            dS2 = dS * dS
            s1[n] = dS.sum()
            s2[n] = dS2.sum()
            s3[n] = (dS2 * dS).sum()
            s4[n] = (dS2 * dS2).sum()
    else:
        # Integrate at full dt but store only at the requested indices
        # (memory O(n_store)); store_idx[0] == 0 is the t=0 point, left at 0.
        k = 1
        next_store = store_idx[k] if n_out > 1 else -1
        for n in step_iter:
            s = step(s, dt, rng)
            mx = s[..., 0].mean(axis=0)
            dS -= half_dt * (mx_prev + mx)   
            mx_prev = mx
            if n == next_store:
                dS2 = dS * dS
                s1[k] = dS.sum()
                s2[k] = dS2.sum()
                s3[k] = (dS2 * dS).sum()
                s4[k] = (dS2 * dS2).sum()
                k += 1
                if k < n_out:
                    next_store = store_idx[k]

    return n_traj, s1, s2, s3, s4


def collective_qfi_omega(
    N: int,
    omega: float,
    delta: float,
    kappa: float,
    times: np.ndarray,
    n_traj: int,
    seed: int = 42,
    sampling: str = "dtwa",
    n_jobs: int = 1,
    state: str = "down",
    theta0: float = np.pi / 2,
    phi0: float = 0.0,
    spread: float = 0.0,
    Gamma: np.ndarray | None = None,
    n_store: int | None = None,
    batch_size: int = 25,
    progress: bool = False,
) -> dict:
    """Collective-BTC semiclassical QFI for sensing the Rabi frequency. This 
    function is best used for small scale runs on a single machine. For large scale runs, use
    :func:`collective_qfi_powersums` to accumulate the power sums in batches and then
    call :func:`qfi_from_powersums` to compute the QFI and its standard error.

    Parameters
    ----------
    N        : number of spins 
    omega    : driving frequency 
    delta    : detuning 
    kappa    : single-particle decay rate 
    times    : 1-D time array with uniform spacing
    n_traj   : number of trajectories
    seed     : random seed for reproducibility
    sampling : initial-condition scheme - "dtwa" (default), "wigner_cone" or
               "continuous"
    n_jobs   : number of worker processes. Default 1; use -1 for all available cores.
    state    : "up" / "down" initial polarisation for "dtwa" sampling
    theta0,
    phi0     : mean initial angles for "continuous" sampling
    spread   : Gaussian spread of the "continuous" initial condition
    Gamma    : optional (N, N) PSD decay matrix; ``None`` (default) uses the
               all-to-all Dicke Γ_mn = κ
    n_store  : number of stored intermediate time points to reduce memory
    batch_size : trajectories per independently-seeded batch 
    progress : tqdm progress bar over the batches

    Returns
    -------
    dict with keys:
        'qfi'        - semiclassical QFI at each stored time
        'qfi_stderr' - Monte-Carlo standard error of the QFI
        'times'      - the stored time points matching 'qfi'
        'n_traj'     - number of trajectories actually used
        'N'          - number of spins 
    """
    times = np.asarray(times, dtype=float)
    n_time = len(times)
    if n_time < 2:
        raise ValueError("`times` must contain at least two points.")
    if int(N) != N or N < 1:
        raise ValueError("`N` must be a positive integer.")
    N = int(N)
    dt = float(times[1] - times[0])

    # Decouple stored output from integration resolution: integrate every dt
    # step but keep only ``n_store`` evenly spaced points (endpoints included).
    if n_store is not None and n_store < n_time:
        if n_store < 2:
            raise ValueError("`n_store` must be at least 2.")
        store_idx = np.unique(
            np.linspace(0, n_time - 1, n_store).round().astype(np.int64)
        )
        store_times = times[store_idx]
    else:
        store_idx = None
        store_times = times
    n_out = len(store_times)

    if batch_size < 1:
        raise ValueError("`batch_size` must be a positive integer.")

    # Reproducible, n_jobs-independent Monte Carlo: partition the realisations
    # into fixed-size batches, each with its own deterministic spawned seed.  A
    # batch's trajectories depend only on (seed, batch_size, batch index), never
    # on the worker layout, so any n_jobs gives the same result.
    n_batches = (n_traj + batch_size - 1) // batch_size
    batch_counts = [
        min(batch_size, n_traj - b * batch_size) for b in range(n_batches)
    ]
    batch_seeds = np.random.SeedSequence(seed).spawn(n_batches)

    def _batch_args(b):
        return (N, omega, delta, kappa, dt, n_time, store_idx,
                batch_counts[b], batch_seeds[b],
                sampling, state, theta0, phi0, spread, Gamma)

    if n_jobs is None or n_jobs == 1:
        batch_iter = range(n_batches)
        if progress:
            from tqdm.auto import tqdm

            batch_iter = tqdm(batch_iter, total=n_batches, desc="collective QFI")
        parts = [_collective_qfi_chunk(*_batch_args(b)) for b in batch_iter]
    else:
        if n_jobs < 0:
            n_jobs = os.cpu_count() or 1
        n_workers = min(n_jobs, n_batches)
        parts = [None] * n_batches
        pbar = None
        if progress:
            from tqdm.auto import tqdm

            pbar = tqdm(total=n_traj, desc="collective QFI",
                        unit="traj", dynamic_ncols=True)
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = {
                ex.submit(_collective_qfi_chunk, *_batch_args(b)): b
                for b in range(n_batches)
            }
            for fut in as_completed(futures):
                res = fut.result()
                parts[futures[fut]] = res
                if pbar is not None:
                    pbar.update(res[0])   # res[0] = trajectories in this batch
        if pbar is not None:
            pbar.close()

    # Reduce the power sums in fixed batch order,
    # so the combined result is independent of worker completion order too.
    s1 = np.zeros(n_out)
    s2 = np.zeros(n_out)
    s3 = np.zeros(n_out)
    s4 = np.zeros(n_out)
    count = 0
    for c, p1, p2, p3, p4 in parts:
        count += c
        s1 += p1
        s2 += p2
        s3 += p3
        s4 += p4

    n = count
    mean = s1 / n
    # Compute average QFI over trajectories
    qfi = 4.0 * (s2 - n * mean * mean) / (n - 1)
    # Compute Monte-Carlo standard error of the QFI via the 4th-moment formula
    m2 = s2 / n - mean * mean
    m4 = s4 / n - 4.0 * mean * (s3 / n) + 6.0 * mean * mean * (s2 / n) - 3.0 * mean**4
    var_S2 = (m4 - (n - 3) / (n - 1) * m2 * m2) / n
    qfi_stderr = 4.0 * np.sqrt(np.maximum(var_S2, 0.0))

    return {
        "qfi": qfi,
        "qfi_stderr": qfi_stderr,
        "times": store_times,
        "n_traj": count,
        "N": N,
    }


# -- Batchwise collective QFI --------------------------------------------------

def qfi_from_powersums(
    count: int,
    s1: np.ndarray,
    s2: np.ndarray,
    s3: np.ndarray,
    s4: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce accumulated power sums from :func:`collective_qfi_powersums` to 
    the semiclassical QFI and its Monte-Carlo error.

    Returns ``(qfi, qfi_stderr)``.
    """
    n = count
    mean = s1 / n
    qfi = 4.0 * (s2 - n * mean * mean) / (n - 1)
    m2 = s2 / n - mean * mean
    m4 = s4 / n - 4.0 * mean * (s3 / n) + 6.0 * mean * mean * (s2 / n) - 3.0 * mean**4
    var_S2 = (m4 - (n - 3) / (n - 1) * m2 * m2) / n
    qfi_stderr = 4.0 * np.sqrt(np.maximum(var_S2, 0.0))
    return qfi, qfi_stderr


def collective_qfi_powersums(
    N: int,
    omega: float,
    delta: float,
    kappa: float,
    times: np.ndarray,
    *,
    batch_start: int,
    n_batches: int,
    batch_size: int = 20,
    master_seed: int = 42,
    sampling: str = "wigner_cone",
    state: str = "down",
    theta0: float = np.pi / 2,
    phi0: float = 0.0,
    spread: float = 0.0,
    Gamma: np.ndarray | None = None,
    n_store: int | None = None,
    n_jobs: int = 1,
    progress: bool = False,
) -> dict:
    """Checkpointable, batchwise accumulation of power sums of observables
    from which the collective QFI can be computed. Helpful for simulations
    on a cluster.

    Parameters
    ----------
    N, omega, delta, kappa, times : as in :func:`collective_qfi_omega`
    batch_start : index of first global batch for this call 
    n_batches   : number of batches 
    batch_size  : trajectories per batch 
    master_seed : campaign-wide seed; global batch index supplies independence
    sampling, state, theta0, phi0, spread, Gamma : as in
                  :func:`collective_qfi_omega`
    n_store     : keep only this many evenly-spaced time points; 
                    ``None`` keeps every step
    n_jobs      : number of worker processes 
    progress    : trajectory-weighted tqdm bar

    Returns
    -------
    dict with keys ``count`` (= ``n_batches * batch_size``), ``s1``..``s4``
    (each shape ``(n_out,)``), ``times`` (the stored points), and the echoed
    ``batch_start``, ``n_batches``, ``batch_size``, ``master_seed``.
    """
    times = np.asarray(times, dtype=float)
    n_time = len(times)
    if n_time < 2:
        raise ValueError("`times` must contain at least two points.")
    if int(N) != N or N < 1:
        raise ValueError("`N` must be a positive integer.")
    N = int(N)
    if batch_size < 1 or n_batches < 1 or batch_start < 0:
        raise ValueError("`batch_size`/`n_batches` must be ≥1 and `batch_start` ≥0.")
    dt = float(times[1] - times[0])

    if n_store is not None and n_store < n_time:
        if n_store < 2:
            raise ValueError("`n_store` must be at least 2.")
        store_idx = np.unique(
            np.linspace(0, n_time - 1, n_store).round().astype(np.int64)
        )
        store_times = times[store_idx]
    else:
        store_idx = None
        store_times = times
    n_out = len(store_times)

    globals_ = range(batch_start, batch_start + n_batches)

    def _args(g):
        cs = np.random.SeedSequence(master_seed, spawn_key=(g,))
        return (N, omega, delta, kappa, dt, n_time, store_idx,
                batch_size, cs, sampling, state, theta0, phi0, spread, Gamma)

    # parts keyed by global index so the reduction order is canonical (the
    # combined result is independent of worker/scheduling order).
    parts: dict[int, tuple] = {}
    if n_jobs is None or n_jobs == 1:
        g_iter = globals_
        if progress:
            from tqdm.auto import tqdm

            g_iter = tqdm(g_iter, total=n_batches, desc=f"powersums N={N}",
                          unit="batch", dynamic_ncols=True)
        for g in g_iter:
            parts[g] = _collective_qfi_chunk(*_args(g))
    else:
        if n_jobs < 0:
            n_jobs = os.cpu_count() or 1
        n_workers = min(n_jobs, n_batches)
        pbar = None
        if progress:
            from tqdm.auto import tqdm

            pbar = tqdm(total=n_batches * batch_size, desc=f"powersums N={N}",
                        unit="traj", dynamic_ncols=True)
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(_collective_qfi_chunk, *_args(g)): g
                       for g in globals_}
            for fut in as_completed(futures):
                parts[futures[fut]] = fut.result()
                if pbar is not None:
                    pbar.update(batch_size)
        if pbar is not None:
            pbar.close()

    s1 = np.zeros(n_out)
    s2 = np.zeros(n_out)
    s3 = np.zeros(n_out)
    s4 = np.zeros(n_out)
    count = 0
    for g in range(batch_start, batch_start + n_batches):   # canonical order
        c, p1, p2, p3, p4 = parts[g]
        count += c
        s1 += p1
        s2 += p2
        s3 += p3
        s4 += p4

    return {
        "count": count, "s1": s1, "s2": s2, "s3": s3, "s4": s4,
        "times": store_times, "batch_start": batch_start, "n_batches": n_batches,
        "batch_size": batch_size, "master_seed": master_seed,
    }


# -- Quantum QFI helpers -------------------------------------------------------

def _build_liouvillian_matrix(
    H_mat: sp.spmatrix,
    Sm_mat: sp.spmatrix,
    kappa: float,
) -> sp.csr_matrix:
    """Liouvillian the superoperator for the BTC as a sparse (dim², dim²) matrix.

    Parameters
    ----------
    H_mat  : (dim, dim) complex Hamiltonian, sparse
    Sm_mat : (dim, dim) collective lowering operator S₋, sparse
    kappa  : single-particle decay rate

    Returns
    -------
    L : (dim², dim²) complex sparse matrix (CSR)
    """
    dim = H_mat.shape[0]
    I = sp.identity(dim, dtype=complex, format="csr")
    Sp_mat = Sm_mat.conj().T.tocsr()    
    SpSm = (Sp_mat @ Sm_mat).tocsr()    

    # Hamiltonian contribution
    L = -1j * (sp.kron(I, H_mat, format="csr") - sp.kron(H_mat.T, I, format="csr"))

    # Dissipative contribution
    L = L + kappa * (
        sp.kron(Sm_mat.conj(), Sm_mat, format="csr")
        - 0.5 * sp.kron(I, SpSm, format="csr")
        - 0.5 * sp.kron(SpSm.conj(), I, format="csr")
    )
    return L.tocsr()


def _uhlmann_fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Uhlmann fidelity for two states ``rho`` and ``sigma``.

    Parameters
    ----------
    rho   : (dim, dim) density matrix
    sigma : (dim, dim) density matrix

    Returns
    -------
    F : real float in [0, 1]
    """
    # square root of matrices via eigendecomposition
    evals_rho, evecs_rho = np.linalg.eigh(rho)
    evals_rho = np.maximum(evals_rho.real, 0.0)
    sqrt_rho = evecs_rho @ np.diag(np.sqrt(evals_rho)) @ evecs_rho.conj().T
    M = sqrt_rho @ sigma @ sqrt_rho
    evals_M = np.linalg.eigvalsh(M)
    evals_M = np.maximum(evals_M.real, 0.0)
    return float(np.sum(np.sqrt(evals_M)) ** 2)


# -- Quantum QFI (full) --------------------------------------------------------

def _expm_propagate(
    L: sp.spmatrix,
    y0: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """Propagate the vector y(t) = expm(L·(t - times[0])) y0 at each requested time.

    Parameters
    ----------
    L     : (d, d) constant generator, sparse
    y0    : (d,) state at ``times[0]``
    times : 1-D time array (need not be uniform)

    Returns
    -------
    y_t : (n_time, d) state at each requested time
    """
    times = np.asarray(times, dtype=float)
    n = len(times)
    if n == 1:
        return y0[None, :]

    elapsed = times - times[0]  # propagate relative to the state y0 is given at
    if np.allclose(np.diff(elapsed), elapsed[1] - elapsed[0]):
        return expm_multiply(L, y0, start=0.0, stop=elapsed[-1], num=n, endpoint=True)

    y_t = np.empty((n, len(y0)), dtype=complex)
    y_t[0] = y0
    y = y0
    for i in range(1, n):
        y = expm_multiply(L * (elapsed[i] - elapsed[i - 1]), y)
        y_t[i] = y
    return y_t


def _solve_lindblad(
    L_mat: sp.spmatrix,
    rho0: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """Propagate the vectorised Lindblad equation and return system density matrix
      at each time.

    Parameters
    ----------
    L_mat : (dim², dim²) Liouvillian superoperator, sparse
    rho0  : (dim, dim) initial density matrix (state at ``times[0]``)
    times : 1-D time array (need not be uniform)

    Returns
    -------
    rho_t : (n_time, dim, dim) density matrices
    """
    dim = rho0.shape[0]
    y0 = rho0.flatten(order="F")  
    y_t = _expm_propagate(L_mat, y0, times)
    return np.array([y.reshape(dim, dim, order="F") for y in y_t])


def quantum_qfi_omega(
    S: float,
    omega: float,
    kappa: float,
    times: np.ndarray,
    delta: float = 1e-3,
    state="up",
) -> dict:
    """Quantum QFI of the system only state for sensing ω via the fidelity formula.

    Parameters
    ----------
    S     : total spin quantum number
    omega : driving frequency  
    kappa : single-particle decay rate
    times : 1-D time array (need not be uniform)
    delta : finite-difference step for omega; default 1e-4
    state : initial coherent-state Bloch direction (name or length-3 vector),
            matching the DTWA sampler; default "up".

    Returns
    -------
    dict with keys:
        'qfi'   - F_Q(T) at each time, shape (n_time,)
        'times' - the input time array
    """
    from .operators import make_operators, make_initial_state  

    ops = make_operators(S)
    Sm_mat = sp.csr_matrix(ops["Sm"].full().astype(complex))
    Sx_mat = sp.csr_matrix(ops["Sx"].full().astype(complex))

    rho0 = make_initial_state(S, state).full().astype(complex)

    # Build Liouvillians for ω and ω + δ
    L_omega   = _build_liouvillian_matrix(omega * Sx_mat,         Sm_mat, kappa)
    L_shifted = _build_liouvillian_matrix((omega + delta) * Sx_mat, Sm_mat, kappa)

    # Propagate both
    rho_omega   = _solve_lindblad(L_omega,   rho0, times)   # (n_time, dim, dim)
    rho_shifted = _solve_lindblad(L_shifted, rho0, times)

    # QFI at each time via fidelity formula
    qfi_vals = np.empty(len(times))
    for k in range(len(times)):
        F = _uhlmann_fidelity(rho_omega[k], rho_shifted[k])
        qfi_vals[k] = 8.0 * (1.0 - F) / delta**2

    return {"qfi": qfi_vals, "times": times}


# -- Quantum QFI for continuous monitoring of system and environment state -----

def _block_boundaries(times: np.ndarray, block_time: float) -> np.ndarray:
    """Grid indices partitioning times into blocks spanning ca. block_time.
    """
    elapsed = np.asarray(times, dtype=float) - float(times[0])
    total = float(elapsed[-1])
    if block_time <= 0.0 or block_time >= total:
        return np.array([0, len(times) - 1])
    n_blocks = int(np.ceil(total / block_time))
    targets = np.minimum(np.arange(1, n_blocks + 1) * block_time, total)
    # first grid point at or beyond each block's target elapsed time
    idx = np.searchsorted(elapsed, targets, side="left")
    return np.unique(np.concatenate(([0], np.clip(idx, 1, len(times) - 1))))


def _save_checkpoint(path: str, **arrays) -> None:
    """Atomically write arrays to path as an .npz (temp file + rename) as checkpoints.
    """
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(suffix=".npz", dir=directory)
    try:
        with os.fdopen(fd, "wb") as fh:
            np.savez(fh, **arrays)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def quantum_monitored_qfi_omega(
    S: float,
    omega: float,
    kappa: float,
    times: np.ndarray,
    state="down",
    progress: bool = False,
    checkpoint_path: str | None = None,
    checkpoint_every: int = 1,
    block_time: float = 1.0,
    resume: bool = True,
) -> dict:
    r"""Gammelmark-Molmer QFI for sensing the BTC driving frequency
      under continuous monitoring. Ultimate bound on precision for this system.

    Parameters
    ----------
    S     : total spin quantum number
    omega : driving frequency 
    kappa : single-particle decay rate
    times : 1-D time array 
    state : initial coherent-state Bloch direction (name or length-3 vector);
            must be pure.  Default "up".
    progress : show a tqdm progress bar that advances once per block_time of
            simulated time.
    checkpoint_path : path to write checkpoints to
    checkpoint_every : write a checkpoint every this many blocks; default is 1
    block_time : simulated-time span propagated between progress ticks
    resume : if True (default) and checkpoint_path already exists

    Returns
    -------
    dict with keys:
        'qfi'   - F_Q^monitored(t) at each time, shape (n_time,)
        'times' - the input time array
    """
    from .operators import make_operators, make_initial_state  

    times = np.asarray(times, dtype=float)
    n_time = len(times)
    if n_time < 2:
        raise ValueError("`times` must contain at least two points.")

    ops = make_operators(S)
    Sm_mat = sp.csr_matrix(ops["Sm"].full().astype(complex))
    Sx_mat = sp.csr_matrix(ops["Sx"].full().astype(complex))

    dim = Sx_mat.shape[0]
    I = sp.identity(dim, dtype=complex, format="csr")

    # Ordinary Lindbladian and the ket-side generator 
    L_mat = _build_liouvillian_matrix(omega * Sx_mat, Sm_mat, kappa)
    X_left = sp.kron(I, Sx_mat, format="csr")   # vec(Sx * M) = X_left * vec(M)

    rho0 = make_initial_state(S, state).full().astype(complex)
    d2 = dim * dim

    # Stack (M_0, M_1, M_2) into one vector propagated by a single
    # time-independent block generator (the recursion above), so the whole
    # hierarchy is one Krylov matrix-exponential action.
    Lbig = sp.bmat(
        [[L_mat, None, None],
         [-1j * X_left, L_mat, None],
         [None, -2j * X_left, L_mat]],
        format="csr",
    )

    # Tr M = vec(I) · vec(M); the col-major vec of I_dim has 1's at stride dim+1.
    vec_I = np.zeros(d2, dtype=complex)
    vec_I[:: dim + 1] = 1.0

    def _block_qfi(y_block: np.ndarray) -> np.ndarray:
        """Monitored QFI at each row of ``y_block`` (n_pts, 3 d²)."""
        m1 = y_block[:, d2:2 * d2].T            # (d², n_pts)
        m2 = y_block[:, 2 * d2:].T
        tr_m1 = vec_I @ m1
        tr_m2 = vec_I @ m2
        return 4.0 * (-tr_m2.real - np.abs(tr_m1) ** 2)

    # Partition the time grid into equal-simulated-time blocks and propagate one
    # at a time, carrying the state forward.  Only per-block traces are kept, so
    # the full (n_time, 3 d²) state history is never materialised.
    bounds = _block_boundaries(times, block_time)
    n_blocks = len(bounds) - 1

    qfi_vals = np.full(n_time, np.nan)
    qfi_vals[0] = 0.0                       # M_1 = M_2 = 0 at t = 0 --> F_Q = 0
    y_current = np.zeros(3 * d2, dtype=complex)
    y_current[:d2] = rho0.flatten(order="F")   # M_0(0) = rho_0, others zero
    start_block = 0

    # Resume from a compatible existing checkpoint if requested.
    if resume and checkpoint_path is not None and os.path.exists(checkpoint_path):
        with np.load(checkpoint_path) as ck:
            ck_times = ck["times"]
            incompatible = (
                ck_times.shape != times.shape
                or not np.allclose(ck_times, times)
                or float(ck["S"]) != float(S)
                or float(ck["omega"]) != float(omega)
                or float(ck["kappa"]) != float(kappa)
                or float(ck["block_time"]) != float(block_time)
            )
            if incompatible:
                raise ValueError(
                    f"Checkpoint '{checkpoint_path}' was written for different "
                    "parameters (S / omega / kappa / times / block_time); refusing "
                    "to resume.  Pass a fresh `checkpoint_path` or `resume=False` "
                    "to start over."
                )
            qfi_vals = ck["qfi"].copy()
            y_current = ck["y_current"].copy()
            start_block = int(ck["blocks_done"])

    block_iter = range(start_block, n_blocks)
    if progress:
        from tqdm.auto import tqdm

        block_iter = tqdm(
            block_iter, total=n_blocks, initial=start_block,
            desc=f"monitored QFI (S={S:g})", unit="block", dynamic_ncols=True,
        )

    for b in block_iter:
        i0, i1 = int(bounds[b]), int(bounds[b + 1])
        # Propagate over this block starting from the state at i0; seg[0] is that
        # (already-recorded) state, seg[1:] are the new points i0+1 .. i1.
        seg = _expm_propagate(Lbig, y_current, times[i0:i1 + 1])
        qfi_vals[i0 + 1:i1 + 1] = _block_qfi(seg[1:])
        y_current = seg[-1]

        if checkpoint_path is not None and (
            (b + 1) % checkpoint_every == 0 or b == n_blocks - 1
        ):
            _save_checkpoint(
                checkpoint_path,
                qfi=qfi_vals, times=times, y_current=y_current,
                blocks_done=b + 1, S=float(S), omega=float(omega),
                kappa=float(kappa), block_time=float(block_time),
            )

    return {"qfi": qfi_vals, "times": times}