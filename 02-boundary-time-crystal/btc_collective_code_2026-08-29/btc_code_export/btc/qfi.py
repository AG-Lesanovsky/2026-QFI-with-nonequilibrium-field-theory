"""Semiclassical and quantum Fisher information for sensing the Rabi frequency ω.

Semiclassical QFI
-----------------
The classical action derivative w.r.t. ω along a stochastic trajectory is

    ∂_ω S = −∫₀ᵀ dt  sₓ(t)

and the semiclassical QFI is

    F_sc(T) = 4 · Var_{traj}(∂_ω S)

where the variance is taken over the ensemble of stochastic trajectories
(including the randomness from both the noise realisations and the Wigner /
DTWA initial conditions).

Quantum QFI via the fidelity formula
-------------------------------------
The QFI is estimated from the Uhlmann fidelity between two states evolved
under slightly different Rabi frequencies ω and ω + δ:

    F_Q(T) = 8 · (1 − F(ρ_ω(T), ρ_{ω+δ}(T))) / δ²

where the Uhlmann fidelity is

    F(ρ, σ) = ( Tr √(√ρ σ √ρ) )²

and both density matrices are obtained by integrating the Lindblad master
equation

    dρ/dt = −i[H, ρ] + κ (S₋ ρ S₊ − ½{S₊S₋, ρ}),   H = ω Sₓ,

from the same initial state ρ(0) = |S, +S⟩⟨S, +S|.  The finite-difference
step δ should be small enough that the result is converged but large enough
to avoid numerical cancellation (δ ~ 10⁻⁴ – 10⁻³ is typically appropriate).

Parameter conventions
---------------------
The semiclassical decay rate γ and the quantum single-particle rate κ are
related by

    γ = κ N = 2 κ S         (mean-field / thermodynamic limit)

Pass γ to semiclassical functions and κ to quantum functions.

Spin normalisation
------------------
The semiclassical spin components sₓ ∈ [−1, 1] are normalised by S:
sₓ = Sₓ / S.  The quantum QFI therefore scales as S² relative to the
semiclassical one.  To compare, divide the quantum QFI by S²:

    F_Q / S² ≈ F_sc     (large-N limit)
"""

import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply

from .sde import (
    angles_to_spin,
    rotation_step,
    collective_noise_factor,
    make_collective_stepper,
    sample_initial_conditions,
    sample_initial_conditions_dtwa,
    _sample_collective_initial,
)
# NOTE: ``make_operators`` / ``make_initial_state`` (from .operators) pull in
# QuTiP and are only needed by the *quantum* QFI functions.  They are imported
# lazily inside those functions so the semiclassical path (this script's use
# case) runs on a bare NumPy/SciPy stack, without QuTiP installed.


# ── Semiclassical QFI ─────────────────────────────────────────────────────────

def _semiclassical_qfi_chunk(
    S: float,
    omega: float,
    delta: float,
    gamma_step: float,
    times: np.ndarray,
    n_traj: int,
    seed,
    sampling: str,
    state: str,
    theta0: float,
    phi0: float,
    spread: float,
    progress: bool = False,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Integrate ``n_traj`` trajectories, accumulating running QFI statistics.

    With ∂_ω S^(k)(t) = −∫₀ᵗ sₓ^(k)(t') dt', this returns the per-time partial
    sums needed to combine the variance across chunks:

        s1[t] = Σ_k ∂_ω S^(k)(t),   s2[t] = Σ_k (∂_ω S^(k)(t))².

    Only O(n_time) + O(n_traj) memory is used — the full (n_time, n_traj)
    trajectory array is never materialised.  ``seed`` is anything
    :func:`numpy.random.default_rng` accepts (an ``int`` or a ``SeedSequence``).
    """
    rng = np.random.default_rng(seed)
    n_time = len(times)
    dt = float(times[1] - times[0])

    if sampling == "dtwa":
        theta, phi = sample_initial_conditions_dtwa(n_traj, rng, state)
    elif sampling == "continuous":
        theta, phi = sample_initial_conditions(
            n_traj, rng, theta0, phi0, spread / np.sqrt(2.0 * S)
        )
    else:
        raise ValueError(
            f"Unknown sampling scheme '{sampling}'. Choose 'dtwa' or 'continuous'."
        )

    s1 = np.zeros(n_time)
    s2 = np.zeros(n_time)

    # Cartesian state for the pole-safe rotation solver; sₓ is the x-component.
    s = np.stack(angles_to_spin(theta, phi), axis=-1)   # (n_traj, 3)

    # ∂_ω S(0) = 0 for every trajectory; sₓ = √3 sin θ cos φ = s[..., 0].
    sx_prev = s[..., 0]
    dS = np.zeros(n_traj)          # running −∫ sₓ dt  (trapezoid)
    half_dt = 0.5 * dt

    step_iter = range(1, n_time)
    if progress:
        from tqdm.auto import tqdm

        step_iter = tqdm(step_iter, desc="semiclassical QFI")

    for n in step_iter:
        s = rotation_step(s, dt, omega, delta, gamma_step, rng)
        sx = s[..., 0]
        dS -= half_dt * (sx_prev + sx)     # trapezoid increment of −∫ sₓ dt
        sx_prev = sx
        s1[n] = dS.sum()
        s2[n] = dS @ dS

    return n_traj, s1, s2


def semiclassical_qfi_omega(
    S: float,
    omega: float,
    delta: float,
    gamma: float,
    times: np.ndarray,
    n_traj: int,
    seed: int = 42,
    sampling: str = "continuous",
    n_jobs: int = 1,
    state: str = "up",
    theta0: float = np.pi / 2,
    phi0: float = 0.0,
    spread: float = 0.0,
    progress: bool = False,
) -> dict:
    """Semiclassical QFI for sensing ω via the stochastic action derivative.

    Integrates the SDE ensemble and computes, for each trajectory k,

        ∂_ω S^(k)(T) = −∫₀ᵀ dt sₓ^(k)(t),

    then

        F_sc(T) = 4 · Var_k(∂_ω S^(k)(T)).

    The variance is accumulated *on the fly* from running sums, so memory scales
    as O(n_time) rather than O(n_time · n_traj): the full trajectory array is
    never stored.  Trajectories are independent, so the ensemble is split into
    ``n_jobs`` chunks integrated in parallel processes.

    Parameters
    ----------
    S        : total spin quantum number
    omega    : driving (Rabi) frequency  (semiclassical ω)
    delta    : detuning shift δ
    gamma    : semiclassical decay rate  (γ = κN = 2κS)
    times    : 1-D time array with uniform spacing
    n_traj   : number of stochastic trajectories
    seed     : random seed for reproducibility
    sampling : initial-condition scheme – ``"continuous"`` or ``"dtwa"``
    n_jobs   : number of worker processes.  ``1`` (default) runs serially and
               reproduces the legacy single-stream result bit-for-bit; ``> 1``
               splits the ensemble across that many processes; ``-1`` uses all
               available cores.  With ``n_jobs > 1`` each worker draws from an
               independent ``SeedSequence``-spawned RNG stream, so results are
               statistically equivalent to but not bitwise-identical to the
               serial run.
    state    : ``"up"``/``"down"`` initial polarisation for ``"dtwa"`` sampling
    theta0,
    phi0     : mean initial angles for ``"continuous"`` sampling
    spread   : Gaussian spread of the ``"continuous"`` initial condition
    progress : show a tqdm progress bar.  Serial (``n_jobs == 1``) tracks the
               time-step integration; parallel (``n_jobs > 1``) tracks the
               worker chunks as they complete.

    Returns
    -------
    dict with keys:
        ``'qfi'``    – F_sc(t) for each time step, shape (n_time,)
        ``'times'``  – the input time array
        ``'n_traj'`` – total number of trajectories actually integrated
    """
    times = np.asarray(times, dtype=float)
    n_time = len(times)
    if n_time < 2:
        raise ValueError("`times` must contain at least two points.")

    # Mirror the legacy parameter scaling exactly: the old code passed
    # gamma/(2S) as Γ to run_trajectories, which then stepped with Γ/(2S).
    gamma_step = (gamma / (2.0 * S))

    if n_jobs is None or n_jobs == 1:
        count, s1, s2 = _semiclassical_qfi_chunk(
            S, omega, delta, gamma_step, times, n_traj, seed,
            sampling, state, theta0, phi0, spread, progress,
        )
    else:
        if n_jobs < 0:
            n_jobs = os.cpu_count() or 1
        n_workers = min(n_jobs, n_traj)
        # Even split of the ensemble; independent reproducible RNG streams.
        counts = [
            n_traj // n_workers + (1 if i < n_traj % n_workers else 0)
            for i in range(n_workers)
        ]
        child_seeds = np.random.SeedSequence(seed).spawn(n_workers)

        s1 = np.zeros(n_time)
        s2 = np.zeros(n_time)
        count = 0
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = [
                ex.submit(
                    _semiclassical_qfi_chunk,
                    S, omega, delta, gamma_step, times, c, cs,
                    sampling, state, theta0, phi0, spread,
                )
                for c, cs in zip(counts, child_seeds)
            ]
            done_iter = as_completed(futures)
            if progress:
                from tqdm.auto import tqdm

                done_iter = tqdm(
                    done_iter, total=len(futures), desc="semiclassical QFI"
                )
            for fut in done_iter:
                c, p1, p2 = fut.result()
                count += c
                s1 += p1
                s2 += p2

    # Combined sample variance (ddof=1) over all trajectories at each time.
    mean = s1 / count
    qfi = 4.0 * (s2 - count * mean * mean) / (count - 1)

    return {"qfi": qfi, "times": times, "n_traj": count}


# ── Collective (BTC) semiclassical QFI ────────────────────────────────────────

def _ckpt_save(path: str, hdr: dict, n: int, k: int, s, dS, mx_prev,
               s1, s2, s3, s4, rng) -> None:
    """Atomically dump the integrator state so the loop can resume at step n+1.

    Written to a temp file then os.replace()d, so a kill mid-write leaves the
    previous checkpoint intact rather than a truncated one.  ``rng`` is stored
    as ``bit_generator.state``; restoring it continues the identical random
    stream, making a resumed run bit-identical to an uninterrupted one.
    """
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "wb") as fh:
        pickle.dump(
            {"hdr": hdr, "n": n, "k": k, "s": s, "dS": dS, "mx_prev": mx_prev,
             "s1": s1, "s2": s2, "s3": s3, "s4": s4,
             "rng": rng.bit_generator.state},
            fh, protocol=pickle.HIGHEST_PROTOCOL,
        )
    os.replace(tmp, path)


def _ckpt_load(path: str, hdr: dict):
    """Return the checkpoint dict if it exists and matches ``hdr``, else None.

    A header mismatch means the file belongs to a different run (different N,
    dt, grid, seed, ...), so it is ignored rather than silently resumed from --
    resuming across a parameter change would splice two different physics runs
    into one trajectory.
    """
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as fh:
            d = pickle.load(fh)
    except Exception:
        return None                      # truncated/corrupt -> start over
    return d if d.get("hdr") == hdr else None


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
    scheme: str = "drift",
    Gamma: np.ndarray | None = None,
    progress: bool = False,
    ckpt_path: str | None = None,
    ckpt_every: int = 0,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Integrate ``n_traj`` collective realisations, streaming QFI statistics.

    Each realisation is ``N`` collectively coupled spins evolved with the chosen
    ``scheme`` (see :func:`~btc.sde.make_collective_stepper`).  The action
    derivative uses the *normalised collective* spin mₓ^(k)(t) = (1/N) Σₙ sₓ^{n,k}(t),

        ∂_ω 𝒮^(k)(t) = −∫₀ᵗ mₓ^(k)(t') dt',

    and the power sums s_p[t] = Σ_k (∂_ω 𝒮^(k))^p for p = 1..4 are accumulated on
    the fly at the ``store_idx`` output points (``None`` = every step), so memory
    is O(n_store) + O(N · n_traj) — no trajectory history is stored.  The first
    two give the QFI (∝ sample variance); the 3rd/4th moments let the caller form
    its Monte-Carlo standard error.  ``seed`` is anything
    :func:`numpy.random.default_rng` accepts (an ``int`` or a ``SeedSequence``).
    """
    rng = np.random.default_rng(seed)
    S = N / 2.0

    G = None if scheme == "drift" else collective_noise_factor(N, kappa, Gamma)
    step = make_collective_stepper(scheme, omega, delta, kappa, G)

    theta, phi = _sample_collective_initial(
        N * n_traj, rng, sampling, state, theta0, phi0, spread, S
    )
    theta = theta.reshape(N, n_traj)
    phi = phi.reshape(N, n_traj)

    # Cartesian state (N, n_traj, 3) for the pole-safe rotation solver.
    s = np.stack(angles_to_spin(theta, phi), axis=-1)

    # Output length: full grid, or only the requested sub-sampled points.
    n_out = n_time if store_idx is None else len(store_idx)
    # Streaming power sums Σ X^p (p = 1..4) of the trajectory action derivative
    # X = ∂_ω 𝒮.  s1,s2 give the QFI (∝ Var X); s3,s4 give the 3rd/4th moments
    # needed for its Monte-Carlo standard error.  All are additive across workers.
    s1 = np.zeros(n_out)
    s2 = np.zeros(n_out)
    s3 = np.zeros(n_out)
    s4 = np.zeros(n_out)

    # ∂_ω 𝒮(0) = 0; mₓ = (1/N) Σₙ s_xⁿ for each realisation.
    mx_prev = s[..., 0].mean(axis=0)   # (n_traj,)
    dS = np.zeros(n_traj)           # running −∫ mₓ dt  (trapezoid)
    half_dt = 0.5 * dt

    # ---- Checkpoint identity -------------------------------------------
    # Everything that changes the trajectory. A checkpoint whose header does
    # not match is from a different run and is ignored (see _ckpt_load).
    hdr = {"N": N, "n_traj": n_traj, "n_time": n_time, "n_out": n_out,
           "dt": dt, "omega": omega, "delta": delta, "kappa": kappa,
           "scheme": scheme, "sampling": sampling, "state": state,
           "seed": repr(seed)}

    start, k = 1, 1
    ck = _ckpt_load(ckpt_path, hdr) if ckpt_path else None
    if ck is not None:
        # Overwrite the freshly sampled state (and the RNG position, which the
        # initial sampling above already advanced) with the saved one.
        s, dS, mx_prev = ck["s"], ck["dS"], ck["mx_prev"]
        s1, s2, s3, s4 = ck["s1"], ck["s2"], ck["s3"], ck["s4"]
        rng.bit_generator.state = ck["rng"]
        start, k = ck["n"] + 1, ck["k"]

    step_iter = range(start, n_time)
    if progress:
        from tqdm.auto import tqdm

        step_iter = tqdm(step_iter, desc=f"collective QFI ({scheme})",
                         initial=start - 1, total=n_time - 1)

    if store_idx is None:
        # Record every integration step (memory O(n_time)).
        for n in step_iter:
            s = step(s, dt, rng)
            mx = s[..., 0].mean(axis=0)
            dS -= half_dt * (mx_prev + mx)   # trapezoid increment of −∫ mₓ dt
            mx_prev = mx
            dS2 = dS * dS
            s1[n] = dS.sum()
            s2[n] = dS2.sum()
            s3[n] = (dS2 * dS).sum()
            s4[n] = (dS2 * dS2).sum()
            if ckpt_every and n % ckpt_every == 0:
                _ckpt_save(ckpt_path, hdr, n, k, s, dS, mx_prev,
                           s1, s2, s3, s4, rng)
    else:
        # Integrate at full dt but store only at the requested indices
        # (memory O(n_store)); store_idx[0] == 0 is the t=0 point, left at 0.
        next_store = store_idx[k] if k < n_out else -1
        for n in step_iter:
            s = step(s, dt, rng)
            mx = s[..., 0].mean(axis=0)
            dS -= half_dt * (mx_prev + mx)   # trapezoid increment of −∫ mₓ dt
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
            if ckpt_every and n % ckpt_every == 0:
                _ckpt_save(ckpt_path, hdr, n, k, s, dS, mx_prev,
                           s1, s2, s3, s4, rng)

    if ckpt_path and os.path.exists(ckpt_path):
        os.remove(ckpt_path)             # completed: drop the resume state

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
    scheme: str = "drift",
    Gamma: np.ndarray | None = None,
    n_store: int | None = None,
    batch_size: int = 25,
    progress: bool = False,
) -> dict:
    """Collective-BTC semiclassical QFI for sensing ω.

    The genuine boundary-time-crystal counterpart of
    :func:`semiclassical_qfi_omega`: it evolves the collective DCTWA ensemble of
    ``N`` coupled spins (collective decay √κ S₋, superradiant mean-field
    coupling) rather than an isolated spin, and computes

        F_sc(T) = 4 · Var_k(∂_ω 𝒮^(k)(T)),   ∂_ω 𝒮^(k) = −∫₀ᵀ mₓ^(k) dt,

    where mₓ^(k) = (1/N) Σₙ sₓ^{n,k} is the normalised collective spin of each
    realisation.  Because mₓ = Sₓ/S, this is comparable to the quantum QFI
    normalised by S²:

        F_Q(T) / S²  ≈  F_sc(T),       S = N/2.

    The estimator (the action-variance formula above) is identical for every
    ``scheme``; only the trajectory dynamics differ.

    Validity (compare against :func:`quantum_monitored_qfi_omega` / S², the
    *monitored* QFI — the proper quantum counterpart of the trajectory-action QFI,
    not the bare system QFI :func:`quantum_qfi_omega`):

    * ``scheme="drift"`` (default) — the drift-only mean-field scheme.  Matches in
      the closed / weak-decay limit (for all N; at N = 1 it reduces exactly to
      :func:`semiclassical_qfi_omega`) but *underestimates* the QFI as κt and N
      grow, because it drops the collective (off-diagonal) noise of 𝒟[S₋].
    * ``scheme="diffusive"`` — the pole-safe diffusion-completed scheme: it adds
      the positive-Γ collective noise (the missing second-moment ingredient), so
      it lifts the QFI toward the monitored value with a residual ∝ 1/S → 0.
      Best paired with ``sampling="wigner_cone"``.
    * ``scheme="diffusive_em"`` — the (θ,φ) Euler–Maruyama cross-check of the
      diffusive scheme; pole-prone, for validation only.

    The variance is accumulated on the fly (no time history stored).  The
    ``n_traj`` realisations are partitioned into fixed-size batches with
    deterministic per-batch seeds, so the result depends only on
    ``(seed, n_traj, batch_size)`` — identical whether run serially or across
    any number of ``n_jobs`` workers (and on any machine, for a given NumPy
    version).  ``n_jobs`` only controls how those batches are distributed.

    Parameters
    ----------
    N        : number of spins (collective spin S = N/2); positive integer
    omega    : driving (Rabi) frequency ω  (H = ω Sₓ)
    delta    : detuning shift δ
    kappa    : single-particle decay rate κ  (collective rate g = κS = κN/2)
    times    : 1-D time array with uniform spacing
    n_traj   : number of independent collective realisations
    seed     : random seed for reproducibility
    sampling : initial-condition scheme – ``"dtwa"`` (default) or ``"continuous"``
    n_jobs   : number of worker processes over which to distribute the batches.
               ``1`` (default) runs serially; ``> 1`` uses processes; ``-1`` uses
               all cores.  The choice does not affect the result — only the
               wall-time — because the RNG is keyed per batch, not per worker.
    state    : ``"up"`` / ``"down"`` initial polarisation for ``"dtwa"`` sampling
    theta0,
    phi0     : mean initial angles for ``"continuous"`` sampling
    spread   : Gaussian spread of the ``"continuous"`` initial condition
    n_store  : if given, integrate on the full ``times`` grid but return the QFI
               at only this many (evenly spaced, endpoints included) points.
               Decouples output size from the integration step count, so a fine
               ``dt`` / long ``T`` no longer needs O(n_time) memory per worker
               (nor pickles the full ``times`` array to each one).  ``None``
               (default) returns every step, as before.
    batch_size : trajectories per independently-seeded batch (default 25).  Sets
               the reproducible unit of work: ``ceil(n_traj / batch_size)``
               batches are distributed over ``n_jobs`` workers.  Smaller → finer
               parallel granularity (more cores usable); larger → less overhead.
               Changing it changes the specific realisations (but not the
               statistics); keep it fixed for reproducible output.
    progress : tqdm progress bar over the batches

    Returns
    -------
    dict with keys:
        ``'qfi'``        – F_sc(t) at the stored points, shape (n_store,) — the
                           full grid ``(n_time,)`` when ``n_store`` is ``None``
        ``'qfi_stderr'`` – Monte-Carlo standard error (±1σ) of ``'qfi'`` from the
                           finite ``n_traj``; same shape.  Scales as 1/√n_traj.
        ``'times'``      – the stored time points matching ``'qfi'``
        ``'n_traj'``     – total number of realisations actually integrated
        ``'N'``          – number of spins (S = N/2)
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
                sampling, state, theta0, phi0, spread, scheme, Gamma)

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
        # Trajectory-weighted progress bar over completing batches.  Worker
        # processes cannot share a live bar, so a single bar is advanced in the
        # parent as each batch finishes, incrementing by that batch's trajectory
        # count -> a smooth 0..n_traj read-out even across many parallel cores.
        # tqdm writes to stderr with carriage returns, which `tail -f` on a
        # SLURM log renders correctly for live cluster monitoring.
        pbar = None
        if progress:
            from tqdm.auto import tqdm

            pbar = tqdm(total=n_traj, desc=f"collective QFI ({scheme})",
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

    # Reduce the power sums in fixed batch order (deterministic FP summation),
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

    # Combined sample variance (ddof=1) over all realisations at each time.
    n = count
    mean = s1 / n
    qfi = 4.0 * (s2 - n * mean * mean) / (n - 1)

    # Monte-Carlo standard error of the QFI (= 4·Var[X]).  The error bar on a
    # variance estimate needs the 4th moment: for iid samples
    #   Var(S²) = (1/n)·(μ₄ − (n−3)/(n−1)·μ₂²),
    # (→ the Gaussian 2·μ₂²/n when μ₄ = 3·μ₂²).  μ₂, μ₄ are the sample central
    # moments from the power sums; clip tiny negatives from finite-sample noise.
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


# ── Batchwise collective QFI (checkpointed / distributed accumulation) ─────────

def qfi_from_powersums(
    count: int,
    s1: np.ndarray,
    s2: np.ndarray,
    s3: np.ndarray,
    s4: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce accumulated power sums of X = ∂_ω 𝒮 to the QFI and its stderr.

    ``s_p = Σ_k X_k^p`` (p = 1..4) summed over ``count`` trajectories, at each
    stored time point.  Because these sums are additive across independent
    trajectories, partial sums from any number of separately-run batches / tasks
    can simply be added before calling this — the result is identical (to
    floating-point roundoff) to a single run over all ``count`` trajectories.

    Returns ``(qfi, qfi_stderr)`` — the same estimator and Monte-Carlo error bar
    as :func:`collective_qfi_omega` (see its notes for the moment formulae).
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
    scheme: str = "diffusive",
    sampling: str = "wigner_cone",
    state: str = "down",
    theta0: float = np.pi / 2,
    phi0: float = 0.0,
    spread: float = 0.0,
    Gamma: np.ndarray | None = None,
    n_store: int | None = None,
    n_jobs: int = 1,
    progress: bool = False,
    ckpt_dir: str | None = None,
    ckpt_every: int = 0,
) -> dict:
    """Accumulate the QFI *power sums* over a slice of globally-indexed batches.

    This is the checkpointable building block behind the distributed / dynamic
    BTC runs: instead of returning the reduced QFI, it returns the additive
    power sums ``s1..s4`` (and the trajectory ``count``) for the batches with
    **global** indices ``batch_start .. batch_start + n_batches - 1``.  Every
    batch is ``batch_size`` trajectories seeded by
    ``SeedSequence(master_seed, spawn_key=(global_index,))``, so:

    * batches are reproducible from their global index alone — a task can compute
      any contiguous slice without touching the others;
    * disjoint slices never share trajectories (distinct spawn keys), so their
      power sums can be summed to reconstruct exactly the run over their union
      (use :func:`qfi_from_powersums` on the total);
    * appending more batches later (higher global indices) only adds statistics.

    The dynamics/estimator are identical to :func:`collective_qfi_omega`; only
    the accounting differs (raw moments out, no final variance reduction).

    Parameters
    ----------
    N, omega, delta, kappa, times : as in :func:`collective_qfi_omega`
    batch_start : index of the first global batch this call computes
    n_batches   : number of consecutive batches to compute
    batch_size  : trajectories per batch (must match across the whole campaign)
    master_seed : campaign-wide seed; global batch index supplies independence
    scheme, sampling, state, theta0, phi0, spread, Gamma : as in
                  :func:`collective_qfi_omega`
    n_store     : keep only this many evenly-spaced time points (endpoints
                  included); ``None`` keeps every step
    n_jobs      : worker processes over which to spread this slice's batches
    progress    : trajectory-weighted tqdm bar over the slice's batches

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

    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)

    def _ckpt_path(g):
        if not ckpt_dir:
            return None
        return os.path.join(
            ckpt_dir,
            f"ckpt_N{N}_seed{master_seed}_bs{batch_size}_g{g:07d}.pkl",
        )

    def _args(g):
        cs = np.random.SeedSequence(master_seed, spawn_key=(g,))
        return (N, omega, delta, kappa, dt, n_time, store_idx,
                batch_size, cs, sampling, state, theta0, phi0, spread, scheme,
                Gamma, False, _ckpt_path(g), ckpt_every)

    # A finished batch is persisted immediately, so a task that is killed keeps
    # every batch that had completed.  Without this, mid-trajectory checkpoints
    # alone are not enough: a repeatedly-killed task would redo finished batches
    # from zero each time and could fail to make progress at all.
    def _done_path(g):
        if not ckpt_dir:
            return None
        return os.path.join(
            ckpt_dir,
            f"done_N{N}_seed{master_seed}_bs{batch_size}_g{g:07d}.pkl",
        )

    def _done_hdr(g):
        return {"N": N, "n_traj": batch_size, "n_time": n_time, "n_out": n_out,
                "dt": dt, "omega": omega, "delta": delta, "kappa": kappa,
                "scheme": scheme, "sampling": sampling, "state": state,
                "seed": repr(np.random.SeedSequence(master_seed, spawn_key=(g,)))}

    def _done_save(g, res):
        p = _done_path(g)
        if not p:
            return
        tmp = f"{p}.tmp{os.getpid()}"
        with open(tmp, "wb") as fh:
            pickle.dump({"hdr": _done_hdr(g), "res": res}, fh,
                        protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, p)

    # parts keyed by global index so the reduction order is canonical (→ the
    # combined result is independent of worker/scheduling order).
    parts: dict[int, tuple] = {}
    pending = []
    for g in globals_:
        d = _ckpt_load(_done_path(g), _done_hdr(g)) if ckpt_dir else None
        if d is not None:
            parts[g] = d["res"]
        else:
            pending.append(g)

    if pending and (n_jobs is None or n_jobs == 1):
        g_iter = pending
        if progress:
            from tqdm.auto import tqdm

            g_iter = tqdm(g_iter, total=len(pending), desc=f"powersums N={N}",
                          unit="batch", dynamic_ncols=True)
        for g in g_iter:
            parts[g] = _collective_qfi_chunk(*_args(g))
            _done_save(g, parts[g])
    elif pending:
        if n_jobs < 0:
            n_jobs = os.cpu_count() or 1
        n_workers = min(n_jobs, len(pending))
        pbar = None
        if progress:
            from tqdm.auto import tqdm

            pbar = tqdm(total=len(pending) * batch_size, desc=f"powersums N={N}",
                        unit="traj", dynamic_ncols=True)
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(_collective_qfi_chunk, *_args(g)): g
                       for g in pending}
            for fut in as_completed(futures):
                g = futures[fut]
                parts[g] = fut.result()
                _done_save(g, parts[g])
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

    # Slice reduced successfully -> the per-batch resume state is spent.  Left
    # in place it would silently resurrect these results on a deliberate rerun.
    if ckpt_dir:
        for g in globals_:
            for p in (_done_path(g), _ckpt_path(g)):
                if p and os.path.exists(p):
                    os.remove(p)

    return {
        "count": count, "s1": s1, "s2": s2, "s3": s3, "s4": s4,
        "times": store_times, "batch_start": batch_start, "n_batches": n_batches,
        "batch_size": batch_size, "master_seed": master_seed,
    }


# ── Quantum QFI helpers ───────────────────────────────────────────────────────

def _build_liouvillian_matrix(
    H_mat: sp.spmatrix,
    Sm_mat: sp.spmatrix,
    kappa: float,
) -> sp.csr_matrix:
    """Liouvillian superoperator as a sparse (dim², dim²) matrix.

    Uses the column-vectorisation convention  vec(AXB) = (Bᵀ ⊗ A) vec(X),
    i.e. vec stacks the *columns* of a matrix.

    The resulting matrix L satisfies

        d/dt vec(ρ) = L · vec(ρ)

    for the Lindblad equation with H and a single collapse operator √κ S₋.

    H_mat and Sm_mat are collective spin operators, which are banded
    (O(dim) nonzeros) in the Dicke basis; keeping them and L sparse cuts the
    per-step matvec cost in the ODE solver from O(dim⁴) to O(dim²) — the
    dominant cost at large S.

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
    Sp_mat = Sm_mat.conj().T.tocsr()    # S₊ = S₋†
    SpSm = (Sp_mat @ Sm_mat).tocsr()    # S₊S₋  (Hermitian)

    # −i[H, ρ]:  vec([H,ρ]) = (I⊗H − Hᵀ⊗I) vec(ρ)
    L = -1j * (sp.kron(I, H_mat, format="csr") - sp.kron(H_mat.T, I, format="csr"))

    # κ D[S₋](ρ):  S₋ ρ S₊ → (S₊ᵀ⊗S₋) vec(ρ) = (S₋*⊗S₋) vec(ρ)
    #              −½ S₊S₋ ρ → −½(I⊗S₊S₋) vec(ρ)
    #              −½ ρ S₊S₋ → −½((S₊S₋)ᵀ⊗I) vec(ρ) = −½((S₊S₋)*⊗I) vec(ρ)
    # (SpSm Hermitian ⟹ (SpSm)ᵀ = (SpSm)*)
    L = L + kappa * (
        sp.kron(Sm_mat.conj(), Sm_mat, format="csr")
        - 0.5 * sp.kron(I, SpSm, format="csr")
        - 0.5 * sp.kron(SpSm.conj(), I, format="csr")
    )
    return L.tocsr()


def _uhlmann_fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Uhlmann fidelity  F(ρ, σ) = ( Tr √(√ρ σ √ρ) )².

    Parameters
    ----------
    rho   : (dim, dim) density matrix
    sigma : (dim, dim) density matrix

    Returns
    -------
    F : real float in [0, 1]
    """
    # √ρ via eigendecomposition (ρ is Hermitian PSD)
    evals_rho, evecs_rho = np.linalg.eigh(rho)
    evals_rho = np.maximum(evals_rho.real, 0.0)
    sqrt_rho = evecs_rho @ np.diag(np.sqrt(evals_rho)) @ evecs_rho.conj().T

    # M = √ρ σ √ρ  (Hermitian PSD)
    M = sqrt_rho @ sigma @ sqrt_rho

    # Tr√M = Σᵢ √λᵢ  where λᵢ are eigenvalues of M
    evals_M = np.linalg.eigvalsh(M)
    evals_M = np.maximum(evals_M.real, 0.0)
    return float(np.sum(np.sqrt(evals_M)) ** 2)


# ── Quantum QFI (full) ────────────────────────────────────────────────────────

def _expm_propagate(
    L: sp.spmatrix,
    y0: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """Propagate y(t) = expm(L·(t − times[0])) y0 at each requested time.

    L is a constant (time-independent) generator, so this is an exact
    linear-algebra problem, not an ODE to be integrated approximately: the
    Krylov-subspace matrix-exponential action (:func:`expm_multiply`) gives
    machine-precision accuracy without the step-size penalty an adaptive RK
    stepper pays on an oscillatory/stiff generator (see
    :func:`quantum_monitored_qfi_omega`). For a uniform grid this uses
    :func:`expm_multiply`'s multi-point trajectory mode, which shares work
    across all output times; a non-uniform grid falls back to one exact
    exponential step per interval.

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
    """Propagate the vectorised Lindblad equation and return ρ(t) at each time.

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
    y0 = rho0.flatten(order="F")  # column-vec
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
    """Quantum QFI for sensing ω via the fidelity formula.

    Evolves the Lindblad master equation at ω and ω + δ and estimates

        F_Q(T) = 8 · (1 − F(ρ_ω(T), ρ_{ω+δ}(T))) / δ²

    where F is the Uhlmann fidelity and the Hamiltonian is H = ω Sₓ with
    a single collapse operator √κ S₋.

    Parameters
    ----------
    S     : total spin quantum number
    omega : Rabi frequency  (H = omega * Sx)
    kappa : single-particle decay rate
    times : 1-D time array (need not be uniform)
    delta : finite-difference step for ω  (default 1e-4)
    state : initial coherent-state Bloch direction (name or length-3 vector),
            matching the DTWA sampler; default ``"up"``.

    Returns
    -------
    dict with keys:
        ``'qfi'``   – F_Q(T) at each time, shape (n_time,)
        ``'times'`` – the input time array

    Notes
    -----
    * The quantum QFI scales as S² relative to the semiclassical QFI because
      the semiclassical spin is normalised: sₓ = Sₓ / S.  Compare via
      F_Q / S².
    * The Liouvillian is built as a sparse (dim², dim²) matrix (Sx, S₋ are
      banded in the Dicke basis) and propagated via Krylov-subspace
      matrix-exponential action (:func:`_expm_propagate`) rather than an
      adaptive ODE stepper — exact for this linear, time-independent
      generator, at roughly half the cost of the (already sparse) DOP853
      version. No solver tolerance to choose; accuracy is machine precision.
    * Choose δ small enough that the fidelity approximation is valid but
      large enough to avoid numerical cancellation.  δ ~ 10⁻⁴–10⁻³ works
      well for typical BTC parameters.
    """
    from .operators import make_operators, make_initial_state  # lazy: needs QuTiP

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


# ── Gammelmark–Mølmer monitored (continuous-measurement) QFI ──────────────────

def quantum_monitored_qfi_omega(
    S: float,
    omega: float,
    kappa: float,
    times: np.ndarray,
    state="up",
) -> dict:
    r"""Gammelmark–Mølmer QFI for sensing ω under continuous monitoring.

    This is the *ultimate* QFI of a continuously monitored open system,
    i.e. the Fisher information available to an observer who has access to
    the **entire output field** in addition to a final measurement of the
    system [Gammelmark & Mølmer, PRL 112, 170401 (2014)].  It upper-bounds
    the system-state SLD QFI returned by :func:`quantum_qfi_omega` (which only
    uses the reduced state ρ(T)), and it is the proper quantum counterpart to
    the semiclassical trajectory-action QFI :func:`semiclassical_qfi_omega`.

    Construction
    ------------
    The system+output global state |Ψ_ω(T)⟩ is pure (the initial system state
    must be pure).  Its overlap across two parameter values,
    O(ω, ω′) = ⟨Ψ_ω(T) | Ψ_{ω′}(T)⟩ = Tr M(T), is generated by the two-sided
    ("bra at ω, ket at ω′") map

        d/dt M = −i(H_{ω′} M − M H_ω)
                 + κ ( S₋ M S₊ − ½ S₊S₋ M − ½ M S₊S₋ ),

    which reduces to the ordinary Lindbladian at ω′ = ω (so Tr M = Tr ρ = 1).
    Expanding M(ω, ω+ε) = M⁽⁰⁾ + ε M⁽¹⁾ + ½ε² M⁽²⁾ and using the pure-state
    relation |O(ε)|² = 1 − ¼ F_Q ε² gives the closed form

        F_Q(T) = 4 ( −Re Tr M⁽²⁾(T) − |Tr M⁽¹⁾(T)|² ),

    with the derivative states obeying (here H = ω Sₓ, so ∂_ω H = Sₓ and the
    collapse operator √κ S₋ is ω-independent)

        d/dt M⁽⁰⁾ = ℒ M⁽⁰⁾,                     M⁽⁰⁾(0) = ρ₀
        d/dt M⁽¹⁾ = ℒ M⁽¹⁾ − i Sₓ M⁽⁰⁾,         M⁽¹⁾(0) = 0
        d/dt M⁽²⁾ = ℒ M⁽²⁾ − 2i Sₓ M⁽¹⁾,        M⁽²⁾(0) = 0,

    where ℒ is the ordinary Lindbladian.  No finite differences are required.

    Parameters
    ----------
    S     : total spin quantum number
    omega : Rabi frequency  (H = omega * Sx)
    kappa : single-particle decay rate
    times : 1-D time array (need not be uniform)

    Returns
    -------
    dict with keys:
        ``'qfi'``   – F_Q^monitored(t) at each time, shape (n_time,)
        ``'times'`` – the input time array

    Notes
    -----
    * Like the system QFI, this scales as S² relative to the (normalised)
      semiclassical QFI; compare via F_Q / S².
    * In the closed limit κ → 0 there is no output field and this reduces to
      the unitary QFI 4·Var(∫ Sₓ dt).
    * The initial state must be pure (a coherent state), as the construction
      requires; ``state`` selects its Bloch direction (default |S,+S⟩).
    * L and X_left are sparse (dim², dim²) matrices (Sx, S₋ are banded in
      the Dicke basis). The (M⁽⁰⁾, M⁽¹⁾, M⁽²⁾) hierarchy is a single linear,
      time-independent system, so it is propagated as one Krylov-subspace
      matrix-exponential action (:func:`_expm_propagate`) on the stacked
      block generator [[L,0,0],[-iX,L,0],[0,-2iX,L]], rather than stepped
      with an adaptive RK integrator — exact to machine precision, no
      tolerance to tune, and ~2.5-2.8x faster than the (already sparse)
      DOP853 version. Benchmarked at ratio=Ω/κS=2, κT=100: S=24 (dim=49)
      ~1 min, S=32 (dim=65) ~3 min on a standard workstation. The dominant
      remaining cost driver is the Krylov subspace size, which grows with
      the Hamiltonian bandwidth Ω·S, not matrix size.
    """
    from .operators import make_operators, make_initial_state  # lazy: needs QuTiP

    ops = make_operators(S)
    Sm_mat = sp.csr_matrix(ops["Sm"].full().astype(complex))
    Sx_mat = sp.csr_matrix(ops["Sx"].full().astype(complex))

    dim = Sx_mat.shape[0]
    I = sp.identity(dim, dtype=complex, format="csr")

    # Ordinary Lindbladian ℒ and the ket-side generator ∂_ω H = Sx (left mult.)
    L_mat = _build_liouvillian_matrix(omega * Sx_mat, Sm_mat, kappa)
    X_left = sp.kron(I, Sx_mat, format="csr")   # vec(Sx · M) = X_left · vec(M)

    rho0 = make_initial_state(S, state).full().astype(complex)
    d2 = dim * dim
    y0 = np.zeros(3 * d2, dtype=complex)
    y0[:d2] = rho0.flatten(order="F")      # M⁽⁰⁾(0) = ρ₀, others zero

    # Stack (M⁽⁰⁾, M⁽¹⁾, M⁽²⁾) into one vector propagated by a single
    # time-independent block generator (the recursion above), so the whole
    # hierarchy is one Krylov matrix-exponential action.
    Lbig = sp.bmat(
        [[L_mat, None, None],
         [-1j * X_left, L_mat, None],
         [None, -2j * X_left, L_mat]],
        format="csr",
    )
    y_t = _expm_propagate(Lbig, y0, times)   # (n_time, 3 d2)

    # Tr M = vec(I) · vec(M); the col-major vec of I_dim has 1's at stride dim+1.
    vec_I = np.zeros(d2, dtype=complex)
    vec_I[:: dim + 1] = 1.0
    m1_t = y_t[:, d2:2 * d2].T             # (d², n_time)
    m2_t = y_t[:, 2 * d2:].T
    tr_m1 = vec_I @ m1_t                   # (n_time,)
    tr_m2 = vec_I @ m2_t

    qfi_vals = 4.0 * (-tr_m2.real - np.abs(tr_m1) ** 2)
    return {"qfi": qfi_vals, "times": times}


# ── Gammelmark–Mølmer monitored QFI via MCWF (quantum-jump trajectories) ───────

def _mcwf_monitored_qfi(
    H: np.ndarray,
    c_ops: list[np.ndarray],
    dH: np.ndarray,
    psi0: np.ndarray,
    times: np.ndarray,
    n_traj: int,
    rng: np.random.Generator,
    progress: bool = False,
) -> tuple[int, np.ndarray, np.ndarray]:
    r"""Monte-Carlo-wavefunction estimate of the monitored QFI (one chunk).

    Samples ``n_traj`` quantum-jump trajectories of the unravelled master equation
    and, *along each record*, propagates the parameter derivative of the
    (unnormalised) conditional state.  For a record ``r`` with conditional pure
    state ψ_r(ω) and probability density p(r|ω), the Gammelmark–Mølmer monitored
    QFI is the Fisher information of the full record plus a final optimal
    measurement,

        F_mon = E_r[ (∂_ω ln p)² ] + E_r[ F_Q^pure(ψ_r) ]
              = 4 · E_r[ ⟨ξ|ξ⟩ − (Im⟨ψ|ξ⟩)² ],

    with ψ the normalised conditional state and ξ = ∂_ω ψ̃ / ‖ψ̃‖ the consistently
    rescaled derivative of the *unnormalised* state ψ̃.  Because that combination is
    invariant under a common rescaling of (ψ̃, ∂_ω ψ̃), the trajectory keeps ψ
    normalised at every step while scaling ξ by the same factor.  Sampling records
    ~ p makes the trajectory mean an unbiased estimate of F_mon.

    The collapse operators are assumed ω-independent (true for BTC / radiative
    decay), so a jump sends ξ → c_k ξ.  Unlike the semiclassical action-variance
    estimator (where the QFI itself is a sample variance), F_mon here is a plain
    sample *mean* of the per-trajectory integrand g_r = 4(⟨ξ|ξ⟩ − (Im⟨ψ|ξ⟩)²), so
    its Monte-Carlo error only needs the first two moments.  Returns
    ``(n_traj, s1, s2)`` — the per-time power sums Σ_r g_r and Σ_r g_r², additive
    across independently-seeded chunks/workers.  The caller forms the mean
    (F_mon = s1/n) and its standard error (√[(s2/n − mean²)/n]) after combining.
    """
    d = H.shape[0]
    A = -1j * H
    for c in c_ops:
        A = A - 0.5 * (c.conj().T @ c)            # A = -i H_eff
    Udet = _expm(A * (float(times[1] - times[0])))

    # Bottom-left block of expm([[A,0],[B,A]] dt) gives the derivative source,
    # with B = -i ∂_ω H (the ket-side generator).
    dt = float(times[1] - times[0])
    Mblk = np.zeros((2 * d, 2 * d), dtype=complex)
    Mblk[:d, :d] = A
    Mblk[d:, d:] = A
    Mblk[d:, :d] = -1j * dH
    Ublk = _expm(Mblk * dt)
    Ssrc = Ublk[d:, :d]                           # source block coupling ψ̃ → ξ

    n_time = len(times)
    psi = np.tile(psi0.astype(complex)[:, None], (1, n_traj))
    psi /= np.linalg.norm(psi, axis=0, keepdims=True)
    xi = np.zeros((d, n_traj), dtype=complex)     # ∂_ω ψ̃ / ‖ψ̃‖ ; zero at t=0

    s1 = np.zeros(n_time)                          # Σ_r g_r(t)    -> mean = s1/n_traj
    s2 = np.zeros(n_time)                          # Σ_r g_r(t)²   -> Var  = s2/n_traj - mean²
    # s1[0] = s2[0] = 0 (ψ ω-independent at t=0, so g=0 for every trajectory)

    step_iter = range(1, n_time)
    if progress:
        from tqdm.auto import tqdm

        step_iter = tqdm(step_iter, desc="MCWF monitored QFI")

    for n in step_iter:
        # No-jump candidate (deterministic, non-unitary) for ψ and ξ.
        psi_nj = Udet @ psi
        xi_nj = Ssrc @ psi + Udet @ xi
        nrm2 = np.einsum("ij,ij->j", psi_nj.conj(), psi_nj).real
        u = rng.random(n_traj)
        no_jump = nrm2 >= u

        # No-jump columns: renormalise ψ̃ and rescale ξ by the same factor.
        inv = 1.0 / np.sqrt(nrm2)
        psi_new = psi_nj * inv
        xi_new = xi_nj * inv

        # Jump columns: pick a channel ∝ ‖c_k ψ‖², apply c_k to ψ and ξ, renormalise.
        jcols = np.where(~no_jump)[0]
        if jcols.size:
            psi_j = psi[:, jcols]
            xi_j = xi[:, jcols]
            if len(c_ops) == 1:
                k_of = np.zeros(jcols.size, dtype=int)
            else:
                w = np.stack(
                    [np.einsum("ij,ij->j", (c @ psi_j).conj(), (c @ psi_j)).real
                     for c in c_ops]
                )                                 # (K, |jcols|)
                cw = np.cumsum(w, axis=0)
                pick = rng.random(jcols.size) * cw[-1]
                k_of = (pick[None, :] > cw).sum(axis=0)
            out_psi = np.empty_like(psi_j)
            out_xi = np.empty_like(xi_j)
            for k, c in enumerate(c_ops):
                sel = k_of == k
                if not sel.any():
                    continue
                cp = c @ psi_j[:, sel]
                cx = c @ xi_j[:, sel]
                nn = 1.0 / np.linalg.norm(cp, axis=0, keepdims=True)
                out_psi[:, sel] = cp * nn
                out_xi[:, sel] = cx * nn
            psi_new[:, jcols] = out_psi
            xi_new[:, jcols] = out_xi

        psi, xi = psi_new, xi_new

        # Per-record monitored-QFI integrand g = 4(⟨ξ|ξ⟩ − (Im⟨ψ|ξ⟩)²).
        xixi = np.einsum("ij,ij->j", xi.conj(), xi).real
        psixi = np.einsum("ij,ij->j", psi.conj(), xi)
        g_traj = 4.0 * (xixi - psixi.imag ** 2)    # per-trajectory, shape (n_traj,)
        s1[n] = g_traj.sum()
        s2[n] = (g_traj ** 2).sum()

    return n_traj, s1, s2


def _expm(A: np.ndarray) -> np.ndarray:
    """Dense matrix exponential (scipy if available, else eigenvalue fallback)."""
    from scipy.linalg import expm as _se
    return _se(A)


def quantum_monitored_qfi_omega_mcwf(
    S: float,
    omega: float,
    kappa: float,
    times: np.ndarray,
    n_traj: int = 2000,
    seed: int = 42,
    state="up",
    n_jobs: int = 1,
    progress: bool = False,
) -> dict:
    r"""MCWF (quantum-jump) estimate of the Gammelmark–Mølmer monitored QFI.

    A trajectory-based alternative to the deterministic
    :func:`quantum_monitored_qfi_omega`, giving the *same* monitored QFI but by
    sampling quantum-jump records instead of propagating the (dim²)-sized two-sided
    map.  It is the forward-looking option for spatially extended systems, where
    the density-matrix / two-sided-map cost (∝ 4ᴺ for distinguishable atoms) is
    prohibitive and only statevector MCWF (∝ 2ᴺ) is feasible — there one swaps the
    permutation-symmetric collective operators here for the per-atom ones.

    Model (BTC): H = ω Sₓ, single collapse √κ Ŝ₋, ∂_ω H = Sₓ.  Compare to the
    semiclassical action-variance QFI via F_mon / S² ≈ F_sc.

    Parameters
    ----------
    S       : total spin (S = N/2)
    omega   : Rabi frequency
    kappa   : single-particle decay rate
    times   : 1-D *uniform* time grid (fixed-step MCWF; use a fine dt)
    n_traj  : number of quantum-jump trajectories
    seed    : RNG seed
    state   : initial coherent-state Bloch direction (pure state required)
    n_jobs  : worker processes (``1`` serial; ``>1`` splits trajectories; ``-1`` all)
    progress: tqdm bar (time steps if serial, worker chunks if parallel)

    Returns
    -------
    dict with keys ``'qfi'`` (F_mon(t), shape (n_time,)), ``'qfi_stderr'``
    (Monte-Carlo standard error of ``'qfi'`` from the finite ``n_traj``, same
    shape), ``'times'``, ``'n_traj'`` (trajectories actually run).

    Notes
    -----
    * Fixed-step MCWF carries an O(dt) bias; verify convergence by halving dt.
      The Monte-Carlo error on F_mon (``'qfi_stderr'``) falls as 1/√n_traj.
    * In the closed limit κ → 0 this reduces to the unitary QFI 4·Var(∫ Sₓ dt).
    """
    times = np.asarray(times, dtype=float)
    if len(times) < 2:
        raise ValueError("`times` must contain at least two points.")
    from .operators import make_operators, make_initial_state  # lazy: needs QuTiP

    ops = make_operators(S)
    H = (omega * ops["Sx"]).full().astype(complex)
    dH = ops["Sx"].full().astype(complex)
    c_ops = [np.sqrt(kappa) * ops["Sm"].full().astype(complex)]
    psi0 = make_initial_state(S, state)
    # pure coherent state → take the dominant eigenvector of ρ₀
    evals, evecs = np.linalg.eigh(psi0.full())
    psi0 = evecs[:, int(np.argmax(evals.real))]

    def run_chunk(c, sd):
        return _mcwf_monitored_qfi(
            H, c_ops, dH, psi0, times, c, np.random.default_rng(sd), progress
        )

    n_time = len(times)
    if n_jobs is None or n_jobs == 1:
        count, s1, s2 = run_chunk(n_traj, seed)
    else:
        if n_jobs < 0:
            n_jobs = os.cpu_count() or 1
        n_workers = min(n_jobs, n_traj)
        counts = [
            n_traj // n_workers + (1 if i < n_traj % n_workers else 0)
            for i in range(n_workers)
        ]
        child_seeds = np.random.SeedSequence(seed).spawn(n_workers)
        s1 = np.zeros(n_time)
        s2 = np.zeros(n_time)
        count = 0
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = [
                ex.submit(
                    _mcwf_monitored_qfi, H, c_ops, dH, psi0, times, c,
                    np.random.default_rng(cs), False,
                )
                for c, cs in zip(counts, child_seeds)
            ]
            done_iter = as_completed(futures)
            if progress:
                from tqdm.auto import tqdm
                done_iter = tqdm(done_iter, total=len(futures), desc="MCWF monitored QFI")
            for fut in done_iter:
                c, s1c, s2c = fut.result()
                s1 += s1c            # power sums are additive across independent chunks
                s2 += s2c
                count += c

    qfi = s1 / count
    # Population variance of the per-trajectory integrand g; clip guards against
    # tiny negative values from floating-point cancellation when Var(g) ~ 0.
    var_g = np.clip(s2 / count - qfi ** 2, 0.0, None)
    qfi_stderr = np.sqrt(var_g / count)              # standard error of the mean

    return {"qfi": qfi, "qfi_stderr": qfi_stderr, "times": times, "n_traj": count}


def plot_qfi_comparison(
    times: np.ndarray,
    qfi_quantum: np.ndarray,
    qfi_semiclassical: np.ndarray,
    S: float,
):
    """Plot quantum and semiclassical QFI vs time on the same graph and save the figure as pdf."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 5))
    plt.plot(times, qfi_quantum / S**2, label="Quantum QFI / S²", color="blue")
    plt.plot(times, qfi_semiclassical, label="Semiclassical QFI", color="red", linestyle="--")
    plt.xlabel("Time")
    plt.ylabel("QFI (normalized)")
    plt.title(f"Quantum vs Semiclassical QFI for S={S}")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
    fig.savefig(f"qfi_comparison_S_{S}.pdf", dpi=300)