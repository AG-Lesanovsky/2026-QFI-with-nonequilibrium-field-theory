"""Semiclassical approximation of collective spin dynamics via stochastic 
differential euqations obtained through the truncated Wigner approximation (TWA).
"""

from __future__ import annotations

import numpy as np

SQRT3 = np.sqrt(3.0)


# -- Symbol to observable map -------------------------------------------------- 

def angles_to_spin(
    theta: np.ndarray,
    phi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weyl symbols (s_x, s_y, s_z) of the Pauli operators on the sqrt(3)-sphere.
    """
    sin_t = np.sin(theta)
    sx = SQRT3 * sin_t * np.cos(phi)
    sy = SQRT3 * sin_t * np.sin(phi)
    sz = -SQRT3 * np.cos(theta)
    return sx, sy, sz


def spin_to_angles(
    sx: np.ndarray,
    sy: np.ndarray,
    sz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`angles_to_spin`: (s_x, s_y, s_z) → (theta, phi).
    """
    theta = np.arccos(np.clip(-np.asarray(sz) / SQRT3, -1.0, 1.0))
    phi = np.arctan2(sy, sx)
    return theta, phi


# -- Rotation primitives -------------------------------------------------------
#
# Integrating the Bloch vector s = (s_x, s_y, s_z)

R2 = 3.0  # squared radius of the Weyl sphere


def _rotate_axis(s: np.ndarray, axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Rodrigues rotation of vectors ``s`` about unit ``axis`` by ``angle``.
    """
    c = np.cos(angle)[..., None]
    sn = np.sin(angle)[..., None]
    nds = np.sum(axis * s, axis=-1)[..., None]
    nxs = np.cross(axis, s)
    return c * s + sn * nxs + (1.0 - c) * nds * axis


def _rotate_x(s: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Rotate vectors ``s`` about x-axis by ``angle`` (leaves s_x fixed).
    """
    c = np.cos(angle)
    sn = np.sin(angle)
    sy = c * s[..., 1] - sn * s[..., 2]
    sz = sn * s[..., 1] + c * s[..., 2]
    return np.stack([s[..., 0], sy, sz], axis=-1)


def _rotate_y(s: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Rotate vectors ``s`` about y-axis by ``angle`` (leaves s_y fixed).
    """
    c = np.cos(angle)
    sn = np.sin(angle)
    sx = c * s[..., 0] + sn * s[..., 2]
    sz = -sn * s[..., 0] + c * s[..., 2]
    return np.stack([sx, s[..., 1], sz], axis=-1)


def _half_rotation(s: np.ndarray, f: np.ndarray, half_dt: float) -> np.ndarray:
    """Half deterministic rotation about w = (s x f)/r^2 by angle |w|*half_dt.
    """
    w = np.cross(s, f) / R2
    wn = np.linalg.norm(w, axis=-1)
    wn_safe = np.where(wn > 0.0, wn, 1.0)
    axis = w / wn_safe[..., None]
    return _rotate_axis(s, axis, half_dt * wn)


# -- Initial conditions --------------------------------------------------------

def sample_initial_conditions(
    n_traj: int,
    rng: np.random.Generator,
    theta0: float = np.pi / 2,
    phi0: float = 0.0,
    spread: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Initial (theta, phi) for n_traj trajectories.

    Parameters
    ----------
    n_traj  : number of trajectories
    rng     : NumPy random generator
    theta0  : mean initial polar angle; default pi/2
    phi0    : mean initial azimuthal angle; default 0
    spread  : standard deviation of the initial Gaussian noise

    Returns
    -------
    (theta, phi) : arrays of shape (n_traj,)
    """
    theta = np.full(n_traj, float(theta0))
    phi = np.full(n_traj, float(phi0))
    if spread > 0.0:
        theta = theta + rng.normal(0.0, spread, size=n_traj)
        phi = phi + rng.normal(0.0, spread, size=n_traj)
        np.clip(theta, 1e-6, np.pi - 1e-6, out=theta)
    return theta, phi


# Named Bloch directions n̂ = (⟨σ_x⟩, ⟨σ_y⟩, ⟨σ_z⟩).
_DTWA_DIRECTIONS = {
    "up": (0.0, 0.0, 1.0),
    "down": (0.0, 0.0, -1.0),
    "+x": (1.0, 0.0, 0.0),
    "-x": (-1.0, 0.0, 0.0),
    "+y": (0.0, 1.0, 0.0),
    "-y": (0.0, -1.0, 0.0),
}


def bloch_direction(state) -> np.ndarray:
    """Resolve a coherent-state spec to a unit Bloch vector.

    state is either a name ("up", "down", "+x", "-x", "+y", "-y") or a length-3 
    array-like Bloch vector (normalised here).
    Shared by the DTWA sampler and the quantum :func:`~btc.operators.make_initial_state`
    so both use the same initial-state convention.
    """
    if isinstance(state, str):
        if state not in _DTWA_DIRECTIONS:
            raise ValueError(
                f"Unknown state '{state}'. Choose one of "
                f"{list(_DTWA_DIRECTIONS)} or pass a length-3 Bloch vector."
            )
        return np.array(_DTWA_DIRECTIONS[state], dtype=float)
    n_hat = np.asarray(state, dtype=float)
    if n_hat.shape != (3,):
        raise ValueError("`state` vector must have shape (3,).")
    norm = np.linalg.norm(n_hat)
    if norm == 0.0:
        raise ValueError("`state` Bloch vector must be non-zero.")
    return n_hat / norm


def _transverse_frame(n_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two orthonormal vectors (e₁, e₂) spanning the plane orthogonal to unit n_hat."""
    ref = np.array([1.0, 0.0, 0.0]) if abs(n_hat[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(n_hat, ref)
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(n_hat, e1)   # already unit and ⟂ to both
    return e1, e2


def sample_initial_conditions_dtwa(
    n_traj: int,
    rng: np.random.Generator,
    state="up",
) -> tuple[np.ndarray, np.ndarray]:
    """Discrete-TWA (DTWA) sampling of a spin-1/2 coherent state along any axis.

    Parameters
    ----------
    n_traj : number of trajectories
    rng    : NumPy random generator
    state  : the coherent-state Bloch direction, given as either
             - a name: "up", "down", "+x", "-x", "+y", "-y", or
             - a length-3 array-like n_hat (normalised internally)

    Returns
    -------
    (theta, phi) : arrays of shape (n_traj,)
    """
    n_hat = bloch_direction(state)
    e1, e2 = _transverse_frame(n_hat)
    a = rng.choice([-1.0, 1.0], size=n_traj)
    b = rng.choice([-1.0, 1.0], size=n_traj)
    s = a[:, None] * e1 + b[:, None] * e2 + n_hat   # (n_traj, 3), |s| = √3
    return spin_to_angles(s[:, 0], s[:, 1], s[:, 2])


def sample_initial_conditions_wigner_cone(
    n_traj: int,
    rng: np.random.Generator,
    state="up",
) -> tuple[np.ndarray, np.ndarray]:
    """Continuous Wigner-function sampling of a spin-1/2 coherent state.

    Parameters
    ----------
    n_traj : number of trajectories
    rng    : NumPy random generator
    state  : coherent-state Bloch direction (name or length-3 n_hat; see
             :func:`bloch_direction`)

    Returns
    -------
    (theta, phi) : arrays of shape (n_traj,)
    """
    n_hat = bloch_direction(state)
    e1, e2 = _transverse_frame(n_hat)
    cos_g = 1.0 / SQRT3
    sin_g = np.sqrt(1.0 - cos_g * cos_g)
    alpha = rng.uniform(0.0, 2.0 * np.pi, size=n_traj)
    transverse = np.cos(alpha)[:, None] * e1 + np.sin(alpha)[:, None] * e2
    s = cos_g * n_hat + sin_g * transverse           # unit vector, |s| = 1
    s = SQRT3 * s                                     # onto the sqrt(3)-sphere
    return spin_to_angles(s[:, 0], s[:, 1], s[:, 2])


# -- Collective decay matrix Gamma and its noise factor G (Gamma = G G^T) ---------------

def decay_matrix_dicke(N: int, kappa: float) -> np.ndarray:
    """All-to-all (Dicke / BTC) decay matrix Gamma_{mn} = kappa for every m, n.
    """
    return float(kappa) * np.ones((int(N), int(N)), dtype=float)


def factorize_decay_matrix(Gamma: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Factor a positive-semidefinite decay matrix Gamma = G G^T via eigendecomposition.
    """
    Gamma = np.asarray(Gamma, dtype=float)
    evals, evecs = np.linalg.eigh(Gamma)
    evals = np.maximum(evals, 0.0)
    if evals.max() <= 0.0:
        return np.zeros((Gamma.shape[0], 0))
    keep = evals > tol * evals.max()
    G = evecs[:, keep] * np.sqrt(evals[keep])[None, :]
    return G


# -- Collective (BTC) dynamics -------------------------------------------------
#
# The diffusion-completed (positive-Gamma) collective scheme.  The diffusion is the
# physical decay matrix Gamma = G G^T, and every Wiener channel acts on each spin as a
# rotation about a fixed lab axis, so |s_n| = sqrt(3) is preserved and there is no pole
# singularity.  For the Dicke case G[:,0] = sqrt(kappa): one shared channel, i.e. two
# global rotations of the whole configuration about the x- and y-axis.

def collective_diffusive_drift_cartesian(
    s: np.ndarray,
    Omega: float,
    Delta: float,
    G: np.ndarray,
) -> np.ndarray:
    """Cartesian Stratonovich drift for the diffusion-completed collective scheme.

    Parameters
    ----------
    s     : Bloch vectors, shape (N, n_traj, 3)
    Omega, Delta : drive and detuning
    G     : (N, R) noise factor of the decay matrix

    Returns
    -------
    f : drift vectors, shape (N, n_traj, 3); exactly tangential (s*f = 0)
    """
    sx, sy, sz = s[..., 0], s[..., 1], s[..., 2]
    AX = 0.5 * (G @ (G.T @ sx))          
    AY = 0.5 * (G @ (G.T @ sy))
    fx = sz * AX - Delta * sy
    fy = sz * AY + Delta * sx - Omega * sz
    fz = -(sx * AX + sy * AY) + Omega * sy
    return np.stack([fx, fy, fz], axis=-1)


def _collective_noise_rotations(
    s: np.ndarray,
    dt: float,
    rng: np.random.Generator,
    G: np.ndarray,
) -> np.ndarray:
    """Collective decay noise as exact lab-axis rotations (Strang-symmetrised).
    """
    N, n_traj, _ = s.shape
    R = G.shape[1]
    if R == 0:
        return s
    sdt = np.sqrt(dt)
    dWx = rng.normal(0.0, sdt, size=(R, n_traj))
    dWy = rng.normal(0.0, sdt, size=(R, n_traj))
    alpha = G @ dWx          # (N, n_traj) rotation angle about x-axis for each spin
    beta = G @ dWy           # (N, n_traj) rotation angle about y-axis for each spin
    s = _rotate_x(s, 0.5 * alpha)
    s = _rotate_y(s, beta)
    s = _rotate_x(s, 0.5 * alpha)
    return s


def collective_diffusive_rotation_step(
    s: np.ndarray,
    dt: float,
    Omega: float,
    Delta: float,
    rng: np.random.Generator,
    G: np.ndarray,
) -> np.ndarray:
    """Pole-safe Strang step for the diffusion-completed collective BTC ensemble.

    Parameters
    ----------
    s     : Bloch vectors, shape (N, n_traj, 3)
    dt    : time step
    Omega, Delta : drive and detuning
    rng   : NumPy random generator
    G     : (N, R) noise factor of the decay matrix, Gamma = G Gᵀ (see
            :func:`collective_noise_factor`)
    """
    f = collective_diffusive_drift_cartesian(s, Omega, Delta, G)
    s = _half_rotation(s, f, 0.5 * dt)
    s = _collective_noise_rotations(s, dt, rng, G)
    f = collective_diffusive_drift_cartesian(s, Omega, Delta, G)
    s = _half_rotation(s, f, 0.5 * dt)
    return s


def collective_noise_factor(
    N: int,
    kappa: float,
    Gamma: np.ndarray | None = None,
) -> np.ndarray:
    """Noise factor G (Gamma = G G^T) for the diffusion-completed collective scheme.
    """
    if Gamma is None:
        return np.sqrt(float(kappa)) * np.ones((int(N), 1)) # Return Dicke factor
    return factorize_decay_matrix(np.asarray(Gamma, dtype=float))


def _sample_collective_initial(
    count, rng, sampling, state, theta0, phi0, spread, S
) -> tuple[np.ndarray, np.ndarray]:
    """Initial (theta, phi) for the collective ensemble, dispatched on sampling.
    """
    if sampling == "dtwa":
        return sample_initial_conditions_dtwa(count, rng, state)
    if sampling == "wigner_cone":
        return sample_initial_conditions_wigner_cone(count, rng, state)
    if sampling == "continuous":
        return sample_initial_conditions(count, rng, theta0, phi0, spread / np.sqrt(2.0 * S))
    raise ValueError(
        f"Unknown sampling scheme '{sampling}'. "
        "Choose 'dtwa', 'wigner_cone', or 'continuous'."
    )


def make_collective_stepper(Omega, Delta, G):
    """Return step(s, dt, rng) -> s advancing the Cartesian state one step.
    """
    if G is None:
        raise ValueError("the collective stepper requires the noise factor G.")
    return lambda s, dt, rng: collective_diffusive_rotation_step(
        s, dt, Omega, Delta, rng, G
    )


def run_collective_trajectories(
    N: int,
    Omega: float,
    Delta: float,
    kappa: float,
    times: np.ndarray,
    n_traj: int,
    seed: int = 42,
    sampling: str = "dtwa",
    state: str = "up",
    theta0: float = np.pi / 2,
    phi0: float = 0.0,
    spread: float = 0.0,
    Gamma: np.ndarray | None = None,
    progress: bool = False,
) -> dict:
    """Integrate the collective BTC ensemble of ``N coupled spins by averaging
    over n_traj stochastic trajectories.

    Parameters
    ----------
    N        : number of spins (collective spin S = N/2)
    Omega    : driving frequency
    Delta    : detuning
    kappa    : single-particle decay rate
    times    : 1-D array with uniform spacing
    n_traj   : number of trajectories
    seed     : random seed for reproducibility
    sampling : initial-condition scheme - "dtwa" (default), "wigner_cone" or
               "continuous"
    state    : "up" / "down" initial polarisation for "dtwa" / "wigner_cone"
    theta0,
    phi0     : mean initial angles for "continuous" sampling
    spread   : Gaussian spread of the "continuous" initial condition
    Gamma    : optional (N, N) PSD decay matrix;
               None (default) uses the all-to-all Dicke Γ_mn = κ
    progress : tqdm progress bar over the time steps

    Returns
    -------
    dict with keys (each magnetisation array has shape ``(n_time, n_traj)``):
        'mx', 'my', 'mz' - normalised collective magnetisation per realisation
        'times'          - the input time array
    """
    times = np.asarray(times, dtype=float)
    n_time = len(times)
    if n_time < 2:
        raise ValueError("`times` must contain at least two points.")
    if int(N) != N or N < 1:
        raise ValueError("`N` must be a positive integer.")
    N = int(N)
    S = N / 2.0
    dt = float(times[1] - times[0])

    G = collective_noise_factor(N, kappa, Gamma)
    step = make_collective_stepper(Omega, Delta, G)

    rng = np.random.default_rng(seed)
    theta, phi = _sample_collective_initial(
        N * n_traj, rng, sampling, state, theta0, phi0, spread, S
    )
    theta = theta.reshape(N, n_traj)
    phi = phi.reshape(N, n_traj)

    # Cartesian state (N, n_traj, 3) for the pole-safe rotation solver.
    s = np.stack(angles_to_spin(theta, phi), axis=-1)   # (N, n_traj, 3)

    mx = np.empty((n_time, n_traj))
    my = np.empty((n_time, n_traj))
    mz = np.empty((n_time, n_traj))
    mx[0], my[0], mz[0] = s[..., 0].mean(axis=0), s[..., 1].mean(axis=0), s[..., 2].mean(axis=0)

    step_iter = range(n_time - 1)
    if progress:
        from tqdm.auto import tqdm

        step_iter = tqdm(step_iter, desc="collective BTC SDE")

    for n in step_iter:
        s = step(s, dt, rng)
        mx[n + 1], my[n + 1], mz[n + 1] = (
            s[..., 0].mean(axis=0),
            s[..., 1].mean(axis=0),
            s[..., 2].mean(axis=0),
        )

    return {"mx": mx, "my": my, "mz": mz, "times": times}
