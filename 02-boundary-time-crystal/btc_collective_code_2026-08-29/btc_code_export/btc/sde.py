"""Stochastic dynamics of a single driven–dissipative spin-½ on the S² sphere.

This module integrates the semiclassical (DCTWA) Langevin equations for a
single driven–dissipative spin, as derived from the SU(2) Keldysh path
integral.  These are the *star-product* equations of
motion, which differ from the naive ordinary-product ones in the
dissipative sector.

Model
-----
The spin is described by the two angular coordinates ``(θ, φ)`` of a Bloch
vector on the radius-√3 Weyl sphere (the spin-½ symbol sphere).  The coherent
Hamiltonian in angular variables is

    H_cl(θ, φ) = (√3 Ω / 2) sin θ cos φ  −  (√3 Δ / 2) cos θ,

and the single jump operator is the lowering operator √Γ σ⁻, with Weyl symbol
σ⁻(θ, φ) = (√3 / 2) sin θ e^{−iφ}.

Equations of motion (Eqs. 11–13 of the note)
---------------------------------------------
Deterministic drift:

    θ̇ = Ω sin φ + Γ ( cot θ − csc θ / √3 )                          (drift_θ)
    φ̇ = Ω cot θ cos φ + Δ                                            (drift_φ)

Multiplicative noise (diffusion matrix in the canonical coordinates that map
directly onto θ, φ):

    (BBᵀ)_φφ = Γ ( 1 + 2 cot²θ − 2 cot θ csc θ / √3 ) ≡ b²(θ)
    (BBᵀ)_θθ = 0,        (BBᵀ)_θφ = 0.

The full Langevin system is therefore

    dθ = [ Ω sin φ + Γ ( cot θ − csc θ / √3 ) ] dt
    dφ = [ Ω cot θ cos φ + Δ ] dt  +  b(θ) dW.

Crucially the correct star-product action carries **no θ-noise** — the
"phantom" θ-noise of the ordinary-product approach is absent.  Because the only noise
drives φ with an amplitude b(θ) that depends solely on the (noiseless) angle θ,
∂b/∂φ = 0 and there is no Itô–Stratonovich correction: plain Euler–Maruyama is
consistent to first order in Δt.

The drift fixed point in the absence of driving (Ω = 0) sits at
cos θ = 1/√3, the angular position the dissipator pulls toward.

Spin observables
----------------
The Weyl symbols of the Pauli operators on the radius-√3 sphere are

    s_x = √3 sin θ cos φ,   s_y = √3 sin θ sin φ,   s_z = −√3 cos θ,

so that s_x² + s_y² + s_z² = 3.  Quantum expectation values are recovered by
averaging these symbols over the trajectory ensemble.  The sign of s_z is
fixed by the dissipator: pure decay (Ω = Δ = 0) relaxes to cos θ = 1/√3, which
must be the down state |↓⟩ with ⟨σ_z⟩ = −1, hence s_z = −√3 cos θ.  With this
convention the given H_cl corresponds to the operator H = (Ω/2) σ_x + (Δ/2) σ_z.

Usage
-----
>>> import numpy as np
>>> from btc.sde import run_trajectories
>>> times = np.linspace(0.0, 20.0, 2001)
>>> out = run_trajectories(Omega=1.0, Delta=0.5, Gamma=1.0,
...                        times=times, n_traj=2000, seed=0)
>>> sz_mean = out["sz"].mean(axis=1)   # ⟨σ_z⟩(t)
"""

from __future__ import annotations

import numpy as np

SQRT3 = np.sqrt(3.0)


# ── Right-hand sides ──────────────────────────────────────────────────────────

def drift(
    theta: np.ndarray,
    phi: np.ndarray,
    Omega: float,
    Delta: float,
    Gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic drift (θ̇, φ̇) of the single-spin EoMs (Eqs. 11–12).

    Parameters
    ----------
    theta, phi : polar / azimuthal angles (arrays of equal shape)
    Omega      : Rabi (drive) frequency Ω
    Delta      : detuning Δ
    Gamma      : single-spin decay rate Γ

    Returns
    -------
    (dtheta, dphi) : drift components, same shape as the inputs
    """
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    cot_t = cos_t / sin_t
    csc_t = 1.0 / sin_t

    dtheta = Omega * np.sin(phi) + Gamma * (cot_t - csc_t / SQRT3)
    dphi = Omega * cot_t * np.cos(phi) + Delta
    return dtheta, dphi


def noise_amplitude(theta: np.ndarray, Gamma: float) -> np.ndarray:
    """Azimuthal noise amplitude b(θ) = √(BBᵀ)_φφ (Eq. 13).

    b²(θ) = Γ ( 1 + 2 cot²θ − 2 cot θ csc θ / √3 ).

    The argument of the square root is non-negative on the physically relevant
    range; it is clipped at zero to guard against round-off near the poles.
    """
    sin_t = np.sin(theta)
    cot_t = np.cos(theta) / sin_t
    csc_t = 1.0 / sin_t

    b2 = Gamma * (1.0 + 2.0 * cot_t**2 - 2.0 * cot_t * csc_t / SQRT3)
    return np.sqrt(np.maximum(b2, 0.0))


# ── Symbol → observable map ────────────────────────────────────────────────────

def angles_to_spin(
    theta: np.ndarray,
    phi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weyl symbols (s_x, s_y, s_z) of the Pauli operators on the √3-sphere.

        s_x = √3 sin θ cos φ,  s_y = √3 sin θ sin φ,  s_z = −√3 cos θ.
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
    """Inverse of :func:`angles_to_spin`: (s_x, s_y, s_z) → (θ, φ).

    Assumes the points lie on the radius-√3 sphere (s_x²+s_y²+s_z² = 3).
    """
    theta = np.arccos(np.clip(-np.asarray(sz) / SQRT3, -1.0, 1.0))
    phi = np.arctan2(sy, sx)
    return theta, phi


# ── Single integration step ────────────────────────────────────────────────────

def euler_maruyama_step(
    theta: np.ndarray,
    phi: np.ndarray,
    dt: float,
    Omega: float,
    Delta: float,
    Gamma: float,
    rng: np.random.Generator,
    theta_eps: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance one Euler–Maruyama step for an ensemble of trajectories.

    The noise enters only the φ equation, with amplitude b(θ); since b depends
    on the noiseless θ only, the Itô and Stratonovich schemes coincide and this
    step is consistent to O(dt).

    Parameters
    ----------
    theta, phi : current angles, shape (n_traj,)
    dt         : time step
    Omega, Delta, Gamma : model parameters
    rng        : NumPy random generator (for reproducibility)
    theta_eps  : θ is clipped to [theta_eps, π − theta_eps] after each step to
                 keep cot θ / csc θ finite near the poles.

    Returns
    -------
    (theta_next, phi_next)
    """
    dtheta, dphi = drift(theta, phi, Omega, Delta, Gamma)
    b = noise_amplitude(theta, Gamma)

    dW = rng.normal(0.0, np.sqrt(dt), size=phi.shape)

    theta_next = theta + dtheta * dt
    phi_next = phi + dphi * dt + b * dW

    # Keep θ away from the coordinate-singular poles.
    np.clip(theta_next, theta_eps, np.pi - theta_eps, out=theta_next)
    return theta_next, phi_next


# ── Cartesian rotation integrator (pole-safe) ──────────────────────────────────
#
# The (θ, φ) Euler–Maruyama step above suffers near the poles, where cot θ / csc θ
# blow up.  Working with the Cartesian Bloch vector s = (s_x, s_y, s_z) on the
# radius-√3 sphere removes the coordinate singularity: every sub-update is an
# exact SO(3) rotation, so |s| = √3 is preserved and no clipping is required.

R2 = 3.0  # squared radius of the Weyl sphere (r = √3)


def drift_cartesian(
    s: np.ndarray,
    Omega: float,
    Delta: float,
    Gamma: float,
    rho2_floor: float = 1e-12,
) -> np.ndarray:
    """Cartesian drift d**s**/dt obtained by mapping the (θ, φ) EoMs through s(θ, φ).

    Coherent part is a rigid precession about the fixed axis B = (Ω, 0, Δ); the
    dissipative part is the meridian flow that relaxes s_z → −1 (the |↓⟩ point):

        f_coh  = (Ω, 0, Δ) × s
        f_diss = Γ (s_z + 1) ( s_x s_z / ρ², s_y s_z / ρ², −1 ),   ρ² = s_x² + s_y².

    The coherent part is singularity-free (the cot θ terms of φ̇ cancel); only the
    dissipative meridian rate carries the (physical) 1/ρ pole behaviour.

    Parameters
    ----------
    s          : Bloch vectors, shape (..., 3), on the radius-√3 sphere
    Omega, Delta, Gamma : model parameters
    rho2_floor : floor on ρ² to avoid 0/0 exactly at the poles (NaN guard only)

    Returns
    -------
    f : drift vectors, shape (..., 3)
    """
    sx, sy, sz = s[..., 0], s[..., 1], s[..., 2]

    # Coherent precession about the fixed axis B = (Ω, 0, Δ):  f = B × s
    fx = -Delta * sy
    fy = Delta * sx - Omega * sz
    fz = Omega * sy

    # Dissipative meridian flow
    rho2 = np.maximum(sx * sx + sy * sy, rho2_floor)
    g = Gamma * (sz + 1.0)
    fx = fx + g * sx * sz / rho2
    fy = fy + g * sy * sz / rho2
    fz = fz - g
    return np.stack([fx, fy, fz], axis=-1)


def _rotate_axis(s: np.ndarray, axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Rodrigues rotation of vectors ``s`` about unit ``axis`` by ``angle``.

    Vectorised over arbitrary leading dimensions (the last axis is the 3-vector);
    preserves |s| exactly.
    """
    c = np.cos(angle)[..., None]
    sn = np.sin(angle)[..., None]
    nds = np.sum(axis * s, axis=-1)[..., None]
    nxs = np.cross(axis, s)
    return c * s + sn * nxs + (1.0 - c) * nds * axis


def _rotate_z(s: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Rotate vectors ``s`` about +ẑ by ``angle`` (leaves s_z fixed)."""
    c = np.cos(angle)
    sn = np.sin(angle)
    sx = c * s[..., 0] - sn * s[..., 1]
    sy = sn * s[..., 0] + c * s[..., 1]
    return np.stack([sx, sy, s[..., 2]], axis=-1)


def _rotate_x(s: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Rotate vectors ``s`` about +x̂ by ``angle`` (leaves s_x fixed).

    For small ``angle`` this is ds = angle (x̂ × s); norm-preserving to all orders.
    """
    c = np.cos(angle)
    sn = np.sin(angle)
    sy = c * s[..., 1] - sn * s[..., 2]
    sz = sn * s[..., 1] + c * s[..., 2]
    return np.stack([s[..., 0], sy, sz], axis=-1)


def _rotate_y(s: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Rotate vectors ``s`` about +ŷ by ``angle`` (leaves s_y fixed).

    For small ``angle`` this is ds = angle (ŷ × s); norm-preserving to all orders.
    """
    c = np.cos(angle)
    sn = np.sin(angle)
    sx = c * s[..., 0] + sn * s[..., 2]
    sz = -sn * s[..., 0] + c * s[..., 2]
    return np.stack([sx, s[..., 1], sz], axis=-1)


def _half_rotation(s: np.ndarray, f: np.ndarray, half_dt: float) -> np.ndarray:
    """Half deterministic rotation about ω = (s × f)/r² by angle |ω|·half_dt.

    Shape-generic (last axis is the 3-vector).  Where |ω| = 0 the rotation is the
    identity, so no special-casing or pole clipping is needed.
    """
    w = np.cross(s, f) / R2
    wn = np.linalg.norm(w, axis=-1)
    wn_safe = np.where(wn > 0.0, wn, 1.0)
    axis = w / wn_safe[..., None]
    return _rotate_axis(s, axis, half_dt * wn)


def rotation_step(
    s: np.ndarray,
    dt: float,
    Omega: float,
    Delta: float,
    Gamma: float,
    rng: np.random.Generator,
    rho2_floor: float = 1e-12,
) -> np.ndarray:
    """One pole-safe Strang step for the single-spin SDE in Cartesian form.

    Splitting (each sub-update is an exact SO(3) rotation):

        A) half deterministic rotation   ω = (s × f)/r²,  angle |ω|·dt/2
        B) stochastic rotation about ẑ    angle b(θ)·dW   (φ-noise only)
        C) half deterministic rotation    (drift recomputed at the updated s)

    The φ-noise is a ẑ-rotation, which leaves s_z (and hence b) unchanged, so a
    single rotation reproduces dφ = b dW exactly with no Itô–Stratonovich drift.
    In Cartesian form the noise amplitude is

        b²(θ) = Γ ( 1 + 2 s_z (s_z + 1) / ρ² ),   ρ² = s_x² + s_y².

    Parameters
    ----------
    s     : Bloch vectors, shape (n_traj, 3), on the radius-√3 sphere
    dt    : time step
    Omega, Delta, Gamma : model parameters
    rng   : NumPy random generator
    rho2_floor : floor on ρ² (NaN guard at the poles)

    Returns
    -------
    s_next : Bloch vectors, shape (n_traj, 3)
    """
    # A) half deterministic rotation
    s = _half_rotation(s, drift_cartesian(s, Omega, Delta, Gamma, rho2_floor), 0.5 * dt)

    # B) φ-noise as a ẑ-rotation by b(θ) dW
    s = _noise_z_rotation(s, dt, Gamma, rng, rho2_floor)

    # C) half deterministic rotation (drift recomputed at updated s)
    s = _half_rotation(s, drift_cartesian(s, Omega, Delta, Gamma, rho2_floor), 0.5 * dt)
    return s


def _noise_z_rotation(
    s: np.ndarray,
    dt: float,
    Gamma: float,
    rng: np.random.Generator,
    rho2_floor: float,
) -> np.ndarray:
    """φ-noise sub-step: ẑ-rotation by b(θ) dW with b² = Γ(1 + 2 s_z(s_z+1)/ρ²).

    The ẑ-rotation leaves s_z (hence b) unchanged, so a single rotation reproduces
    dφ = b dW exactly with no Itô–Stratonovich drift.  Shape-generic: an
    independent increment is drawn per element of ``s.shape[:-1]``.
    """
    sz = s[..., 2]
    rho2 = np.maximum(s[..., 0] ** 2 + s[..., 1] ** 2, rho2_floor)
    b = np.sqrt(np.maximum(Gamma * (1.0 + 2.0 * sz * (sz + 1.0) / rho2), 0.0))
    dW = rng.normal(0.0, np.sqrt(dt), size=s.shape[:-1])
    return _rotate_z(s, b * dW)


# ── Initial conditions ─────────────────────────────────────────────────────────

def sample_initial_conditions(
    n_traj: int,
    rng: np.random.Generator,
    theta0: float = np.pi / 2,
    phi0: float = 0.0,
    spread: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Initial (θ, φ) for ``n_traj`` trajectories.

    By default every trajectory starts at the same point ``(theta0, phi0)``
    (a delta-distributed / purely deterministic initial condition).  Setting
    ``spread > 0`` adds an isotropic Gaussian fuzz of that standard deviation
    to both angles, a crude Wigner-style sampling of the initial state.

    Parameters
    ----------
    S       : spin length
    n_traj  : number of trajectories
    rng     : NumPy random generator
    theta0  : mean initial polar angle      (default π/2, the equator)
    phi0    : mean initial azimuthal angle  (default 0)
    spread  : standard deviation of the initial Gaussian fuzz (default 0)

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


# Named Bloch directions n̂ = (⟨σ_x⟩, ⟨σ_y⟩, ⟨σ_z⟩) for the DTWA coherent state.
_DTWA_DIRECTIONS = {
    "up": (0.0, 0.0, 1.0),
    "down": (0.0, 0.0, -1.0),
    "+x": (1.0, 0.0, 0.0),
    "-x": (-1.0, 0.0, 0.0),
    "+y": (0.0, 1.0, 0.0),
    "-y": (0.0, -1.0, 0.0),
}


def bloch_direction(state) -> np.ndarray:
    """Resolve a coherent-state spec to a unit Bloch vector n̂ = (⟨σ_x⟩,⟨σ_y⟩,⟨σ_z⟩).

    ``state`` is either a name (``"up"``, ``"down"``, ``"+x"``, ``"-x"``,
    ``"+y"``, ``"-y"``) or a length-3 array-like Bloch vector (normalised here).
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
    """Two orthonormal vectors (e₁, e₂) spanning the plane ⟂ to unit ``n_hat``."""
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
    """Discrete-TWA (DTWA) sampling of a spin-½ coherent state along any axis.

    The spin-½ DTWA ensemble for a coherent state pointing along the Bloch unit
    vector n̂ = (⟨σ_x⟩, ⟨σ_y⟩, ⟨σ_z⟩) is the ±z ("up") ensemble rotated into the
    frame aligned with n̂: the symbol along n̂ is fixed to +1, and the two
    transverse symbols are drawn independently from {−1, +1},

        s = a e₁ + b e₂ + n̂,   a, b ∈ {−1, +1},

    with (e₁, e₂) ⟂ n̂.  Every sample has |s| = √3 (it lies on the Weyl sphere),
    its mean is ⟨s⟩ = n̂ (so ⟨σ_a⟩ = n_a), and the transverse symbols reproduce
    the correct DTWA second moments.  For n̂ = ±ẑ this is the standard
    Schachenmayer spin-½ ensemble.

    Parameters
    ----------
    n_traj : number of trajectories
    rng    : NumPy random generator
    state  : the coherent-state Bloch direction, given as either
             - a name: ``"up"``, ``"down"``, ``"+x"``, ``"-x"``, ``"+y"``,
               ``"-y"``, or
             - a length-3 array-like n̂ = (⟨σ_x⟩, ⟨σ_y⟩, ⟨σ_z⟩) (normalised
               internally), e.g. ``(np.sin θ cos φ, np.sin θ sin φ, np.cos θ)``
               for an arbitrary coherent state.

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
    """Continuous Wigner-function sampling of a spin-½ coherent state (Eq. 6 of the
    collective-DTWA paper, arXiv:2305.19829).

    The paper's positive Wigner function of a coherent state |ψ⟩ pointing along the
    Bloch unit vector n̂ is a *cone*: a fixed polar tilt ``γ`` away from n̂ with the
    azimuth around n̂ drawn uniformly,

        s = √3 [ cos γ · n̂ + sin γ ( cos α · e₁ + sin α · e₂ ) ],   α ~ U[0, 2π),

    with cos γ = 1/√3 (so the tilt is the magic angle).  Every sample lies on the
    √3-sphere (|s| = √3), has mean ⟨s⟩ = n̂ (radius-1 magnetisation, i.e.
    ⟨σ_a⟩ = n_a), and transverse second moment ⟨s_⊥²⟩ = 1 per component — the same
    first/second moments as :func:`sample_initial_conditions_dtwa`, but as a
    *continuous* distribution.  This is the initial-condition sampler matched to the
    truncated correspondence-rule dynamics (the ``"diffusive"`` collective scheme),
    whereas the discrete ±1 DTWA sampler injects spurious higher cumulants.

    Parameters
    ----------
    n_traj : number of trajectories
    rng    : NumPy random generator
    state  : coherent-state Bloch direction (name or length-3 n̂; see
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
    s = SQRT3 * s                                     # onto the √3-sphere
    return spin_to_angles(s[:, 0], s[:, 1], s[:, 2])


# ── Collective decay matrix Γ and its noise factor G (Γ = G Gᵀ) ─────────────────

def decay_matrix_dicke(N: int, kappa: float) -> np.ndarray:
    """All-to-all (Dicke / BTC) decay matrix Γ_{mn} = κ for every m, n.

    This is the collective single jump √κ Ŝ₋ written as the rank-1 decay matrix of
    the general master equation ½ Σ_{mn} Γ_{mn}(2σ⁻_m ρ σ⁺_n − {σ⁺_m σ⁻_n, ρ}).
    Returns an ``(N, N)`` array.
    """
    return float(kappa) * np.ones((int(N), int(N)), dtype=float)


def factorize_decay_matrix(Gamma: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Factor a positive-semidefinite decay matrix Γ = G Gᵀ via eigendecomposition.

    Returns ``G`` of shape ``(N, R)`` keeping only the ``R`` eigenvectors with
    eigenvalue above ``tol · max(eig)`` — so the rank-1 Dicke Γ yields a single
    noise channel (``R = 1``).  Eigendecomposition (not Cholesky) is used because
    physical Γ are generically rank-deficient/PSD-singular, where Cholesky fails.
    Tiny negative eigenvalues (round-off) are clipped to zero.

    The columns of ``G`` are the *noise channels*: with ``G``, the collective
    decay noise couples spin ``n`` to channel ``c`` with weight ``G[n, c]``.  For
    the Dicke case ``G[:, 0] = √κ`` — all spins share one channel (two real Wiener
    processes), exactly the cavity amplitude dα = dW_x + i dW_y of the paper.
    """
    Gamma = np.asarray(Gamma, dtype=float)
    evals, evecs = np.linalg.eigh(Gamma)
    evals = np.maximum(evals, 0.0)
    if evals.max() <= 0.0:
        return np.zeros((Gamma.shape[0], 0))
    keep = evals > tol * evals.max()
    G = evecs[:, keep] * np.sqrt(evals[keep])[None, :]
    return G


# ── Trajectory ensemble integrator ─────────────────────────────────────────────

def run_trajectories(
    S: float,
    Omega: float,
    Delta: float,
    Gamma: float,
    times: np.ndarray,
    n_traj: int,
    seed: int = 42,
    sampling: str = "dtwa",
    state: str = "up",
    theta0: float = np.pi / 2,
    phi0: float = 0.0,
    spread: float = 0.0,
    progress: bool = False,
) -> dict:
    """Integrate the single-spin SDE ensemble on a uniform time grid.

    Parameters
    ----------
    S        : spin length
    Omega, Delta, Gamma : model parameters (Rabi frequency, detuning, decay),
               decay rate Gamma adjusted for spin size as Gamma / (2S)
    times    : 1-D array with *uniform* spacing
    n_traj   : number of stochastic trajectories
    seed     : random seed for reproducibility
    sampling : initial-condition scheme – ``"dtwa"`` (default, spin-½ DTWA
               sampling of ``state``) or ``"continuous"`` (delta/Gaussian start
               at ``theta0, phi0`` with width ``spread``)
    state    : ``"up"`` or ``"down"`` – initial polarisation for DTWA sampling
    theta0,
    phi0     : mean initial angles for ``"continuous"`` sampling
    spread   : Gaussian spread of the ``"continuous"`` initial condition,
               adjusted internally by factor of 1 / (2S) for spin length
    progress : show a tqdm progress bar over the time steps

    Returns
    -------
    dict with keys (each spin/angle array has shape ``(n_time, n_traj)``):
        ``'theta'``  – polar angle θ(t)
        ``'phi'``    – azimuthal angle φ(t)
        ``'sx'``,
        ``'sy'``,
        ``'sz'``     – normalized Weyl symbols of S_x / S, S_y / S, S_z / S
        ``'times'``  – the input time array
    """
    times = np.asarray(times, dtype=float)
    n_time = len(times)
    if n_time < 2:
        raise ValueError("`times` must contain at least two points.")
    dt = float(times[1] - times[0])

    rng = np.random.default_rng(seed)
    if sampling == "dtwa":
        theta, phi = sample_initial_conditions_dtwa(n_traj, rng, state)
    elif sampling == "continuous":
        theta, phi = sample_initial_conditions(n_traj, rng, theta0, phi0, spread / np.sqrt(2*S))
    else:
        raise ValueError(
            f"Unknown sampling scheme '{sampling}'. Choose 'dtwa' or 'continuous'."
        )

    # Integrate in Cartesian with the pole-safe rotation solver; the spin
    # symbols are the state itself, and θ, φ are recovered for output.  The
    # effective decay rate is Γ/(2S), matching the spin-size rescaling.
    Gamma_eff = Gamma / (2 * S)
    s = np.stack(angles_to_spin(theta, phi), axis=1)   # (n_traj, 3)

    sx_t = np.empty((n_time, n_traj))
    sy_t = np.empty((n_time, n_traj))
    sz_t = np.empty((n_time, n_traj))
    sx_t[0], sy_t[0], sz_t[0] = s[:, 0], s[:, 1], s[:, 2]

    step_iter = range(n_time - 1)
    if progress:
        from tqdm.auto import tqdm

        step_iter = tqdm(step_iter, desc="single-spin SDE")

    for n in step_iter:
        s = rotation_step(s, dt, Omega, Delta, Gamma_eff, rng)
        sx_t[n + 1], sy_t[n + 1], sz_t[n + 1] = s[:, 0], s[:, 1], s[:, 2]

    theta_t, phi_t = spin_to_angles(sx_t, sy_t, sz_t)
    return {
        "theta": theta_t,
        "phi": phi_t,
        "sx": sx_t,
        "sy": sy_t,
        "sz": sz_t,
        "times": times,
    }


# ── Collective (BTC) dynamics ───────────────────────────────────────────────────
#
# The isolated-spin equations above describe one spin-½ decaying into its own
# bath (jump √Γ σ⁻).  The boundary time crystal instead has a single *collective*
# jump √κ S₋ = √κ Σₙ σ₋ⁿ.  Decomposing 𝒟[S₋] into diagonal (m=n) and off-diagonal
# (m≠n) parts:
#
#   • the diagonal part is Σₙ 𝒟[σ₋ⁿ] — the isolated-spin decay above, at rate κ;
#   • the off-diagonal part, after the standard DTWA mean-field factorisation,
#     reduces to an effective single-spin Hamiltonian Ĥⁿ_coll = (κ/4)(Mₓσ_yⁿ −
#     M_yσₓⁿ) with the collective transverse field Mₐ = Σ_{m≠n} sₐᵐ.  Mapped to
#     the (θ,φ) sphere it is a pure drift,
#
#         θ̇|coll = −(κ/2)(Mₓ cos φ + M_y sin φ),
#         φ̇|coll = +(κ/2) cot θ (Mₓ sin φ − M_y cos φ),
#
#     i.e. each spin precesses in the field of the collective dipole.  Being a
#     commutator it contributes no extra noise — the only stochastic term stays
#     the single-spin decay noise b(θ); collective fluctuations enter through the
#     spread of M across the sampled ensemble.
#
# As N→∞ with g ≡ κS = κN/2 fixed, the O(N) collective drift dominates the O(1)
# single-spin decay and reproduces the superradiant mean field
# ṁₓ = g mₓm_z,  ṁ_y = −ω m_z + g m_y m_z,  ṁ_z = ω m_y − g(mₓ²+m_y²).

def collective_euler_maruyama_step(
    theta: np.ndarray,
    phi: np.ndarray,
    dt: float,
    Omega: float,
    Delta: float,
    kappa: float,
    rng: np.random.Generator,
    theta_eps: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """One Euler–Maruyama step for an ensemble of *collectively coupled* spins.

    ``theta`` and ``phi`` have shape ``(N, n_traj)``: ``N`` collective spins for
    each of ``n_traj`` independent realisations.  Each spin evolves under the
    isolated-spin drive/detuning/decay of :func:`drift` and :func:`noise_amplitude`
    (rate ``kappa`` — the diagonal m=n part of 𝒟[S₋]) plus the superradiant
    collective drift derived from the off-diagonal m≠n part,

        θ̇|coll = −(κ/2)(Mₓ cos φ + M_y sin φ),
        φ̇|coll = +(κ/2) cot θ (Mₓ sin φ − M_y cos φ),

    with the collective transverse field Mₐ = Σ_{m≠n} sₐᵐ (each spin's own
    contribution removed).  The collective coupling is pure drift, so the only
    noise is the single-spin decay term on φ.

    Returns ``(theta_next, phi_next)`` of shape ``(N, n_traj)``.
    """
    sin_t = np.sin(theta)
    cos_p = np.cos(phi)
    sin_p = np.sin(phi)

    # Cartesian √3-sphere symbols of every spin, and the collective field seen by
    # each spin (total over spins minus its own contribution).
    sx = SQRT3 * sin_t * cos_p
    sy = SQRT3 * sin_t * sin_p
    Mx = sx.sum(axis=0, keepdims=True) - sx
    My = sy.sum(axis=0, keepdims=True) - sy

    # Diagonal (isolated-spin) drive + detuning + decay at rate kappa …
    dtheta, dphi = drift(theta, phi, Omega, Delta, kappa)
    # … plus the off-diagonal collective superradiant drift.
    cot_t = np.cos(theta) / sin_t
    dtheta = dtheta - 0.5 * kappa * (Mx * cos_p + My * sin_p)
    dphi = dphi + 0.5 * kappa * cot_t * (Mx * sin_p - My * cos_p)

    b = noise_amplitude(theta, kappa)
    dW = rng.normal(0.0, np.sqrt(dt), size=phi.shape)

    theta_next = theta + dtheta * dt
    phi_next = phi + dphi * dt + b * dW

    np.clip(theta_next, theta_eps, np.pi - theta_eps, out=theta_next)
    return theta_next, phi_next


def collective_drift_cartesian(
    s: np.ndarray,
    Omega: float,
    Delta: float,
    kappa: float,
    rho2_floor: float = 1e-12,
) -> np.ndarray:
    """Cartesian drift of the collective BTC ensemble.

    ``s`` has shape ``(N, n_traj, 3)`` — the spin axis is axis 0.  The drift is
    the single-spin part (drive Ω, detuning Δ, decay κ; see
    :func:`drift_cartesian`) plus the superradiant collective precession, which
    maps from the (θ, φ) collective term to a pure precession about the
    transverse collective field (no cot θ singularity):

        f_coll = b_coll × s,   b_coll = (κ/2) (−M_y, M_x, 0),

    with M_a = Σ_{m≠n} s_a^m (the collective field on spin n, self excluded).
    """
    f = drift_cartesian(s, Omega, Delta, kappa, rho2_floor)

    sx, sy, sz = s[..., 0], s[..., 1], s[..., 2]
    Mx = sx.sum(axis=0, keepdims=True) - sx     # Σ_{m≠n} s_x^m, shape (N, n_traj)
    My = sy.sum(axis=0, keepdims=True) - sy

    # b_coll × s  with  b_coll = (κ/2)(−M_y, M_x, 0)
    bx = -0.5 * kappa * My
    by = 0.5 * kappa * Mx
    fcx = by * sz
    fcy = -bx * sz
    fcz = bx * sy - by * sx
    return f + np.stack([fcx, fcy, fcz], axis=-1)


def collective_rotation_step(
    s: np.ndarray,
    dt: float,
    Omega: float,
    Delta: float,
    kappa: float,
    rng: np.random.Generator,
    rho2_floor: float = 1e-12,
) -> np.ndarray:
    """Pole-safe Strang step for the collective BTC ensemble (Cartesian form).

    Same A) half-drift / B) ẑ-noise / C) half-drift splitting as
    :func:`rotation_step`, but with the collective drift (which recomputes the
    collective field M at the updated configuration in step C).  The only noise
    is the per-spin decay term — an independent ẑ-rotation on each of the
    ``N × n_traj`` spins — exactly as in the (θ, φ) collective step.

    Parameters
    ----------
    s     : Bloch vectors, shape (N, n_traj, 3), on the radius-√3 sphere
    dt    : time step
    Omega, Delta, kappa : model parameters
    rng   : NumPy random generator
    rho2_floor : floor on ρ² (NaN guard at the poles)
    """
    s = _half_rotation(s, collective_drift_cartesian(s, Omega, Delta, kappa, rho2_floor), 0.5 * dt)
    s = _noise_z_rotation(s, dt, kappa, rng, rho2_floor)
    s = _half_rotation(s, collective_drift_cartesian(s, Omega, Delta, kappa, rho2_floor), 0.5 * dt)
    return s


# ── Diffusion-completed collective dynamics (the paper's positive-Γ scheme) ──────
#
# The drift-only steps above drop the *collective* off-diagonal noise of 𝒟[Ŝ₋] and
# therefore underestimate second moments (the QFI).  Following Mink & Fleischhauer
# (arXiv:2305.19829) we instead apply the truncated correspondence rules to the
# full collective dissipator written with the decay matrix Γ_{mn} (= κ for Dicke).
# The diffusion is then the *physical* decay matrix Γ = G Gᵀ ⪰ 0, so the noise is
# genuinely real.  Geometrically (verified to machine precision) each Wiener
# channel acts on every spin as a rotation about a *fixed lab axis*:
#
#     channel "x":  rotate sₙ about +x̂ by  αₙ = Σ_c G[n,c] dW^x_c
#     channel "y":  rotate sₙ about +ŷ by  βₙ = Σ_c G[n,c] dW^y_c
#
# For the Dicke case G[:,0]=√κ, so all spins share one channel — two *global*
# rotations of the whole configuration about x̂ and ŷ (the cavity amplitude
# dα = dW_x + i dW_y).  Because these rotations are exact, |sₙ| = √3 is preserved
# and there is no pole singularity — the route the angle-chart Euler step lacks.
#
# CAUTION when editing: the rotations realise the *Stratonovich* SDE, so they
# pair with the Stratonovich drift :func:`collective_diffusive_drift_cartesian`
# (the full-Γ collective field plus the drive) and with no further Itô
# conversion term.  Adding one — in either chart — double-counts the conversion
# and injects a spurious drift toward the equator.
#
# (``collective_diffusive_em_step`` below is a legacy (θ,φ) Euler–Maruyama
# cross-check that still carries the old, incorrect drift — do not use it.)

def collective_diffusive_drift_cartesian(
    s: np.ndarray,
    Omega: float,
    Delta: float,
    G: np.ndarray,
) -> np.ndarray:
    """Cartesian *Stratonovich* drift for the diffusion-completed collective scheme.

    Pairs with the full-Γ lab-axis rotation noise of
    :func:`_collective_noise_rotations`.  Because the rotations realise the
    Stratonovich SDE, this is the plain chain-rule image of the Stratonovich
    angular drift — the collective field built from the **full** Γ (diagonal
    included) plus the coherent precession, and nothing else:

        AX_n = ½ Σ_m Γ_mn s_x^m ,   AY_n = ½ Σ_m Γ_mn s_y^m
        f = s × (AY, −AX, 0)  +  B × s ,      B = (Ω, 0, Δ)

    A separate single-atom decay term is *not* added: for Γ_mn = κ its role is
    played exactly by the n = m term of the collective sum,
    ½κ(sₓs_z, s_y s_z, −(sₓ²+s_y²)).  Adding the (θ,φ) Itô term on top — or the
    Cartesian correction ½κ(sₓ, s_y, 2s_z) — applies the Itô→Stratonovich
    conversion twice, in two different charts.  Both are tangential-safe: every
    term is a cross product with s, so |s| = √3 is preserved exactly.

    Taking the noise factor ``G`` (Γ = G Gᵀ) rather than the scalar ``kappa``
    generalises the drift to any PSD decay matrix and costs O(N·R) instead of
    O(N²).

    Parameters
    ----------
    s     : Bloch vectors, shape (N, n_traj, 3), on the radius-√3 sphere
    Omega, Delta : drive and detuning
    G     : (N, R) noise factor of the decay matrix, Γ = G Gᵀ (see
            :func:`collective_noise_factor`)

    Returns
    -------
    f : drift vectors, shape (N, n_traj, 3); exactly tangential (s·f = 0)
    """
    sx, sy, sz = s[..., 0], s[..., 1], s[..., 2]
    AX = 0.5 * (G @ (G.T @ sx))          # = ½ Γ·sₓ , shape (N, n_traj)
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

    ``s`` has shape ``(N, n_traj, 3)``; ``G`` is the ``(N, R)`` factor of Γ = G Gᵀ.
    Draws ``R`` independent Wiener increments per axis (shared across spins through
    the rows of ``G``) and applies the symmetric split ½X · Y · ½X so the two
    non-commuting axes contribute a mean-zero O(dt) error that vanishes as dt → 0.
    """
    N, n_traj, _ = s.shape
    R = G.shape[1]
    if R == 0:
        return s
    sdt = np.sqrt(dt)
    dWx = rng.normal(0.0, sdt, size=(R, n_traj))
    dWy = rng.normal(0.0, sdt, size=(R, n_traj))
    alpha = G @ dWx          # (N, n_traj) rotation angle about x̂ for each spin
    beta = G @ dWy           # (N, n_traj) rotation angle about ŷ for each spin
    s = _rotate_x(s, 0.5 * alpha)
    s = _rotate_y(s, beta)
    s = _rotate_x(s, 0.5 * alpha)
    return s


def collective_diffusive_axis(
    s: np.ndarray,
    Omega: float,
    Delta: float,
    G: np.ndarray,
) -> np.ndarray:
    """Precession axis ω_n of the Stratonovich drift, f_n = ω_n × s_n.

    From :func:`collective_diffusive_drift_cartesian`, f = s × (AY, −AX, 0) + B × s
    = (B − A) × s with A = (AY, −AX, 0), B = (Ω, 0, Δ); so
        ω_n = (Ω − AY_n, AX_n, Δ),   AX_n = ½ Σ_m Γ_mn s_x^m,  AY_n = ½ Σ_m Γ_mn s_y^m.
    For the Dicke Γ the axis is the same for every spin (rigid rotation of the whole
    configuration, |Σ_n s_n| conserved).  Shape (N, n_traj, 3).
    """
    sx, sy = s[..., 0], s[..., 1]
    AX = 0.5 * (G @ (G.T @ sx))
    AY = 0.5 * (G @ (G.T @ sy))
    return np.stack([Omega - AY, AX, np.full_like(AX, Delta)], axis=-1)


def _axis_half_rotation(s: np.ndarray, w: np.ndarray, half_dt: float) -> np.ndarray:
    """Exact rotation of s about the axis w by |w|·half_dt (solves ṡ = w × s exactly).

    Unlike :func:`_half_rotation`, which rotates about the *tangential projection*
    w − ŝ(ŝ·w) and therefore moves s along a great circle instead of the small circle
    around w, this has no O(dt²) drift away from the axis.  That spurious drift is
    ∝ |w|² dt ∝ Ω·dt_factor, i.e. it grows with N at fixed dt_factor (measured:
    +1.6°/unit time at N=256, dt_factor=1e-3), which biased the N-dependence of the QFI.
    """
    wn = np.linalg.norm(w, axis=-1)
    wn_safe = np.where(wn > 0.0, wn, 1.0)
    return _rotate_axis(s, w / wn_safe[..., None], half_dt * wn)


def collective_diffusive_rotation_step(
    s: np.ndarray,
    dt: float,
    Omega: float,
    Delta: float,
    kappa: float,
    rng: np.random.Generator,
    G: np.ndarray,
    rho2_floor: float = 1e-12,
    stratonovich: bool = True,
) -> np.ndarray:
    """Pole-safe Strang step for the diffusion-completed collective BTC ensemble.

    Same A) half-drift / B) noise / C) half-drift skeleton as
    :func:`collective_rotation_step`, but the noise is the *collective* decay noise
    (exact lab-axis rotations from Γ = G Gᵀ) instead of the per-spin decay noise.
    The rotations realise the **Stratonovich** SDE, so the drift is the matching
    Stratonovich one, :func:`collective_diffusive_drift_cartesian` — the full-Γ
    collective field plus the coherent precession, with no further Itô conversion
    term in either chart.

    Parameters
    ----------
    s     : Bloch vectors, shape (N, n_traj, 3), on the radius-√3 sphere
    dt    : time step
    Omega, Delta : drive and detuning
    rng   : NumPy random generator
    G     : (N, R) noise factor of the decay matrix, Γ = G Gᵀ (see
            :func:`collective_noise_factor`) — it sets both the noise and the
            collective drift field, so the decay rate enters only through it
    kappa, rho2_floor, stratonovich : accepted for call-site compatibility but
            unused — the corrected drift needs no rate argument and no
            Itô→Stratonovich correction (adding one double-counts it)
    """
    # Exact rotation about the true axis ω = B − A (fix 2026-08-28; the previous
    # _half_rotation about the tangential projection had an O(Ω·dt_factor) spurious
    # drift away from the axis, growing with N at fixed dt_factor).
    s = _axis_half_rotation(s, collective_diffusive_axis(s, Omega, Delta, G), 0.5 * dt)
    s = _collective_noise_rotations(s, dt, rng, G)
    s = _axis_half_rotation(s, collective_diffusive_axis(s, Omega, Delta, G), 0.5 * dt)
    return s


def collective_diffusive_em_step(
    theta: np.ndarray,
    phi: np.ndarray,
    dt: float,
    Omega: float,
    Delta: float,
    kappa: float,
    rng: np.random.Generator,
    G: np.ndarray,
    theta_eps: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """(θ,φ) Euler–Maruyama step for the diffusion-completed collective ensemble.

    Itô integrator (no Stratonovich correction) of the same drift as
    :func:`collective_euler_maruyama_step` plus the collective decay noise written
    in the angle chart,

        dθₙ ⊃ Σ_c G[n,c] ( sin φₙ dW^x_c − cos φₙ dW^y_c ),
        dφₙ ⊃ Σ_c G[n,c] cot θₙ ( cos φₙ dW^x_c + sin φₙ dW^y_c ),

    which is the chart image of the lab-axis rotation noise.  The cot θ factor makes
    this step blow up near the poles — it exists only as an independent cross-check
    of the pole-safe :func:`collective_diffusive_rotation_step`.  ``theta`` and
    ``phi`` have shape ``(N, n_traj)``; ``G`` is the ``(N, R)`` factor of Γ.
    """
    sin_t = np.sin(theta)
    cos_p = np.cos(phi)
    sin_p = np.sin(phi)
    cot_t = np.cos(theta) / sin_t

    # Deterministic drift: drive + detuning + (exact) single-spin decay …
    dtheta, dphi = drift(theta, phi, Omega, Delta, kappa)
    # … plus the off-diagonal collective superradiant drift (self excluded).
    sx = SQRT3 * sin_t * cos_p
    sy = SQRT3 * sin_t * sin_p
    Mx = sx.sum(axis=0, keepdims=True) - sx
    My = sy.sum(axis=0, keepdims=True) - sy
    dtheta = dtheta - 0.5 * kappa * (Mx * cos_p + My * sin_p)
    dphi = dphi + 0.5 * kappa * cot_t * (Mx * sin_p - My * cos_p)

    # Collective decay noise (Itô, chart form).
    n_traj = theta.shape[1]
    R = G.shape[1]
    sdt = np.sqrt(dt)
    ax = G @ rng.normal(0.0, sdt, size=(R, n_traj))   # (N, n_traj) effective dW^x
    ay = G @ rng.normal(0.0, sdt, size=(R, n_traj))   # (N, n_traj) effective dW^y
    dtheta_noise = sin_p * ax - cos_p * ay
    dphi_noise = cot_t * (cos_p * ax + sin_p * ay)

    theta_next = theta + dtheta * dt + dtheta_noise
    phi_next = phi + dphi * dt + dphi_noise

    np.clip(theta_next, theta_eps, np.pi - theta_eps, out=theta_next)
    return theta_next, phi_next


COLLECTIVE_SCHEMES = ("drift", "diffusive", "diffusive_em")


def collective_noise_factor(
    N: int,
    kappa: float,
    Gamma: np.ndarray | None = None,
) -> np.ndarray:
    """Noise factor ``G`` (Γ = G Gᵀ) for the diffusion-completed collective schemes.

    With ``Gamma=None`` returns the rank-1 Dicke factor ``√κ · 1`` of shape
    ``(N, 1)`` directly (no eigendecomposition).  Otherwise factorises the supplied
    ``(N, N)`` PSD decay matrix via :func:`factorize_decay_matrix` — the hook for
    future spatially-extended systems (Γ from the Green's tensor).
    """
    if Gamma is None:
        return np.sqrt(float(kappa)) * np.ones((int(N), 1))
    return factorize_decay_matrix(np.asarray(Gamma, dtype=float))


def _sample_collective_initial(
    count, rng, sampling, state, theta0, phi0, spread, S
) -> tuple[np.ndarray, np.ndarray]:
    """Initial (θ, φ) for the collective ensemble, dispatched on ``sampling``."""
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


def make_collective_stepper(scheme, Omega, Delta, kappa, G=None):
    """Return ``step(s, dt, rng) -> s`` advancing the Cartesian state one step.

    ``scheme`` selects the variant: ``"drift"`` (drift-only, the original
    mean-field scheme), ``"diffusive"`` (pole-safe Cartesian rotation with the
    collective noise), or ``"diffusive_em"`` ((θ,φ) Euler–Maruyama cross-check).
    ``G`` (the Γ = G Gᵀ factor) is required for the diffusive variants.
    """
    if scheme == "drift":
        return lambda s, dt, rng: collective_rotation_step(s, dt, Omega, Delta, kappa, rng)
    if scheme == "diffusive":
        if G is None:
            raise ValueError("scheme 'diffusive' requires the noise factor G.")
        return lambda s, dt, rng: collective_diffusive_rotation_step(
            s, dt, Omega, Delta, kappa, rng, G
        )
    if scheme == "diffusive_em":
        if G is None:
            raise ValueError("scheme 'diffusive_em' requires the noise factor G.")

        def step(s, dt, rng):
            th, ph = spin_to_angles(s[..., 0], s[..., 1], s[..., 2])
            th, ph = collective_diffusive_em_step(th, ph, dt, Omega, Delta, kappa, rng, G)
            return np.stack(angles_to_spin(th, ph), axis=-1)

        return step
    raise ValueError(
        f"Unknown collective scheme '{scheme}'. Choose one of {COLLECTIVE_SCHEMES}."
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
    scheme: str = "drift",
    Gamma: np.ndarray | None = None,
    progress: bool = False,
) -> dict:
    """Integrate the collective BTC DCTWA ensemble of ``N`` coupled spins.

    Simulates ``n_traj`` independent realisations, each of ``N`` permutationally
    symmetric spin-½'s with Hamiltonian H = Ω Sₓ (+ detuning) and collective
    decay √κ S₋.  Returns, for each realisation, the **normalised collective
    magnetisation**

        mₐ^(k)(t) = (1/N) Σ_{n=1}^N sₐ^{n,k}(t)  =  (⟨Sₐ⟩/S) of realisation k,

    so that averaging over realisations (axis 1) yields the collective ⟨Sₐ⟩/S(t).
    For ``N = 1`` this reduces to the isolated single-spin model of
    :func:`run_trajectories`.

    Parameters
    ----------
    N        : number of spins (collective spin S = N/2); positive integer
    Omega    : Rabi (drive) frequency Ω  (H = Ω Sₓ)
    Delta    : detuning Δ
    kappa    : single-particle decay rate κ  (collective rate g = κS = κN/2)
    times    : 1-D array with *uniform* spacing
    n_traj   : number of independent collective realisations
    seed     : random seed for reproducibility
    sampling : ``"dtwa"`` (discrete spin-½ sampling of ``state``) or
               ``"continuous"`` (delta/Gaussian start at ``theta0, phi0``)
    state    : ``"up"`` / ``"down"`` initial polarisation for DTWA / cone sampling
    theta0,
    phi0     : mean initial angles for ``"continuous"`` sampling
    spread   : Gaussian spread of the ``"continuous"`` initial condition,
               scaled internally by 1/√(2S)
    scheme   : collective integrator – ``"drift"`` (default, drift-only mean field),
               ``"diffusive"`` (pole-safe Cartesian rotation with the collective
               decay noise of Γ = G Gᵀ), or ``"diffusive_em"`` ((θ,φ)
               Euler–Maruyama cross-check, pole-prone).  The diffusive schemes add
               the missing collective noise needed for quantitative second moments.
    Gamma    : optional ``(N, N)`` PSD decay matrix for the diffusive schemes; if
               ``None`` (default) the all-to-all Dicke Γ_{mn} = κ is used.
    progress : show a tqdm progress bar over the time steps

    Returns
    -------
    dict with keys (each magnetisation array has shape ``(n_time, n_traj)``):
        ``'mx'``, ``'my'``, ``'mz'`` – normalised collective magnetisation
                                       ⟨Sₐ⟩/S per realisation
        ``'times'`` – the input time array
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

    G = None if scheme == "drift" else collective_noise_factor(N, kappa, Gamma)
    step = make_collective_stepper(scheme, Omega, Delta, kappa, G)

    rng = np.random.default_rng(seed)
    theta, phi = _sample_collective_initial(
        N * n_traj, rng, sampling, state, theta0, phi0, spread, S
    )
    theta = theta.reshape(N, n_traj)
    phi = phi.reshape(N, n_traj)

    # Cartesian state (N, n_traj, 3) for the pole-safe rotation solver; the
    # normalised collective magnetisation is the per-spin mean (1/N) Σₙ sₐⁿ.
    s = np.stack(angles_to_spin(theta, phi), axis=-1)   # (N, n_traj, 3)

    mx = np.empty((n_time, n_traj))
    my = np.empty((n_time, n_traj))
    mz = np.empty((n_time, n_traj))
    mx[0], my[0], mz[0] = s[..., 0].mean(axis=0), s[..., 1].mean(axis=0), s[..., 2].mean(axis=0)

    step_iter = range(n_time - 1)
    if progress:
        from tqdm.auto import tqdm

        step_iter = tqdm(step_iter, desc=f"collective BTC SDE ({scheme})")

    for n in step_iter:
        s = step(s, dt, rng)
        mx[n + 1], my[n + 1], mz[n + 1] = (
            s[..., 0].mean(axis=0),
            s[..., 1].mean(axis=0),
            s[..., 2].mean(axis=0),
        )

    return {"mx": mx, "my": my, "mz": mz, "times": times}
