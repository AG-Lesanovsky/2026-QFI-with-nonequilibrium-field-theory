"""Semiclassical dynamics of a driven atomic array. Integrates the collective
 SDEs of Mink & Fleischhauer (arXiv:2305.19829) in Cartesian coordinates. 
 Cartesian EoM obtained after transition from Ito to Stratonovich in spherical
 coordinates.

 The whole drift is a precession, f = w x s, with the closed-form axis of
 :func:`array_axis`; the deterministic sub-steps rotate about that axis
 (:func:`_axis_half_rotation`) rather than about the tangential projection
 (s x f)/|s|^2.  Same pattern as ``btc.sde.collective_diffusive_axis`` /
 ``btc.sde._axis_half_rotation``.  The axis is sampled at the sub-step midpoint
 (:func:`_axis_midpoint_rotation`), which makes the deterministic part second
 order; the drift/noise splitting itself stays first order.  The noise sub-step is a
 single exact rotation (:func:`_collective_noise`).
"""

from __future__ import annotations

import numpy as np

SQRT3 = np.sqrt(3.0)
R2 = 3.0  


# -- Rotation primitives -------------------------------------------------------

def _rotate_x(s, angle):
    c, sn = np.cos(angle), np.sin(angle)
    sy = c * s[..., 1] - sn * s[..., 2]
    sz = sn * s[..., 1] + c * s[..., 2]
    return np.stack([s[..., 0], sy, sz], axis=-1)


def _rotate_y(s, angle):
    c, sn = np.cos(angle), np.sin(angle)
    sx = c * s[..., 0] + sn * s[..., 2]
    sz = -sn * s[..., 0] + c * s[..., 2]
    return np.stack([sx, s[..., 1], sz], axis=-1)


def _rotate_axis(s, axis, angle):
    """Rodrigues rotation of s about unit axis by angle."""
    c = np.cos(angle)[..., None]
    sn = np.sin(angle)[..., None]
    nds = np.sum(axis * s, axis=-1)[..., None]
    return c * s + sn * np.cross(axis, s) + (1.0 - c) * nds * axis


def _half_rotation(s, f, half_dt):
    """Rotate s about w = (s x f)/|s|^2 by |w|·half_dt (identity where w=0).

    Legacy.  This rotates about the *tangential projection* of the precession axis,
    so s follows a great circle instead of the small circle around the true axis --
    an O(dt^2) error per step even when the true axis is constant.  Superseded in
    :func:`array_step` by :func:`_axis_half_rotation` + :func:`array_axis`; kept for
    reproducing pre-2026-08-29 runs, and for any drift with no closed-form axis.
    """
    w = np.cross(s, f) / R2
    wn = np.linalg.norm(w, axis=-1)
    wn_safe = np.where(wn > 0.0, wn, 1.0)
    return _rotate_axis(s, w / wn_safe[..., None], half_dt * wn)


def _axis_half_rotation(s, w, half_dt):
    """Rotate s about the axis w by |w|·half_dt (identity where w=0).

    Solves ds/dt = w x s *exactly* for frozen w, so -- unlike :func:`_half_rotation`
    -- there is no spurious drift away from the axis.  Ported from
    ``btc.sde._axis_half_rotation``.
    """
    wn = np.linalg.norm(w, axis=-1)
    wn_safe = np.where(wn > 0.0, wn, 1.0)
    return _rotate_axis(s, w / wn_safe[..., None], half_dt * wn)


# -- Initial-condition sampling ------------------------------------------------

def _transverse_frame(n_hat):
    ref = np.array([1.0, 0.0, 0.0]) if abs(n_hat[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(n_hat, ref)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(n_hat, e1)


def sample_cone_cartesian(count, rng, n_hat=(0.0, 0.0, -1.0)):
    """Wigner-cone samples of a spin-1/2 coherent state along n_hat.

    Returns s of shape (count, 3) on the sqrt(3)-sphere: a fixed magic-angle
    tilt (cos g = 1/sqrt(3)) from n_hat with uniform azimuth, so avg(s) = n_hat.
    """
    n_hat = np.asarray(n_hat, dtype=float)
    n_hat = n_hat / np.linalg.norm(n_hat)
    e1, e2 = _transverse_frame(n_hat)
    cos_g = 1.0 / SQRT3
    sin_g = np.sqrt(1.0 - cos_g * cos_g)
    alpha = rng.uniform(0.0, 2.0 * np.pi, size=count)
    transverse = np.cos(alpha)[:, None] * e1 + np.sin(alpha)[:, None] * e2
    return SQRT3 * (cos_g * n_hat + sin_g * transverse)


# -- Drift and diffusion -------------------------------------------------------

def array_drift(s, Omega, Delta, Gamma, J):
    """Cartesian Stratonovich drift of the array ensemble.

    ----------------------------------------------------------------------------
    Inputs:
    -----------
    s     : (N, n_traj, 3) array of spin vectors 
    Omega : Rabi frequency of the coherent drive
    Delta : Detuning of the coherent drive
    Gamma : Collective decay rates
    J     : Collective interaction strengths

    ----------------------------------------------------------------------------
    Returns:
    -----------
    f     : (N, n_traj, 3) array of drift vectors for each spin in the ensemble
    """
    sx, sy, sz = s[..., 0], s[..., 1], s[..., 2]

    # Collective transverse fields (matrix–vector over the spin index; Gamma, 
    # J symmetric).
    AX = 0.5 * (Gamma @ sx) + (J @ sy)
    AY = 0.5 * (Gamma @ sy) - (J @ sx)
    # s x W  with  W = (AY, −AX, 0)
    fcx = sz * AX
    fcy = sz * AY
    fcz = -(sx * AX + sy * AY)

    # Coherent drive/detuning adds precession about the lab axes (x̂, ŷ, ẑ). 
    # Assumes Omega real.
    fcx = fcx + sy * Delta
    fcy = fcy + sz * (2.0 * Omega) - sx * Delta
    fcz = fcz - sy * (2.0 * Omega)

    return np.stack([fcx, fcy, fcz], axis=-1)


def array_axis(s, Omega, Delta, Gamma, J):
    """Precession axis w of the Stratonovich drift, so that f = w x s exactly.

    Every term of :func:`array_drift` -- the collective Gamma/J field, the drive and
    the detuning -- is a cross product with s, so the drift is a pure precession
    f = (B - A) x s with A = (AY, -AX, 0) and B = (-2 Omega, 0, -Delta):

        w_n = (-AY_n - 2 Omega,  AX_n,  -Delta),
        AX_n = 1/2 (Gamma s_x)_n + (J s_y)_n,
        AY_n = 1/2 (Gamma s_y)_n - (J s_x)_n.

    Rotating about w (see :func:`_axis_half_rotation`) is exact for frozen w, whereas
    rotating about the tangential projection (s x f)/|s|^2 discards the radial part of
    w -- here ~48% of |w| on average -- and traces the wrong circle.  Same construction
    as ``btc.sde.collective_diffusive_axis``.

    ----------------------------------------------------------------------------
    Inputs:
    -----------
    s     : (N, n_traj, 3) array of spin vectors
    Omega : Rabi frequency of the coherent drive
    Delta : Detuning of the coherent drive
    Gamma : Collective decay rates
    J     : Collective interaction strengths

    ----------------------------------------------------------------------------
    Returns:
    -----------
    w     : (N, n_traj, 3) precession axes; np.cross(w, s) reproduces array_drift
    """
    sx, sy = s[..., 0], s[..., 1]

    # Collective transverse fields (matrix–vector over the spin index; Gamma,
    # J symmetric).
    AX = 0.5 * (Gamma @ sx) + (J @ sy)
    AY = 0.5 * (Gamma @ sy) - (J @ sx)

    w = np.empty_like(s)
    w[..., 0] = -AY - 2.0 * Omega
    w[..., 1] = AX
    w[..., 2] = -Delta
    return w


def _axis_midpoint_rotation(s, half_dt, Omega, Delta, Gamma, J):
    """Rotate s over `half_dt` about the precession axis sampled at the sub-step midpoint.

    Explicit midpoint on the rotation group.  Freezing the axis at the *start* of the
    sub-step (:func:`_axis_half_rotation` alone) leaves an O(dt^2) local error and makes
    :func:`array_step` globally first order; sampling it at the midpoint instead gives
    O(dt^3) local, i.e. second order, for one extra :func:`array_axis` evaluation.

    The predictor is a plain Euler push, not a rotation: it only has to locate the
    midpoint to O(half_dt^2) because it is used solely to *evaluate the axis*.  It
    leaves the sphere by O(half_dt^2), which is harmless -- the state update below is
    still an exact rotation of the on-sphere s, so |s| = sqrt(3) is preserved to
    round-off (measured 1e-13 over 4000 steps).
    """
    w = array_axis(s, Omega, Delta, Gamma, J)
    s_mid = s + (0.5 * half_dt) * np.cross(w, s)
    return _axis_half_rotation(s, array_axis(s_mid, Omega, Delta, Gamma, J), half_dt)



def _collective_noise(s, dt, rng, G):
    """Collective decay noise as one exact rotation about the lab-plane axis (alpha, beta, 0).

    The noise generator is alpha*Lx + beta*Ly with Lx, Ly the generators of rotation
    about x-hat and y-hat, so its exponential is exactly the Rodrigues rotation about
    (alpha, beta, 0) by the angle sqrt(alpha^2 + beta^2).  The previous
    Rx(alpha/2) Ry(beta) Rx(alpha/2) sandwich was a Strang approximation *of that
    exponential*, carrying ~1e-3 of avoidable error per step at the typical angle
    sqrt(Gamma dt) ~ 0.1; the closed form below is exact to round-off and slightly
    cheaper (n_z = 0 kills half the Rodrigues terms).

    What remains -- and what no per-step rotation can remove -- is the Levy-area term:
    the exponential is the flow of the SDE driven by the piecewise-linear (Wong-Zakai)
    interpolant of the Wiener path, which converges to the Stratonovich solution but
    differs from it at O(dt) because Lx and Ly do not commute.  Simulating Levy areas
    is the only way past that.

    The two Wiener draws are unchanged in count, shape and order, so the RNG stream is
    identical to the sandwich version.
    """
    R = G.shape[1]
    if R == 0:
        return s
    n_traj = s.shape[1]
    sdt = np.sqrt(dt)
    alpha = G @ rng.normal(0.0, sdt, size=(R, n_traj))
    beta = G @ rng.normal(0.0, sdt, size=(R, n_traj))

    theta = np.hypot(alpha, beta)
    theta_safe = np.where(theta > 0.0, theta, 1.0)
    nx = alpha / theta_safe          # rotation axis (nx, ny, 0), unit where theta > 0
    ny = beta / theta_safe
    c = np.cos(theta)
    sn = np.sin(theta)
    omc = 1.0 - c

    sx, sy, sz = s[..., 0], s[..., 1], s[..., 2]
    nds = nx * sx + ny * sy          # n . s   (n_z = 0)
    return np.stack([
        c * sx + sn * (ny * sz) + omc * nds * nx,
        c * sy - sn * (nx * sz) + omc * nds * ny,
        c * sz + sn * (nx * sy - ny * sx),
    ], axis=-1)


def array_step(s, dt, Omega, Delta, Gamma, J, G, rng):
    """One pole-safe Strang step: 1/2 drift-rotation * noise * 1/2 drift-rotation.

    The deterministic half-steps rotate about the exact precession axis
    :func:`array_axis` (not the tangential projection of the drift), sampled at the
    midpoint of each half-step, so the deterministic part is second order.  The noise
    sub-step is unchanged, so the RNG stream is identical to the previous scheme.

    Note the *splitting* between drift and noise remains first order, so this is not a
    globally second-order weak scheme -- what becomes O(dt^2) is the deterministic
    bias, the part that does not average out over trajectories.
    """
    s = _axis_midpoint_rotation(s, 0.5 * dt, Omega, Delta, Gamma, J)
    s = _collective_noise(s, dt, rng, G)
    s = _axis_midpoint_rotation(s, 0.5 * dt, Omega, Delta, Gamma, J)
    return s


# -- Trajectory ensemble integrator (means, for validation) --------------------

def run_array_trajectories(
    positions,
    polarization,
    Omega: float,
    times: np.ndarray,
    n_traj: int,
    Delta: float = 0.0,
    Gamma0: float = 1.0,
    seed: int = 42,
    n_hat=(0.0, 0.0, -1.0),
    progress: bool = False,
) -> dict:
    """Integrate the array ensemble and return the mean magnetisations avg(S_a)/N.

    Returns a dict with mx, my, mz of shape (n_time,) (per-atom means,
    averaged over spins and trajectories) plus times.

    ----------------------------------------------------------------------------
    Inputs:
    -----------
    positions    : (N, 3) array of atom positions
    polarization : (3,) array of the atomic dipole polarization vector
    Omega        : Rabi frequency of the coherent drive
    times        : (n_time,) array of time points to integrate to
    n_traj       : number of trajectories to average over
    Delta        : Detuning of the coherent drive (default 0.0)
    Gamma0       : Single-atom decay rate (default 1.0)
    seed         : Random seed for reproducibility (default 42)
    n_hat        : Initial spin orientation (default (0, 0, -1))
    progress     : Show a progress bar (default False)

    ----------------------------------------------------------------------------
    Returns:
    -----------
    dict with keys 'mx', 'my', 'mz', 'times' containing the mean magnetisations and time points.

    """
    from .geometry import rate_matrices, factorize_decay_matrix

    times = np.asarray(times, dtype=float)
    n_time = len(times)
    dt = float(times[1] - times[0])
    N = np.asarray(positions).shape[0]

    Gamma, J = rate_matrices(positions, polarization, Gamma0)
    G = factorize_decay_matrix(Gamma)

    rng = np.random.default_rng(seed)
    s = sample_cone_cartesian(N * n_traj, rng, n_hat).reshape(N, n_traj, 3)

    mx = np.empty(n_time)
    my = np.empty(n_time)
    mz = np.empty(n_time)
    rec = lambda k: (
        s[..., 0].mean(), s[..., 1].mean(), s[..., 2].mean()
    )
    mx[0], my[0], mz[0] = rec(0)

    step_iter = range(1, n_time)
    if progress:
        from tqdm.auto import tqdm
        step_iter = tqdm(step_iter, desc="array SDE")
    for n in step_iter:
        s = array_step(s, dt, Omega, Delta, Gamma, J, G, rng)
        mx[n], my[n], mz[n] = rec(n)

    return {"mx": mx, "my": my, "mz": mz, "times": times}
