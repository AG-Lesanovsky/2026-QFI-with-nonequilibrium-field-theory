"""Reduced (rigid-body) form of the corrected diffusive collective TWA (btc/sde.py, post-Aug-22),
numba-compiled.

Diffusive step there = half drift / noise / half drift with
    drift  f_n = (B - A) x s_n,  B = (Omega,0,0),  A = (kappa/2)(M_y, -M_x, 0),  M = sum_n s_n
    noise  global rotations about x by sqrt(kappa) dW_x and about y by sqrt(kappa) dW_y.
Every sub-step is a rotation about an axis COMMON to all spins, so the configuration moves
rigidly and M(t) obeys a closed SDE.  This integrates M directly with EXACT rotations about the
common axis and stores per-trajectory X(t) = int_0^t M_x/N dt' at selected times (F = 4 Var[X]).
Verified identical (1e-13) to the fixed full-N stepper with the same noise stream.
Usage: python reduced_btc.py N dt_factor n_traj state seed outdir
"""
import os, sys, time, math, numpy as np
sys.path.insert(0, os.path.expanduser("~/BTC_cluster"))
from numba import njit

@njit(cache=True, fastmath=False)
def _rot_axis(vx, vy, vz, wx, wy, wz, ang):
    wn = math.sqrt(wx*wx + wy*wy + wz*wz)
    if wn == 0.0:
        return vx, vy, vz
    ux, uy, uz = wx/wn, wy/wn, wz/wn
    c = math.cos(ang); s = math.sin(ang); d = (1.0 - c) * (ux*vx + uy*vy + uz*vz)
    cx = uy*vz - uz*vy; cy = uz*vx - ux*vz; cz = ux*vy - uy*vx
    return c*vx + s*cx + d*ux, c*vy + s*cy + d*uy, c*vz + s*cz + d*uz

@njit(cache=True, fastmath=False)
def _integrate(M, N, Omega, kappa, dt, nsteps, t_out_steps, noise, X_out, seed):
    np.random.seed(seed)
    n = M.shape[0]; sdt = math.sqrt(dt); sk = math.sqrt(kappa); hdt = 0.5 * dt
    X = np.zeros(n); mxp = np.empty(n)
    for j in range(n):
        mxp[j] = M[j, 0] / N
    k = 0
    for i in range(1, nsteps + 1):
        for j in range(n):
            x, y, z = M[j, 0], M[j, 1], M[j, 2]
            # half drift: rotate about omega = (Omega - k/2 M_y, k/2 M_x, 0) by |omega| dt/2
            wx = Omega - 0.5 * kappa * y; wy = 0.5 * kappa * x
            wn = math.sqrt(wx*wx + wy*wy)
            x, y, z = _rot_axis(x, y, z, wx, wy, 0.0, wn * hdt)
            # noise: x-rot by a/2, y-rot by b, x-rot by a/2
            a = sk * sdt * np.random.standard_normal(); b = sk * sdt * np.random.standard_normal()
            ca = math.cos(0.5*a); sa = math.sin(0.5*a); cb = math.cos(b); sb = math.sin(b)
            y, z = ca*y - sa*z, sa*y + ca*z
            x, z = cb*x + sb*z, -sb*x + cb*z
            y, z = ca*y - sa*z, sa*y + ca*z
            # half drift at the new configuration
            wx = Omega - 0.5 * kappa * y; wy = 0.5 * kappa * x
            wn = math.sqrt(wx*wx + wy*wy)
            x, y, z = _rot_axis(x, y, z, wx, wy, 0.0, wn * hdt)
            M[j, 0], M[j, 1], M[j, 2] = x, y, z
            mx = x / N
            X[j] += 0.5 * (mx + mxp[j]) * dt; mxp[j] = mx
        if k < t_out_steps.shape[0] and i >= t_out_steps[k]:
            for j in range(n):
                X_out[k, j] = X[j]
            k += 1
    return X

def sample_M0(N, n_traj, state, rng):
    from btc import sde
    th, ph = sde.sample_initial_conditions_wigner_cone(N * n_traj, rng, state)
    sx, sy, sz = sde.angles_to_spin(th, ph)
    return np.ascontiguousarray(np.stack([sx, sy, sz], -1).reshape(N, n_traj, 3).sum(0))

def run(N, dtf, n_traj, state, seed, T=100.0, t_out=None):
    if t_out is None: t_out = np.linspace(0.5, T, 200)
    S = N / 2; Omega = 2.0 * S; kappa = 1.0
    dt = dtf / Omega; nsteps = int(round(T / dt))
    rng = np.random.default_rng([seed, N, int(round(1 / dtf))])
    M = sample_M0(N, n_traj, state, rng)
    L0 = np.linalg.norm(M, axis=-1)
    t_out = np.array(t_out, dtype=float); t_steps = np.rint(t_out / dt).astype(np.int64)
    X_out = np.zeros((len(t_out), n_traj))
    _integrate(M, float(N), Omega, kappa, dt, nsteps, t_steps, None, X_out, int(rng.integers(2**31 - 1)))
    return dict(zip(t_out, X_out)), L0, np.linalg.norm(M, axis=-1)

if __name__ == "__main__":
    N = int(sys.argv[1]); dtf = float(sys.argv[2]); n = int(sys.argv[3]); state = sys.argv[4]
    seed = int(sys.argv[5]); outdir = sys.argv[6]; os.makedirs(outdir, exist_ok=True)
    t0 = time.time(); Xs, L0, L1 = run(N, dtf, n, state, seed)
    ts = np.array(sorted(Xs)); Xarr = np.stack([Xs[t] for t in ts])
    np.savez(f"{outdir}/red_N{N}_dt{dtf:g}_{state}_seed{seed}.npz", t=ts, X=Xarr, L0=L0, L1=L1,
             N=N, dt_factor=dtf, state=state, seed=seed)
    print(f"N={N} dt_factor={dtf:g} n={n} state={state} seed={seed}: |M|/N={L0.mean()/N:.4f}, "
          f"max|Delta|M||={np.abs(L1-L0).max():.1e}, {time.time()-t0:.0f}s", flush=True)
    for t, x in zip(ts, Xarr): print(f"   t={t:5.1f}  F={4*x.var(ddof=1):8.3f}")
