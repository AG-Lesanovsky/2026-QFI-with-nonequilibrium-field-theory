"""Pool reduced-model seeds -> F(t) = 4 Var[X] with stderr, and compare to exact det_N*.npz."""
import glob, os, sys, numpy as np
d = sys.argv[1]; ex_dir = sys.argv[2] if len(sys.argv) > 2 else None
groups = {}
for f in glob.glob(f"{d}/red_N*_seed*.npz"):
    z = np.load(f); key = (int(z["N"]), float(z["dt_factor"]), str(z["state"]))
    groups.setdefault(key, []).append((z["t"], z["X"]))
for key in sorted(groups):
    N, dtf, state = key; t = groups[key][0][0]; X = np.concatenate([x for _, x in groups[key]], axis=1)
    n = X.shape[1]; v = X.var(axis=1, ddof=1); m4 = ((X - X.mean(1, keepdims=True)) ** 4).mean(1)
    F = 4 * v; e = 4 * np.sqrt(np.maximum(m4 - v * v, 0) / n)
    ex = None
    if ex_dir and os.path.exists(f"{ex_dir}/det_N{N}_exact.npz"):
        z = np.load(f"{ex_dir}/det_N{N}_exact.npz"); ex = np.interp(t, z["times"], z["qfi"] / (N / 2) ** 2)
    print(f"== N={N} dt_factor={dtf:g} state={state} n_traj={n} ({len(groups[key])} seeds)")
    for i, tt in enumerate(t):
        line = f"   t={tt:5.1f}  F={F[i]:8.3f} ± {e[i]:.3f}"
        if ex is not None: line += f"   exact {ex[i]:8.3f}   diff {100*(F[i]-ex[i])/ex[i]:+.1f}%"
        print(line)
