import glob, sys, numpy as np
d, N, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
fs = sorted(glob.glob(f"{d}/red_N{N}_*_seed*.npz")); t = np.load(fs[0])["t"]
X = np.concatenate([np.load(f)["X"] for f in fs], axis=1); n = X.shape[1]
v = X.var(1, ddof=1); m4 = ((X - X.mean(1, keepdims=True))**4).mean(1)
F = 4*v; e = 4*np.sqrt(np.maximum(m4 - v*v, 0)/n)
rows = np.c_[np.r_[0, t], np.r_[0, F], np.r_[0, e], np.full(len(t)+1, n)]
np.savetxt(out, rows, delimiter=",", header="t,qfi,stderr,ntraj", comments="", fmt="%.6f,%.6f,%.6f,%d")
print(f"N={N}: {len(fs)} seeds, {n} traj -> {out}; F(t=100)={F[-1]:.3f}±{e[-1]:.3f}")
