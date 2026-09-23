import glob, sys, numpy as np
d, out = sys.argv[1], sys.argv[2]
for N in (2,4,8,16,32,64,128):
    fs = sorted(glob.glob(f"{d}/red_N{N}_dt0.01_up_seed*.npz"))
    if not fs: continue
    t = np.load(fs[0])["t"]; X = np.concatenate([np.load(f)["X"] for f in fs], axis=1); n = X.shape[1]
    v = X.var(1, ddof=1); m4 = ((X - X.mean(1, keepdims=True))**4).mean(1); F = 4*v; e = 4*np.sqrt(np.maximum(m4 - v*v, 0)/n)
    np.savetxt(f"{out}/twa_fixed_N{N}_dt1e-2_up_100k.csv", np.c_[np.r_[0,t], np.r_[0,F], np.r_[0,e], np.full(len(t)+1, n)],
               delimiter=",", header="t,qfi,stderr,ntraj", comments="", fmt="%.6f,%.6f,%.6f,%d")
    print(f"N={N:3d}: {len(fs):3d} seeds {n:6d} traj  F(100)={F[-1]:8.3f} ± {e[-1]:.3f}  ({100*e[-1]/F[-1]:.2f}%)")
