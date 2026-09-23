#!/usr/bin/env python3
"""Quantum monitored-QFI reference for the driven array, one mcwf_L<L>.npz per L.

For each lattice side L, computes F_mon(t) with one of two engines (--method):
  exact  atomarray.array_monitored_qfi_exact   — no MC / no dt bias; 4ᴺ cost, L≤3
  mcwf   atomarray.array_monitored_qfi_mcwf     — statevector MCWF, feasible to 4×4
'auto' uses exact for N ≤ --exact-threshold, MCWF above. This is the exact/quasi-
exact benchmark the semiclassical F_sc (sc_run_batches.py) is compared against;
run it on a local machine (small L only).  Requires only NumPy + SciPy.

A quick check plot (mcwf_monitored_qfi.png) of F_mon(t)/N is written after the
run(s); disable with --no-plot or show it with --show.

Usage:
    python mcwf_run.py L [L ...] [options]
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import atomarray as aa  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Quantum monitored-QFI reference (exact / MCWF) per lattice side L.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("L", type=int, nargs="+", help="lattice side(s); one .npz per L")
    p.add_argument("--a", type=float, default=0.8, help="lattice spacing (λe units)")
    p.add_argument("--omega", type=float, default=2.0, help="Rabi frequency Ω")
    p.add_argument("--gamma0", type=float, default=1.0, help="single-atom decay Γ0")
    p.add_argument("--delta", type=float, default=0.0, help="detuning Δ")
    p.add_argument("--T", type=float, default=10.0, help="final time")
    p.add_argument("--n-store", type=int, default=200, help="time points kept per L")
    p.add_argument("--method", choices=["auto", "exact", "mcwf"], default="auto",
                   help="reference engine; 'auto' picks by --exact-threshold")
    p.add_argument("--exact-threshold", type=int, default=9,
                   help="'auto': exact for N ≤ this, MCWF above")
    p.add_argument("--n-traj-mcwf", type=int, default=4000,
                   help="quantum-jump trajectories for the MCWF engine")
    p.add_argument("--dt-factor-mcwf", type=float, default=0.04,
                   help="fixed MCWF step dt = dt_factor_mcwf / max(2Ω, Γ0); the "
                        "collapse is applied at the interpolated crossing time, so "
                        "the unravelling bias is O(dt²) — 0.04 (Γ0·dt = 10⁻²) was "
                        "measured bias-free to ≲0.5%% at L=2, and beyond ~0.16 the "
                        "4th-order Taylor no-jump step starts to contribute")
    p.add_argument("--seed", type=int, default=42, help="MCWF RNG seed")
    p.add_argument("--n-jobs", type=int, default=-1,
                   help="MCWF worker processes (-1 = all); ignored by 'exact'")
    p.add_argument("--traj-block", type=int, default=None,
                   help="MCWF records propagated together per worker "
                        "(default: auto, ~256 MB of dense state per process); "
                        "lower it if memory-bound, raise it if not")
    p.add_argument("--outdir", type=str, default=os.getcwd(), help="output directory")
    p.add_argument("--overwrite", action="store_true",
                   help="recompute even if mcwf_L<L>.npz exists")
    p.add_argument("--no-progress", action="store_true", help="disable the progress bar")
    p.add_argument("--no-plot", action="store_true", help="skip the check plot")
    p.add_argument("--show", action="store_true", help="display the check plot")
    return p.parse_args(argv)


def _load(outfile: str) -> dict:
    d = np.load(outfile)
    return {"L": int(d["L"]), "N": int(d["N"]), "times": d["times"], "qfi": d["qfi"],
            "qfi_stderr": d["qfi_stderr"], "n_traj": int(d["n_traj"]),
            "method": str(d["method"])}


def run_one_L(L: int, args: argparse.Namespace, outdir: str) -> dict:
    outfile = os.path.join(outdir, f"mcwf_L{L}.npz")
    if os.path.exists(outfile) and not args.overwrite:
        print(f"[skip] mcwf_L{L}.npz already exists (use --overwrite to redo)", flush=True)
        return _load(outfile)

    N = L * L
    pos = aa.square_lattice(L, args.a)
    method = args.method
    if method == "auto":
        method = "exact" if N <= args.exact_threshold else "mcwf"
    print(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] "
          f"L={L} (N={N}): Ω={args.omega:g}, method={method}", flush=True)

    if method == "exact":
        times = np.linspace(0.0, args.T, args.n_store)
        out = aa.array_monitored_qfi_exact(
            pos, aa.EP_CIRCULAR_XY, args.omega, times,
            Delta=args.delta, Gamma0=args.gamma0, progress=not args.no_progress)
        qfi, qfi_stderr, n_traj = out["qfi"], out["qfi_stderr"], 0
        dt = float(times[1] - times[0])
    else:
        dt = args.dt_factor_mcwf / max(2.0 * args.omega, args.gamma0)
        tf = np.arange(0.0, args.T + dt, dt)
        out = aa.array_monitored_qfi_mcwf(
            pos, aa.EP_CIRCULAR_XY, args.omega, tf, n_traj=args.n_traj_mcwf,
            seed=args.seed, Delta=args.delta, Gamma0=args.gamma0,
            n_jobs=args.n_jobs, traj_block=args.traj_block,
            progress=not args.no_progress)
        idx = np.unique(np.linspace(0, len(tf) - 1, args.n_store).round().astype(np.int64))
        times, qfi = tf[idx], out["qfi"][idx]
        qfi_stderr, n_traj = out["qfi_stderr"][idx], int(out["n_traj"])

    np.savez(
        outfile, times=times, qfi=qfi, qfi_stderr=qfi_stderr,
        L=L, N=N, a=args.a, Omega=args.omega, gamma0=args.gamma0, delta=args.delta,
        T=args.T, dt=dt, n_store=args.n_store, method=method, n_traj=n_traj,
        engine_kind="quantum_monitored")
    j = int(np.argmin(np.abs(times - args.T)))
    print(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] L={L}: saved mcwf_L{L}.npz "
          f"(F_mon(T)/N = {qfi[j] / N:.3f}, method={method})", flush=True)
    return {"L": L, "N": N, "times": times, "qfi": qfi, "qfi_stderr": qfi_stderr,
            "n_traj": n_traj, "method": method}


def plot_results(results: list, outdir: str, show: bool = False) -> None:
    try:
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plot skipped] matplotlib unavailable: {exc}", flush=True)
        return
    results = sorted(results, key=lambda r: r["L"])
    colors = plt.cm.viridis(np.linspace(0.12, 0.88, len(results)))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for r, c in zip(results, colors):
        t, N = np.asarray(r["times"], float), r["N"]
        q = np.asarray(r["qfi"], float) / N
        e = np.asarray(r["qfi_stderr"], float) / N
        lbl = f"L={r['L']} (N={N})" + (f", n={r['n_traj']}" if r["n_traj"] else ", exact")
        ax.plot(t, q, color=c, lw=1.6, label=lbl)
        if np.any(e > 0):
            ax.fill_between(t, q - e, q + e, color=c, alpha=0.2, linewidth=0)
    ax.set_xlabel(r"time $\Gamma_0 t$")
    ax.set_ylabel(r"monitored QFI per atom  $F_\mathrm{mon}(t)/N$")
    ax.margins(x=0); ax.grid(True, alpha=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if len(results) > 1:
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out = os.path.join(outdir, "mcwf_monitored_qfi.png")
    fig.savefig(out, dpi=150)
    print(f"Wrote check plot   {out}", flush=True)
    if show:
        plt.show()
    plt.close(fig)


def main(argv=None) -> None:
    args = parse_args(argv)
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    print(f"Output directory: {outdir}", flush=True)
    results = [run_one_L(L, args, outdir) for L in args.L if L >= 1]
    print(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] all done "
          f"({len(args.L)} value(s) of L).", flush=True)
    if results and not args.no_plot:
        plot_results(results, outdir, show=args.show)


if __name__ == "__main__":
    main()
