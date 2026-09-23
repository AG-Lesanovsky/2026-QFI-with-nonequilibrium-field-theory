#!/usr/bin/env python3
"""Deterministic monitored-QFI reference, one standalone det_N<N>.npz per N.

For each N, computes F_Q(t) exactly with btc.quantum_monitored_qfi_omega. The
solver is single-threaded, so this is meant for a local machine. Requires QuTiP.

The tqdm progress bar (disable with --no-progress) advances once per unit of
simulated time, and det_N<N>.ckpt.npz is rewritten after each, so a crashed or
killed run resumes from the last completed step just by re-running the same
command (the checkpoint is deleted once det_N<N>.npz is written). This matters
for the largest N, where a single solve can run for days.

After the run(s), a quick check plot (det_monitored_qfi.png) of F_Q(t)/S² and
its rate F_Q(t)/(S²·t) is written, one line per N; disable with --no-plot or
pop it up interactively with --show.

Usage:
    python det_run.py N [N ...] [options]
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import btc  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Deterministic monitored-QFI reference per N.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("N", type=int, nargs="+", help="system size(s); one .npz per N")
    p.add_argument("--kappa", type=float, default=1.0, help="decay rate κ")
    p.add_argument("--ratio", type=float, default=2.0, help="sets Ω = ratio·κ·N/2")
    p.add_argument("--T", type=float, default=100.0, help="final time")
    p.add_argument("--n-store", type=int, default=200,
                   help="time points kept per N")
    p.add_argument("--outdir", type=str, default=os.getcwd(),
                   help="output directory (default: current dir)")
    p.add_argument("--overwrite", action="store_true",
                   help="recompute even if det_N<N>.npz exists")
    p.add_argument("--no-progress", action="store_true",
                   help="disable the tqdm progress bar")
    p.add_argument("--no-plot", action="store_true",
                   help="skip the quick check plot written after the run(s)")
    p.add_argument("--show", action="store_true",
                   help="display the check plot interactively (default: save PNG only)")
    return p.parse_args(argv)


def _load_result(outfile: str) -> dict:
    """Load a saved det_N<N>.npz into the light-weight dict `plot_results` wants."""
    d = np.load(outfile)
    return {"N": int(d["N"]), "S": float(d["S"]), "times": d["times"],
            "qfi": d["qfi"]}


def run_one_N(N: int, args: argparse.Namespace, outdir: str) -> dict:
    outfile = os.path.join(outdir, f"det_N{N}.npz")
    if os.path.exists(outfile) and not args.overwrite:
        print(f"[skip] det_N{N}.npz already exists (use --overwrite to redo)", flush=True)
        return _load_result(outfile)

    S = N / 2.0
    Omega = args.ratio * args.kappa * S

    print(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] "
          f"N={N} (S={S:g}): Ω={Omega:g}", flush=True)

    # Checkpoint for the long, single-threaded solve.  It advances the tqdm bar
    # once per unit of simulated time and rewrites this file after each, so a
    # crash resumes from the last completed step instead of restarting.
    ckpt = os.path.join(outdir, f"det_N{N}.ckpt.npz")

    times = np.linspace(0.0, args.T, args.n_store)
    out = btc.quantum_monitored_qfi_omega(
        S, Omega, args.kappa, times,
        progress=not args.no_progress,
        checkpoint_path=ckpt,
    )
    qfi = out["qfi"]
    dt = float(times[1] - times[0])

    # `qfi_stderr`, `method` and `n_traj` are constant now that the MCWF engine
    # is gone; they are still written so the on-disk schema matches the det_N*.npz
    # files already in deterministic_qfi/.
    np.savez(
        outfile,
        times=times, qfi=qfi,                          # qfi = F_Q (unnormalised)
        qfi_stderr=np.zeros_like(qfi),
        N=N, S=S, Omega=Omega, kappa=args.kappa, ratio=args.ratio, T=args.T,
        dt=dt, n_store=args.n_store, method="exact", n_traj=0,
        engine_kind="deterministic_monitored",
    )
    # Final result is safely on disk; drop the resume checkpoint if any.
    if os.path.exists(ckpt):
        os.remove(ckpt)
    j = int(np.argmin(np.abs(times - args.T)))
    print(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] "
          f"N={N}: saved det_N{N}.npz  "
          f"(F_Q/(T·S²) at t=T = {qfi[j] / (args.T * S**2):.4f})",
          flush=True)
    return {"N": N, "S": S, "times": times, "qfi": qfi}


def plot_results(results: list, outdir: str, show: bool = False) -> None:
    """Quick sanity plot of the monitored QFI after the run(s) finish.

    Two panels vs time, one viridis line per N (dark = small N → yellow = large):
    the normalised QFI F_Q(t)/S² (comparable across N and to the semiclassical
    F_sc) and its rate F_Q(t)/(S²·t) — the quantity the end-of-run print reports
    at t=T, so a glance confirms the curve is smooth, positive, and plateauing as
    expected.  Saves det_monitored_qfi.png; never raises if matplotlib is missing.
    """
    try:
        import matplotlib
        if not show:
            matplotlib.use("Agg")   # headless: save without a display
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plot skipped] matplotlib unavailable: {exc}", flush=True)
        return

    results = sorted(results, key=lambda r: r["N"])
    colors = plt.cm.viridis(np.linspace(0.12, 0.88, len(results)))
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.0, 4.2))
    for r, c in zip(results, colors):
        t = np.asarray(r["times"], dtype=float)
        S = float(r["S"])
        q = np.asarray(r["qfi"], dtype=float) / S**2
        lbl = f"N={r['N']}"

        ax0.plot(t, q, color=c, lw=1.6, label=lbl)
        m = t > 0                         # rate F_Q/(S² t) is undefined at t=0
        ax1.plot(t[m], q[m] / t[m], color=c, lw=1.6, label=lbl)

    ax0.set_ylabel(r"monitored QFI  $F_Q(t)/S^2$")
    ax1.set_ylabel(r"QFI rate  $F_Q(t)/(S^2\,t)$")
    for ax in (ax0, ax1):
        ax.set_xlabel(r"time $t$")
        ax.margins(x=0)
        ax.grid(True, alpha=0.3)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    if len(results) > 1:
        ax0.legend(frameon=False, fontsize=8)
    else:
        fig.suptitle(f"N={results[0]['N']}", fontsize=10)
    fig.tight_layout()
    out = os.path.join(outdir, "det_monitored_qfi.png")
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
    results = []
    for N in args.N:
        if N < 1:
            raise ValueError(f"N must be positive, got {N}")
        results.append(run_one_N(N, args, outdir))
    print(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] all done "
          f"({len(args.N)} value(s) of N).", flush=True)
    if results and not args.no_plot:
        plot_results(results, outdir, show=args.show)


if __name__ == "__main__":
    main()
