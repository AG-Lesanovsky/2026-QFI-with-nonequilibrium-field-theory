#!/usr/bin/env python3
"""Combine sc_L*.npz power-sum slices into one file per lattice size L.

Groups the partial files (from sc_run_batches.py) by (L, master_seed,
batch_size), sums the additive power sums over their batch ranges, and writes
sc_array_L<L>.npz with keys: times, qfi, qfi_stderr, n_traj, n_batches_total,
L, N, a, Omega, delta, gamma0, T, dt, batch_size, master_seed, method.

Idempotent: re-run as new slices land. Errors on overlapping batch ranges
(double-count) and same-L campaigns that differ in (master_seed, batch_size);
warns on gaps. Prints n_traj and relative stderr at t=T per L.

Usage:
    python sc_aggregate.py [--dir DIR] [--out DIR] [--L L ...]
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atomarray import qfi_from_powersums  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Combine semiclassical power-sum checkpoints into per-L QFI files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dir", default=os.getcwd(),
                   help="directory holding the sc_L*.npz partial files")
    p.add_argument("--out", default=None,
                   help="output directory (default: same as --dir)")
    p.add_argument("--L", type=int, nargs="*", default=None,
                   help="restrict to these L (default: every L found)")
    p.add_argument("--no-plot", action="store_true", help="skip the QFI(t) plot")
    p.add_argument("--show", action="store_true", help="also display the plot window")
    return p.parse_args(argv)


def load_partials(directory: str) -> dict:
    groups: dict[tuple, list] = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(directory, "sc_L*.npz"))):
        d = np.load(path, allow_pickle=False)
        if "count" not in d.files or "s1" not in d.files:
            continue
        key = (int(d["L"]), int(d["master_seed"]), int(d["batch_size"]))
        groups[key].append((path, d))
    return groups


def combine_group(key: tuple, items: list) -> dict:
    L, master_seed, batch_size = key
    items = sorted(items, key=lambda pd: int(pd[1]["batch_start"]))
    ref = items[0][1]
    times = ref["times"]
    n_out = len(times)
    s1 = np.zeros(n_out); s2 = np.zeros(n_out)
    s3 = np.zeros(n_out); s4 = np.zeros(n_out)
    count = 0
    n_batches_total = 0
    covered_end = None
    gaps = []

    for path, d in items:
        if len(d["times"]) != n_out or not np.allclose(d["times"], times):
            raise SystemExit(
                f"ERROR: grid mismatch in {os.path.basename(path)} for L={L}; "
                "combine only slices from the same campaign (T/dt/n_store).")
        b0 = int(d["batch_start"]); nb = int(d["n_batches"]); b1 = b0 + nb
        if covered_end is not None:
            if b0 < covered_end:
                raise SystemExit(
                    f"ERROR: overlapping batch ranges for L={L} "
                    f"(…{covered_end}) vs ({b0}…) in {os.path.basename(path)}; "
                    "this would double-count trajectories.")
            if b0 > covered_end:
                gaps.append((covered_end, b0))
        covered_end = b1
        s1 += d["s1"]; s2 += d["s2"]; s3 += d["s3"]; s4 += d["s4"]
        count += int(d["count"]); n_batches_total += nb

    qfi, qfi_stderr = qfi_from_powersums(count, s1, s2, s3, s4)
    return {
        "times": times, "qfi": qfi, "qfi_stderr": qfi_stderr,
        "n_traj": count, "n_batches_total": n_batches_total,
        "L": L, "N": int(ref["N"]), "a": float(ref["a"]),
        "Omega": float(ref["Omega"]), "delta": float(ref["delta"]),
        "gamma0": float(ref["gamma0"]), "T": float(ref["T"]), "dt": float(ref["dt"]),
        "batch_size": batch_size, "master_seed": master_seed,
        "method": "semiclassical_array", "n_files": len(items), "gaps": gaps,
    }


def plot_results(results: list, outdir: str, show: bool = False) -> None:
    try:
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plot skipped] matplotlib unavailable: {exc}")
        return

    results = sorted(results, key=lambda r: r["L"])
    colors = plt.cm.viridis(np.linspace(0.12, 0.88, len(results)))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for r, c in zip(results, colors):
        t, q, e, N = r["times"], r["qfi"], r["qfi_stderr"], r["N"]
        ax.fill_between(t, (q - e) / N, (q + e) / N, color=c, alpha=0.20, linewidth=0)
        ax.plot(t, q / N, color=c, lw=1.6,
                label=f"L={r['L']} (N={N}, n={r['n_traj']})")
    ax.set_xlabel(r"time $\Gamma_0 t$")
    ax.set_ylabel(r"semiclassical QFI per atom  $F_\mathrm{sc}(t)/N$")
    ax.margins(x=0)
    ax.grid(True, alpha=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if len(results) > 1:
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out = os.path.join(outdir, "sc_array_qfi.png")
    fig.savefig(out, dpi=150)
    print(f"Wrote plot   {out}")
    if show:
        plt.show()
    plt.close(fig)


def main(argv=None) -> None:
    args = parse_args(argv)
    directory = os.path.abspath(args.dir)
    outdir = os.path.abspath(args.out) if args.out else directory
    os.makedirs(outdir, exist_ok=True)

    groups = load_partials(directory)
    if not groups:
        raise SystemExit(f"No sc_L*.npz partial files found in {directory}")

    by_L: dict[int, list] = defaultdict(list)
    for (L, seed, bs) in groups:
        by_L[L].append((seed, bs))
    for L, variants in by_L.items():
        if len(variants) > 1:
            raise SystemExit(
                f"ERROR: L={L} has multiple (master_seed, batch_size) campaigns "
                f"{variants}; separate them into different directories.")

    wanted = set(args.L) if args.L else None
    print(f"{'L':>4} {'N':>5} {'n_traj':>10} {'n_batch':>8} {'files':>6} "
          f"{'QFI(T)':>12} {'stderr(T)':>10} {'rel':>7}")
    results = []
    for key in sorted(groups, key=lambda k: k[0]):
        L = key[0]
        if wanted is not None and L not in wanted:
            continue
        res = combine_group(key, groups[key])
        outfile = os.path.join(outdir, f"sc_array_L{L}.npz")
        np.savez(outfile, **{k: v for k, v in res.items() if k != "gaps"})
        j = int(np.argmin(np.abs(res["times"] - res["T"])))
        q, e = res["qfi"][j], res["qfi_stderr"][j]
        rel = e / abs(q) if q else float("nan")
        print(f"{L:>4} {res['N']:>5} {res['n_traj']:>10} {res['n_batches_total']:>8} "
              f"{res['n_files']:>6} {q:>12.3f} {e:>10.3f} {rel:>6.2%}")
        if res["gaps"]:
            print(f"       ! gaps in batch coverage for L={L}: {res['gaps']}")
        results.append(res)

    print(f"\nWrote {len(results)} combined file(s) to {outdir}")
    if results and not args.no_plot:
        plot_results(results, outdir, show=args.show)


if __name__ == "__main__":
    main()
