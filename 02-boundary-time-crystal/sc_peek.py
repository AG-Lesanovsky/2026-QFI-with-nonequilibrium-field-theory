#!/usr/bin/env python3
"""Reduce whatever exists right now -- finished .npz, finished batches, and
in-flight checkpoints -- into a QFI(t) curve you can look at mid-campaign.

Nothing here touches the running jobs: checkpoints are written with os.replace,
so a reader always sees a complete file.

    python3 sc_peek.py /lustre/.../dtchk512/N_512
    python3 sc_peek.py <dir> --csv qfi_now.csv --plot qfi_now.png

Partial batches have only filled the store points they have reached, so the
sample size shrinks with t: every output row carries the trajectory count that
actually contributed at that time, and the error bar widens accordingly.  Late
times being noisier than early ones is expected, not a bug.
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import btc  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Aggregate finished + in-flight BTC batches into QFI(t).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("dir", help="results dir (contains *.npz and/or ckpt_t*/)")
    p.add_argument("--csv", default=None, help="write the curve here")
    p.add_argument("--plot", default=None, help="write a PNG here")
    p.add_argument("--rows", type=int, default=12,
                   help="table rows to print (0 = none)")
    return p.parse_args(argv)


def _store_times(hdr):
    """Rebuild the output time grid from a checkpoint header."""
    n_time, n_out, dt = hdr["n_time"], hdr["n_out"], hdr["dt"]
    idx = np.unique(np.linspace(0, n_time - 1, n_out).round().astype(np.int64))
    return idx * dt


def collect(d):
    """Return (times, s1..s4, count_per_timepoint, provenance counts).

    Each contributor adds its power sums only over the time points it actually
    reached, and bumps `count` over exactly that range -- so `count[j]` is the
    true number of trajectories behind the estimate at t[j].
    """
    times = None
    acc = None
    cnt = None
    seen = {"npz": 0, "done": 0, "ckpt": 0}
    ntraj = {"npz": 0, "done": 0, "ckpt": 0}

    def _ensure(t):
        nonlocal times, acc, cnt
        if times is None:
            times = t
            acc = [np.zeros(len(t)) for _ in range(4)]
            cnt = np.zeros(len(t), dtype=np.int64)
        elif len(t) != len(times):
            raise SystemExit(
                f"inconsistent output grids in {d}: {len(t)} vs {len(times)}. "
                "Mixing runs with different n_store/T here would be meaningless."
            )

    def _add(sums, n_traj, upto):
        for a, s in zip(acc, sums):
            a[:upto] += np.asarray(s)[:upto]
        cnt[:upto] += n_traj

    # finished slices
    for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
        z = np.load(f)
        _ensure(z["times"])
        _add((z["s1"], z["s2"], z["s3"], z["s4"]), int(z["count"]), len(times))
        seen["npz"] += 1
        ntraj["npz"] += int(z["count"])

    # finished batches and in-flight batches from any task's ckpt dir
    for f in sorted(glob.glob(os.path.join(d, "ckpt_t*", "*.pkl"))):
        try:
            with open(f, "rb") as fh:
                p = pickle.load(fh)
        except Exception:
            continue                      # being written / corrupt: skip
        hdr = p["hdr"]
        _ensure(_store_times(hdr))
        base = os.path.basename(f)
        if base.startswith("done_"):
            n_traj, s1, s2, s3, s4 = p["res"]
            _add((s1, s2, s3, s4), n_traj, len(times))
            seen["done"] += 1
            ntraj["done"] += n_traj
        else:
            k = p["k"]                    # store points filled so far
            _add((p["s1"], p["s2"], p["s3"], p["s4"]), hdr["n_traj"], k)
            seen["ckpt"] += 1
            ntraj["ckpt"] += hdr["n_traj"]

    if times is None:
        raise SystemExit(f"no .npz, done_*.pkl or ckpt_*.pkl found under {d}")
    return times, acc, cnt, seen, ntraj


def main(argv=None):
    a = parse_args(argv)
    times, (s1, s2, s3, s4), cnt, seen, ntraj = collect(a.dir)

    print(f"{a.dir}")
    print(f"  finished slices (.npz) : {seen['npz']:>4}  ({ntraj['npz']} traj)")
    print(f"  finished batches       : {seen['done']:>4}  ({ntraj['done']} traj)")
    print(f"  in-flight batches      : {seen['ckpt']:>4}  ({ntraj['ckpt']} traj)")
    print(f"  trajectories at t=0    : {cnt[0]}")
    print(f"  trajectories at t=T    : {cnt[-1]}")

    live = cnt > 1
    if not live.any():
        raise SystemExit("no time point has >1 trajectory yet -- too early to reduce")

    # Reduce each time point with the sample size that actually reached it.
    qfi = np.full(len(times), np.nan)
    err = np.full(len(times), np.nan)
    for j in np.flatnonzero(live):
        n = int(cnt[j])
        f, e = btc.qfi_from_powersums(
            n, s1[j:j + 1], s2[j:j + 1], s3[j:j + 1], s4[j:j + 1])
        qfi[j], err[j] = f[0], e[0]

    tmax = times[live][-1]
    print(f"  reduced out to t = {tmax:g} of {times[-1]:g} "
          f"({100 * tmax / times[-1]:.1f}% of the run)")

    if a.rows:
        idx = np.linspace(0, np.flatnonzero(live)[-1], a.rows).round().astype(int)
        print(f"\n{'t':>10} {'QFI':>14} {'stderr':>12} {'rel%':>7} {'ntraj':>8}")
        print("-" * 55)
        for j in idx:
            if not live[j]:
                continue
            rel = 100 * err[j] / abs(qfi[j]) if qfi[j] else np.nan
            print(f"{times[j]:>10.4g} {qfi[j]:>14.6g} {err[j]:>12.4g} "
                  f"{rel:>7.2f} {cnt[j]:>8d}")

    if a.csv:
        np.savetxt(a.csv, np.column_stack([times, qfi, err, cnt]),
                   delimiter=",", header="t,qfi,stderr,ntraj", comments="")
        print(f"\n-> {a.csv}")

    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
        m = live & np.isfinite(qfi)
        ax.fill_between(times[m], qfi[m] - err[m], qfi[m] + err[m],
                        alpha=0.25, lw=0, label="±1 s.e.")
        ax.plot(times[m], qfi[m], lw=1.6, label=f"QFI  (n≤{cnt.max()})")
        ax.set_xlabel("t")
        ax.set_ylabel("QFI")
        ax.set_title(os.path.basename(os.path.normpath(a.dir)))
        ax.legend(loc="upper right", frameon=False)
        fig.savefig(a.plot, dpi=150)
        print(f"-> {a.plot}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
