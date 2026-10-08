#!/usr/bin/env python3
"""Combine sc_N*.npz power-sum slices into one file per N.

Groups the partial files (from sc_run_batches.py) by (N, master_seed,
batch_size), sums the additive power sums over their batch ranges, and writes
sc_diffusive_N<N>.npz with keys: times, qfi, qfi_stderr, n_traj,
n_batches_total, N, S, Omega, kappa, ratio, T, dt, batch_size, master_seed,
scheme, sampling, method (plus state/integrator/X_definition and the campaign
lists master_seeds/batch_sizes when the slices carry them).

Idempotent: re-run as new slices land. Errors on repeated trajectory streams
(double-count) and on grid/parameter mismatches; warns on gaps. Prints n_traj
and relative stderr at t=T per N.

Two extra input modes:

--combine-campaigns
    Pool several campaigns of the same N into one file — e.g. a cluster array
    that gave every task its own master seed instead of its own batch range.
    The power sums are additive, so pooling independently seeded campaigns is
    exactly one longer run.  A trajectory stream is fixed by (master_seed,
    global batch index) alone — batch_size does *not* enter the seed — so the
    double-count check runs per master seed across all campaigns, and two
    slices sharing a seed must still hold disjoint batch ranges even if their
    batch sizes differ.  Every pooled slice must agree on the time grid and on
    (S, Omega, delta, kappa, ratio, T, dt) and on scheme/sampling/method/state.
    Without the flag, an N with more than one campaign is an error, as before.

--from-csv
    Read already-reduced `t,qfi,stderr,ntraj` tables instead of power sums, for
    data that arrives without its partials.  N is taken from the file name
    (`…N<N>…`), the Ω dt factor from a `dt<factor>` token if present, and
    Ω = ratio·κ·N/2 from --ratio/--kappa.  A reduced table carries no power
    sums, so it can only be converted one-to-one — never summed with anything.

Usage:
    python sc_aggregate.py [--dir DIR] [--out DIR] [--N N ...] [--combine-campaigns]
    python sc_aggregate.py --from-csv --dir DIR [--csv-pattern GLOB]
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from btc import qfi_from_powersums  # noqa: E402

# Physics/grid scalars every pooled slice must agree on (mixing them would
# average unlike runs into one curve).  dt is the sharp one: the same N at two
# dt factors is two different simulations, not more statistics.
MATCH_FLOAT = ("S", "Omega", "delta", "kappa", "ratio", "T", "dt")
MATCH_STR = ("scheme", "sampling", "method", "state")
# Provenance carried through from the reference slice when present.
CARRY_EXTRA = ("delta", "dt_factor", "n_store", "state", "integrator", "X_definition")
# Legacy files predate some of these; fall back to what the old runner wrote.
CARRY_DEFAULT = {
    "scheme": "diffusive",
    "sampling": "wigner_cone",
    "method": "semiclassical_diffusive",
}

# --from-csv: accepted column spellings, and the file-name tokens to mine.
CSV_ALIASES = {
    "t": ("t", "time", "times", "kt"),
    "qfi": ("qfi", "f", "fq", "qfi_mean", "mean"),
    "stderr": ("stderr", "qfi_stderr", "err", "error", "sem", "std_err"),
    "ntraj": ("ntraj", "n_traj", "count", "n"),
}
CSV_N_RE = re.compile(r"N(\d+)")
CSV_DT_RE = re.compile(r"dt([0-9][0-9.]*(?:[eE][+-]?[0-9]+)?)")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Combine semiclassical power-sum checkpoints into per-N QFI files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dir", default=os.getcwd(),
                   help="directory holding the sc_N*.npz partial files "
                        "(or the CSV tables, with --from-csv)")
    p.add_argument("--out", default=None,
                   help="output directory for combined files (default: same as --dir)")
    p.add_argument("--N", type=int, nargs="*", default=None,
                   help="restrict to these N (default: every N found)")
    p.add_argument("--combine-campaigns", action="store_true",
                   help="pool every campaign of the same N (differing master_seed "
                        "and/or batch_size) into one file instead of erroring")
    p.add_argument("--from-csv", action="store_true",
                   help="read reduced t,qfi,stderr,ntraj CSV tables instead of "
                        "power-sum partials (converted one-to-one, not summed)")
    p.add_argument("--csv-pattern", default="*.csv",
                   help="glob for the CSV tables inside --dir (--from-csv only)")
    p.add_argument("--kappa", type=float, default=1.0,
                   help="κ recorded for CSV input (--from-csv only)")
    p.add_argument("--ratio", type=float, default=2.0,
                   help="Ω/ω_c recorded for CSV input, sets Ω = ratio·κ·N/2 "
                        "(--from-csv only)")
    p.add_argument("--no-plot", action="store_true",
                   help="skip the QFI(t) diagnostic plot")
    p.add_argument("--show", action="store_true",
                   help="also display the plot window (default: save file only)")
    return p.parse_args(argv)


def _scalar(d, key):
    """An npz 0-d entry as a plain str/int/float, or None when absent."""
    if key not in d.files:
        return None
    v = d[key]
    if v.dtype.kind in "US":
        return str(v)
    if v.dtype.kind in "iub":
        return int(v)
    return float(v)


def load_partials(directory: str) -> dict:
    """Group partial files by (N, master_seed, batch_size)."""
    groups: dict[tuple, list] = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(directory, "sc_N*.npz"))):
        d = np.load(path, allow_pickle=False)
        if "count" not in d.files or "s1" not in d.files:
            continue  # not a power-sum partial
        key = (int(d["N"]), int(d["master_seed"]), int(d["batch_size"]))
        groups[key].append((path, d))
    return groups


def check_consistent(N: int, items: list) -> None:
    """Every slice of one N must describe the same simulation."""
    _, ref = items[0]
    times = ref["times"]
    for path, d in items[1:]:
        if len(d["times"]) != len(times) or not np.allclose(d["times"], times):
            raise SystemExit(
                f"ERROR: grid mismatch in {os.path.basename(path)} for N={N}; "
                "combine only slices from the same campaign (T/dt/n_store)."
            )
        for k in MATCH_FLOAT:
            a, b = _scalar(ref, k), _scalar(d, k)
            if a is None or b is None or np.isclose(a, b, rtol=1e-12, atol=0.0):
                continue
            raise SystemExit(
                f"ERROR: {k} mismatch for N={N}: {a!r} in "
                f"{os.path.basename(items[0][0])} vs {b!r} in "
                f"{os.path.basename(path)}; these are different runs, not more "
                "statistics — separate them into different directories."
            )
        for k in MATCH_STR:
            a, b = _scalar(ref, k), _scalar(d, k)
            if a is None or b is None or a == b:
                continue
            raise SystemExit(
                f"ERROR: {k} mismatch for N={N}: {a!r} in "
                f"{os.path.basename(items[0][0])} vs {b!r} in "
                f"{os.path.basename(path)}; these are different estimators — "
                "separate them into different directories."
            )


def check_streams(N: int, items: list) -> list:
    """Reject repeated trajectory streams; report coverage gaps.

    A batch's trajectories come from ``SeedSequence(master_seed, spawn_key=(g,))``
    (see btc.collective_qfi_powersums), so a stream is identified by
    (master_seed, g) *alone*: two slices that share a seed and a batch index
    replay the same trajectories even at different batch sizes — one is then a
    prefix of the other.  Hence the interval check runs per master seed and
    ignores batch_size, which is what makes pooling campaigns safe.

    Returns the gaps as (master_seed, first_missing, first_present) triples.
    """
    by_seed: dict[int, list] = defaultdict(list)
    for path, d in items:
        b0 = int(d["batch_start"])
        by_seed[int(d["master_seed"])].append((b0, b0 + int(d["n_batches"]), path))

    gaps = []
    for seed, intervals in sorted(by_seed.items()):
        intervals.sort()
        covered_end = None       # end of the last consumed interval
        for b0, b1, path in intervals:
            if covered_end is not None:
                if b0 < covered_end:
                    raise SystemExit(
                        f"ERROR: overlapping batch ranges for N={N}, "
                        f"master_seed={seed} (…{covered_end}) vs ({b0}…) in "
                        f"{os.path.basename(path)}; this would double-count "
                        "trajectories."
                    )
                if b0 > covered_end:
                    gaps.append((seed, covered_end, b0))
            covered_end = b1
    return gaps


def combine(N: int, items: list) -> dict:
    """Sum the power sums of one N over its (disjoint) trajectory streams."""
    # canonical order: by (master_seed, batch_start), so the reduction is
    # independent of file-system order even when campaigns are pooled
    items = sorted(items, key=lambda pd: (int(pd[1]["master_seed"]),
                                          int(pd[1]["batch_start"])))
    check_consistent(N, items)
    gaps = check_streams(N, items)

    ref = items[0][1]
    times = ref["times"]
    n_out = len(times)
    s1 = np.zeros(n_out); s2 = np.zeros(n_out)
    s3 = np.zeros(n_out); s4 = np.zeros(n_out)
    count = 0
    n_batches_total = 0

    for _, d in items:
        s1 += d["s1"]; s2 += d["s2"]; s3 += d["s3"]; s4 += d["s4"]
        count += int(d["count"]); n_batches_total += int(d["n_batches"])

    campaigns = sorted({(int(d["master_seed"]), int(d["batch_size"]))
                        for _, d in items})
    seeds = sorted({c[0] for c in campaigns})
    sizes = sorted({c[1] for c in campaigns})

    qfi, qfi_stderr = qfi_from_powersums(count, s1, s2, s3, s4)
    res = {
        "times": times, "qfi": qfi, "qfi_stderr": qfi_stderr,
        "n_traj": count, "n_batches_total": n_batches_total,
        "N": N, "S": float(ref["S"]), "Omega": float(ref["Omega"]),
        "kappa": float(ref["kappa"]), "ratio": float(ref["ratio"]),
        "T": float(ref["T"]), "dt": float(ref["dt"]),
        # scalars stay scalar for a single campaign (back-compatible); -1 marks
        # a pooled file, whose full make-up is in master_seeds / batch_sizes
        "batch_size": sizes[0] if len(sizes) == 1 else -1,
        "master_seed": seeds[0] if len(seeds) == 1 else -1,
        "master_seeds": np.array(seeds, dtype=np.int64),
        "batch_sizes": np.array(sizes, dtype=np.int64),
        "n_campaigns": len(campaigns),
        "source": "powersums",
        "n_files": len(items), "gaps": gaps,
    }
    for k, default in CARRY_DEFAULT.items():
        v = _scalar(ref, k)
        res[k] = default if v is None else v
    for k in CARRY_EXTRA:
        v = _scalar(ref, k)
        if v is not None:
            res[k] = v
    return res


def collect_powersums(directory: str, combine_campaigns: bool,
                      wanted: set | None) -> list:
    """Every N found under `directory`, reduced to one result dict each."""
    groups = load_partials(directory)
    if not groups:
        raise SystemExit(f"No sc_N*.npz partial files found in {directory}")

    by_N: dict[int, list] = defaultdict(list)
    for key, items in groups.items():
        by_N[key[0]].extend(items)

    if not combine_campaigns:
        # guard: one combinable campaign per N (else the file name would collide)
        for N in sorted(by_N):
            variants = sorted(k[1:] for k in groups if k[0] == N)
            if len(variants) > 1:
                raise SystemExit(
                    f"ERROR: N={N} has multiple (master_seed, batch_size) campaigns "
                    f"{variants}; pass --combine-campaigns to pool them into one "
                    "file, or separate them into different directories.")

    results = []
    for N in sorted(by_N):
        if wanted is not None and N not in wanted:
            continue
        results.append(combine(N, by_N[N]))
    return results


def read_csv_table(path: str) -> dict:
    """One reduced `t,qfi,stderr[,ntraj]` table as arrays, by column alias."""
    tab = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    if tab.dtype.names is None:
        raise SystemExit(f"ERROR: {os.path.basename(path)} has no CSV header row; "
                         "expected columns t,qfi,stderr,ntraj.")
    found = {n.lower(): n for n in tab.dtype.names}
    cols = {}
    for want, aliases in CSV_ALIASES.items():
        hit = next((found[a] for a in aliases if a in found), None)
        if hit is None:
            if want == "ntraj":
                continue        # optional: only used for the summary table
            raise SystemExit(
                f"ERROR: {os.path.basename(path)} has no '{want}' column "
                f"(looked for {aliases}, found {tuple(tab.dtype.names)})."
            )
        cols[want] = np.atleast_1d(tab[hit]).astype(float)
    return cols


def collect_csv(directory: str, pattern: str, kappa: float, ratio: float,
                wanted: set | None) -> list:
    """Convert reduced CSV tables one-to-one into result dicts."""
    paths = sorted(glob.glob(os.path.join(directory, pattern)))
    if not paths:
        raise SystemExit(f"No CSV files matching {pattern!r} in {directory}")

    seen: dict[int, str] = {}
    results = []
    for path in paths:
        base = os.path.basename(path)
        m = CSV_N_RE.search(base)
        if m is None:
            print(f"[skip] {base}: no N<number> token in the file name")
            continue
        N = int(m.group(1))
        if wanted is not None and N not in wanted:
            continue
        if N in seen:
            raise SystemExit(
                f"ERROR: N={N} appears in both {seen[N]} and {base}; they would "
                f"write the same sc_diffusive_N{N}.npz. Narrow the selection "
                "with --csv-pattern (or --N)."
            )
        seen[N] = base

        cols = read_csv_table(path)
        t = cols["t"]
        S = N / 2.0
        Omega = ratio * kappa * S
        # the file-name `dt` token is the dimensionless Ω dt factor, as in the
        # runner: dt = dt_factor / max(Ω, κ)
        mdt = CSV_DT_RE.search(base)
        dt_factor = float(mdt.group(1)) if mdt else float("nan")
        dt = dt_factor / max(Omega, kappa)
        n_traj = int(np.max(cols["ntraj"])) if "ntraj" in cols else 0

        results.append({
            "times": t, "qfi": cols["qfi"], "qfi_stderr": cols["stderr"],
            "n_traj": n_traj, "n_batches_total": 0,
            "N": N, "S": S, "Omega": Omega, "kappa": kappa, "ratio": ratio,
            "T": float(t[-1]), "dt": dt, "dt_factor": dt_factor,
            "batch_size": -1, "master_seed": -1,
            "master_seeds": np.array([], dtype=np.int64),
            "batch_sizes": np.array([], dtype=np.int64),
            "n_campaigns": 0,
            "scheme": "diffusive", "sampling": "wigner_cone",
            "method": "semiclassical_diffusive",
            "source": f"csv:{base}",
            "n_files": 1, "gaps": [],
        })
    return sorted(results, key=lambda r: r["N"])


def plot_results(results: list, outdir: str, show: bool = False) -> None:
    """Quick QFI(t)±stderr check, one line + band per N. Saves a PNG."""
    try:
        import matplotlib
        if not show:
            matplotlib.use("Agg")   # headless: save without a display
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plot skipped] matplotlib unavailable: {exc}")
        return

    results = sorted(results, key=lambda r: r["N"])
    # N is an ordered series -> perceptually-uniform, CVD-safe viridis ramp
    # (small N = dark purple -> large N = yellow), matching the analysis notebook.
    colors = plt.cm.viridis(np.linspace(0.12, 0.88, len(results)))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for r, c in zip(results, colors):
        t, q, e = r["times"], r["qfi"], r["qfi_stderr"]
        ax.fill_between(t, q/ t - e/ t, q/ t + e/ t, color=c, alpha=0.20, linewidth=0)
        ax.plot(t, q / t, color=c, lw=1.6, label=f"N={r['N']}  (n={r['n_traj']})")
    ax.set_xlabel(r"time $t$")
    ax.set_ylabel(r"semiclassical QFI  $F_\mathrm{sc}(t) / t$")
    ax.margins(x=0)
    ax.grid(True, alpha=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if len(results) > 1:
        ax.legend(frameon=False, fontsize=8)
    else:
        ax.set_title(f"N={results[0]['N']}  (n={results[0]['n_traj']})", fontsize=10)
    fig.tight_layout()
    out = os.path.join(outdir, "sc_diffusive_qfi.png")
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
    wanted = set(args.N) if args.N else None

    if args.from_csv:
        results = collect_csv(directory, args.csv_pattern, args.kappa,
                              args.ratio, wanted)
    else:
        results = collect_powersums(directory, args.combine_campaigns, wanted)

    print(f"{'N':>6} {'n_traj':>10} {'n_batch':>8} {'files':>6} {'camp':>5} "
          f"{'QFI(T)':>12} {'stderr(T)':>10} {'rel':>7}")
    for res in results:
        N = res["N"]
        outfile = os.path.join(outdir, f"sc_diffusive_N{N}.npz")
        np.savez(outfile, **{k: v for k, v in res.items() if k != "gaps"})
        j = int(np.argmin(np.abs(res["times"] - res["T"])))
        q, e = res["qfi"][j], res["qfi_stderr"][j]
        rel = e / abs(q) if q else float("nan")
        nb = res["n_batches_total"] or "-"
        camp = res["n_campaigns"] or "-"
        print(f"{N:>6} {res['n_traj']:>10} {nb:>8} "
              f"{res['n_files']:>6} {camp:>5} {q:>12.3f} {e:>10.3f} {rel:>6.2%}")
        if res["n_campaigns"] > 1:
            print(f"       pooled {res['n_campaigns']} campaigns: "
                  f"master_seeds={res['master_seeds'].tolist()}, "
                  f"batch_sizes={res['batch_sizes'].tolist()}")
        if res["gaps"]:
            print(f"       ! gaps in batch coverage for N={N} "
                  f"(seed, from, to): {res['gaps']}")

    print(f"\nWrote {len(results)} combined file(s) to {outdir}")
    if results and not args.no_plot:
        plot_results(results, outdir, show=args.show)


if __name__ == "__main__":
    main()
