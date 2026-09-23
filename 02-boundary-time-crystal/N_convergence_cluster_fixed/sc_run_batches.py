#!/usr/bin/env python3
"""Compute one batch slice of the semiclassical run and save its power sums.

Runs batches [batch_start, batch_start+n_batches) for a single N and writes the
additive power sums (not a reduced QFI) to one .npz; sc_aggregate.py combines
them. Each batch is seeded from its global index alone, so disjoint slices never
share trajectories and summing the files reproduces a single run over their
union (see cost_model.partition / btc.collective_qfi_powersums).

Output file: sc_N<N>_seed<master_seed>_bs<batch_size>_b<batch_start>+<n_batches>.npz
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
        description="Compute+save one batch slice of semiclassical power sums.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("N", type=int, help="system size (single N per task)")
    p.add_argument("--batch-start", type=int, default=0,
                   help="global index of the first batch in this slice")
    p.add_argument("--n-batches", type=int, default=None,
                   help="number of consecutive batches this task computes "
                        "(give this OR --n-traj)")
    p.add_argument("--n-traj", type=int, default=None,
                   help="convenience for local single-shot runs: compute "
                        "--n-batches = ceil(n_traj / batch_size), batch_start=0")
    p.add_argument("--batch-size", type=int, default=20,
                   help="trajectories per batch (keep fixed across the campaign)")
    p.add_argument("--master-seed", type=int, default=67,
                   help="campaign seed (per-batch independence via global index)")
    p.add_argument("--kappa", type=float, default=1.0, help="decay rate κ")
    p.add_argument("--ratio", type=float, default=2.0, help="sets Ω = ratio·κ·N/2")
    p.add_argument("--T", type=float, default=100.0, help="final time")
    p.add_argument("--dt-factor", type=float, default=0.01,
                   help="time step dt = dt_factor / max(Ω, κ)")
    p.add_argument("--n-store", type=int, default=200,
                   help="time-resolved output points kept")
    p.add_argument("--n-jobs", type=int, default=-1,
                   help="worker processes (-1 = all cores; on SLURM set to $SLURM_CPUS_PER_TASK)")
    p.add_argument("--outdir", type=str, default=os.getcwd(),
                   help="output directory for the partial .npz (default: current dir)")
    p.add_argument("--overwrite", action="store_true",
                   help="recompute even if the partial file already exists")
    p.add_argument("--no-progress", action="store_true",
                   help="disable the tqdm progress bar")
    return p.parse_args(argv)


def outfile_name(N, master_seed, batch_size, batch_start, n_batches) -> str:
    return (f"sc_N{N}_seed{master_seed}_bs{batch_size}"
            f"_b{batch_start:07d}+{n_batches}.npz")


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.N < 1:
        raise ValueError(f"N must be positive, got {args.N}")
    # Resolve n_batches: either given directly, or derived from --n-traj.
    if (args.n_batches is None) == (args.n_traj is None):
        raise SystemExit("give exactly one of --n-batches or --n-traj")
    if args.n_traj is not None:
        args.n_batches = (args.n_traj + args.batch_size - 1) // args.batch_size
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    fname = outfile_name(args.N, args.master_seed, args.batch_size,
                         args.batch_start, args.n_batches)
    outfile = os.path.join(outdir, fname)
    if os.path.exists(outfile) and not args.overwrite:
        print(f"[skip] {fname} already exists (use --overwrite to redo)", flush=True)
        return

    S = args.N / 2.0
    Omega = args.ratio * args.kappa * S
    dt = args.dt_factor / max(Omega, args.kappa)
    times = np.arange(0.0, args.T + dt, dt)
    ntraj_slice = args.n_batches * args.batch_size

    print(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] "
          f"N={args.N} (S={S:g}): Ω={Omega:g}, batches "
          f"[{args.batch_start}, {args.batch_start + args.n_batches}) "
          f"= {ntraj_slice} traj, n_time={len(times)}, n_jobs={args.n_jobs}",
          flush=True)

    ps = btc.collective_qfi_powersums(
        args.N, Omega, 0.0, args.kappa, times,
        batch_start=args.batch_start, n_batches=args.n_batches,
        batch_size=args.batch_size, master_seed=args.master_seed,
        sampling="wigner_cone",
        n_store=args.n_store, n_jobs=args.n_jobs, progress=not args.no_progress,
    )

    np.savez(
        outfile,
        # additive power sums (the payload)
        count=ps["count"], s1=ps["s1"], s2=ps["s2"], s3=ps["s3"], s4=ps["s4"],
        times=ps["times"],
        # slice identity (aggregator groups/combines on these)
        batch_start=args.batch_start, n_batches=args.n_batches,
        batch_size=args.batch_size, master_seed=args.master_seed,
        # run grid (must match across a campaign to combine)
        N=args.N, S=S, Omega=Omega, delta=0.0, kappa=args.kappa, ratio=args.ratio,
        T=args.T, dt=dt, dt_factor=args.dt_factor, n_store=args.n_store,
        scheme="diffusive", sampling="wigner_cone", method="semiclassical_diffusive",
    )
    print(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] saved {fname} "
          f"({ntraj_slice} trajectories)", flush=True)


if __name__ == "__main__":
    main()
