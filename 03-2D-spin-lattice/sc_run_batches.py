#!/usr/bin/env python3
"""Compute one batch slice of the semiclassical array QFI and save its power sums.

Runs batches [batch_start, batch_start+n_batches) for a single L×L lattice and
writes the additive power sums (not a reduced QFI) to one .npz; sc_aggregate.py
combines them. Each batch is seeded from its global index alone, so disjoint
slices never share trajectories and summing the files reproduces a single run
over their union (see cost_model.partition / atomarray.array_qfi_powersums).

Output: sc_L<L>_seed<master_seed>_bs<batch_size>_b<batch_start>+<n_batches>.npz
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
        description="Compute+save one batch slice of semiclassical array power sums.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("L", type=int, help="lattice side (N = L² atoms)")
    p.add_argument("--batch-start", type=int, default=0,
                   help="global index of the first batch in this slice")
    p.add_argument("--n-batches", type=int, default=None,
                   help="number of consecutive batches (give this OR --n-traj)")
    p.add_argument("--n-traj", type=int, default=None,
                   help="local single-shot: n_batches = ceil(n_traj / batch_size)")
    p.add_argument("--batch-size", type=int, default=20,
                   help="trajectories per batch (keep fixed across the campaign)")
    p.add_argument("--master-seed", type=int, default=42, help="campaign seed")
    p.add_argument("--a", type=float, default=0.8, help="lattice spacing (λe units)")
    p.add_argument("--omega", type=float, default=2.0, help="Rabi frequency Ω")
    p.add_argument("--gamma0", type=float, default=1.0, help="single-atom decay Γ0")
    p.add_argument("--delta", type=float, default=0.0, help="detuning Δ")
    p.add_argument("--T", type=float, default=10.0, help="final time")
    p.add_argument("--dt-factor", type=float, default=0.02,
                   help="time step dt = dt_factor / max(2Ω, Γ0)")
    p.add_argument("--n-store", type=int, default=200,
                   help="time-resolved output points kept")
    p.add_argument("--n-jobs", type=int, default=-1,
                   help="worker processes (-1 = all; on SLURM set $SLURM_CPUS_PER_TASK)")
    p.add_argument("--outdir", type=str, default=os.getcwd(),
                   help="output directory for the partial .npz")
    p.add_argument("--overwrite", action="store_true",
                   help="recompute even if the partial file already exists")
    p.add_argument("--no-progress", action="store_true", help="disable the progress bar")
    return p.parse_args(argv)


def outfile_name(L, master_seed, batch_size, batch_start, n_batches) -> str:
    return (f"sc_L{L}_seed{master_seed}_bs{batch_size}"
            f"_b{batch_start:07d}+{n_batches}.npz")


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.L < 1:
        raise ValueError(f"L must be positive, got {args.L}")
    if (args.n_batches is None) == (args.n_traj is None):
        raise SystemExit("give exactly one of --n-batches or --n-traj")
    if args.n_traj is not None:
        args.n_batches = (args.n_traj + args.batch_size - 1) // args.batch_size

    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    fname = outfile_name(args.L, args.master_seed, args.batch_size,
                         args.batch_start, args.n_batches)
    outfile = os.path.join(outdir, fname)
    if os.path.exists(outfile) and not args.overwrite:
        print(f"[skip] {fname} already exists (use --overwrite to redo)", flush=True)
        return

    N = args.L * args.L
    pos = aa.square_lattice(args.L, args.a)
    dt = args.dt_factor / max(2.0 * args.omega, args.gamma0)
    times = np.arange(0.0, args.T + dt, dt)
    ntraj_slice = args.n_batches * args.batch_size

    print(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] "
          f"L={args.L} (N={N}): Ω={args.omega:g}, a={args.a:g}, batches "
          f"[{args.batch_start}, {args.batch_start + args.n_batches}) "
          f"= {ntraj_slice} traj, n_time={len(times)}, n_jobs={args.n_jobs}", flush=True)

    ps = aa.array_qfi_powersums(
        pos, aa.EP_CIRCULAR_XY, args.omega, times,
        batch_start=args.batch_start, n_batches=args.n_batches,
        batch_size=args.batch_size, master_seed=args.master_seed,
        Delta=args.delta, Gamma0=args.gamma0,
        n_store=args.n_store, n_jobs=args.n_jobs, progress=not args.no_progress,
    )

    np.savez(
        outfile,
        count=ps["count"], s1=ps["s1"], s2=ps["s2"], s3=ps["s3"], s4=ps["s4"],
        times=ps["times"],
        batch_start=args.batch_start, n_batches=args.n_batches,
        batch_size=args.batch_size, master_seed=args.master_seed,
        L=args.L, N=N, a=args.a, Omega=args.omega, delta=args.delta,
        gamma0=args.gamma0, T=args.T, dt=dt, dt_factor=args.dt_factor,
        n_store=args.n_store, method="semiclassical_array",
    )
    print(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] saved {fname} "
          f"({ntraj_slice} trajectories)", flush=True)


if __name__ == "__main__":
    main()
