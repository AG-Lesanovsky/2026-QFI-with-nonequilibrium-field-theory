#!/usr/bin/env python3
"""Plan a fixed-N_TRAJ semiclassical campaign: per-L sizing/cost + sbatch commands.

For a target trajectory count per lattice, prints batch size, array size, actual
trajectory count and compute cost, then the exact sbatch line to launch each L.
Does not submit anything; only computes (via cost_model) and prints.

Usage:
    python sc_plan.py --n-traj 20000 --L 2 3 4 6 8 10
    python sc_plan.py --n-traj 20000 --L 10 --target-hours 6 --ncores 16
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cost_model  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plan a fixed-N_TRAJ semiclassical array campaign; print sbatch commands.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n-traj", type=int, required=True,
                   help="total trajectories PER lattice (same for every L)")
    p.add_argument("--L", type=int, nargs="+", default=[2, 3, 4, 6, 8, 10],
                   help="lattice sides to plan (N = L²)")
    p.add_argument("--target-hours", type=float, default=4.0,
                   help="target walltime per array task (sets the array size)")
    p.add_argument("--ncores", type=int, default=16, help="cores per task")
    p.add_argument("--batch-wall-ref", type=float, default=1800.0,
                   help="target single-core sec per batch (sets batch_size)")
    p.add_argument("--master-seed", type=int, default=42, help="campaign seed")
    p.add_argument("--script", default="sc_run_batches.sh",
                   help="batch script name for the printed sbatch commands")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    print(f"Target: {args.n_traj} trajectories per L, "
          f"~{args.target_hours:g} h/task on {args.ncores} cores "
          f"(batch_wall_ref={args.batch_wall_ref:g}s)\n")
    hdr = (f"{'L':>4} {'N':>5} {'batch':>6} {'total':>7} {'actual':>9} {'array':>6} "
           f"{'batches':>8} {'est/task':>9} {'core-h':>9}")
    print(hdr)
    print(f"{'':>4} {'':>5} {'size':>6} {'batch':>7} {'traj':>9} {'size K':>6} "
          f"{'/task':>8} {'[h]':>9} {'total':>9}")
    print("-" * len(hdr))

    plans, total_core_h = [], 0.0
    for L in args.L:
        N = L * L
        pl = cost_model.plan(args.n_traj, N, args.target_hours, args.ncores,
                             args.batch_wall_ref)
        plans.append((L, pl))
        total_core_h += pl["core_hours"]
        print(f"{L:>4} {N:>5} {pl['batch_size']:>6} {pl['total_batches']:>7} "
              f"{pl['actual_traj']:>9} {pl['array_size']:>6} "
              f"{pl['batches_per_task']:>8} {pl['est_task_hours']:>9.1f} "
              f"{pl['core_hours']:>9.0f}")
    print("-" * len(hdr))
    print(f"{'total core-hours (all L):':>52} {total_core_h:>9.0f}\n")

    print("Submit commands (one campaign per L; keep --array, N_TRAJ, BATCH_SIZE, "
          "MASTER_SEED fixed for a given L):\n")
    for L, pl in plans:
        k = pl["array_size"]
        print(f"sbatch --job-name=arr_sc_L{L} --array=0-{k-1} "
              f"--cpus-per-task={args.ncores} \\\n"
              f"       --export=ALL,L={L},N_TRAJ={args.n_traj},"
              f"BATCH_SIZE={pl['batch_size']},MASTER_SEED={args.master_seed} "
              f"{args.script}")
    print("\nThen reduce + inspect convergence:  python sc_aggregate.py --dir results")


if __name__ == "__main__":
    main()
