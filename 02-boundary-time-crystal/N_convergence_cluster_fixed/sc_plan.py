#!/usr/bin/env python3
"""Plan a fixed-N_TRAJ campaign: print per-N sizing/cost and the sbatch commands.

For a target trajectory count per N, prints batch size, array size, actual
trajectory count and compute cost, then the exact sbatch line to launch each N.
Does not submit anything; only computes (via cost_model) and prints.

Usage:
    python sc_plan.py --n-traj 20000 --N 128 256 512 1024
    python sc_plan.py --n-traj 20000 --N 512 --target-hours 6 --ncores 16
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cost_model  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plan a fixed-N_TRAJ semiclassical campaign and print sbatch commands.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n-traj", type=int, required=True,
                   help="total trajectories PER N (same for every N)")
    p.add_argument("--N", type=int, nargs="+", default=[128, 256, 512, 1024],
                   help="system sizes to plan")
    p.add_argument("--target-hours", type=float, default=4.0,
                   help="target walltime per array task (sets the array size)")
    p.add_argument("--ncores", type=int, default=16,
                   help="cores per task (SLURM --cpus-per-task)")
    p.add_argument("--batch-wall-ref", type=float, default=1800.0,
                   help="target single-core sec per batch (sets batch_size)")
    p.add_argument("--master-seed", type=int, default=42, help="campaign seed")
    p.add_argument("--script", default="sc_run_batches.sh",
                   help="batch script name to put in the printed sbatch commands")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    print(f"Target: {args.n_traj} trajectories per N, "
          f"~{args.target_hours:g} h/task on {args.ncores} cores "
          f"(batch_wall_ref={args.batch_wall_ref:g}s)\n")
    hdr = (f"{'N':>6} {'batch':>6} {'total':>7} {'actual':>9} {'array':>6} "
           f"{'batches':>8} {'est/task':>9} {'core-h':>9}")
    print(hdr)
    print(f"{'':>6} {'size':>6} {'batch':>7} {'traj':>9} {'size K':>6} "
          f"{'/task':>8} {'[h]':>9} {'total':>9}")
    print("-" * len(hdr))

    plans = []
    total_core_h = 0.0
    for N in args.N:
        pl = cost_model.plan(args.n_traj, N, args.target_hours, args.ncores,
                             args.batch_wall_ref)
        plans.append(pl)
        total_core_h += pl["core_hours"]
        print(f"{pl['N']:>6} {pl['batch_size']:>6} {pl['total_batches']:>7} "
              f"{pl['actual_traj']:>9} {pl['array_size']:>6} "
              f"{pl['batches_per_task']:>8} {pl['est_task_hours']:>9.1f} "
              f"{pl['core_hours']:>9.0f}")
    print("-" * len(hdr))
    print(f"{'total core-hours (all N):':>44} {total_core_h:>9.0f}\n")

    print("Submit commands (one campaign per N; keep --array, N_TRAJ, BATCH_SIZE, "
          "MASTER_SEED and TARGET_HOURS fixed for a given N):\n")
    for pl, N in zip(plans, args.N):
        k = pl["array_size"]
        print(
            f"sbatch --job-name=btc_sc_N{N} --array=0-{k-1} "
            f"--cpus-per-task={args.ncores} \\\n"
            f"       --export=ALL,N={N},N_TRAJ={args.n_traj},"
            f"BATCH_SIZE={pl['batch_size']},MASTER_SEED={args.master_seed} "
            f"{args.script}"
        )
    print("\nThen reduce + inspect convergence:  python sc_aggregate.py --dir results")


if __name__ == "__main__":
    main()
