#!/usr/bin/env python3
"""Shared batch-sizing / cost model. Used by sc_plan.py and sc_run_batches.sh
so both derive the same batch_size and task partition from N_TRAJ.
"""

from __future__ import annotations

import math

# Measured single-core seconds per trajectory (at the fixed run grid).
# UPDATE if your nodes differ; affects only sizing and time estimates.
PER_TRAJ: dict[int, float] = {
    4: 1.3, 8: 2.5, 16: 4.8, 32: 11.0, 64: 27.0,
    128: 71.0, 256: 159.0, 512: 600.0, 1024: 2460.0,
}


def per_traj(N: int) -> float:
    if N not in PER_TRAJ:
        raise KeyError(f"no per_traj cost for N={N}; add it to PER_TRAJ.")
    return float(PER_TRAJ[N])


def batch_size(N: int, batch_wall_ref: float = 1800.0, cap: int = 200) -> int:
    """Trajectories per batch ≈ batch_wall_ref sec of work, in [1, cap].

    A batch is the unit of parallel work (one per core at a time), so it must
    stay small at expensive N. It is also the RNG unit: keep batch_wall_ref
    fixed within a campaign, else slices use a different batch_size and cannot
    be combined.
    """
    return int(min(cap, max(1, round(batch_wall_ref / per_traj(N)))))


def total_batches(n_traj: int, bsize: int) -> int:
    """Batches needed for n_traj trajectories (rounded up)."""
    return (int(n_traj) + bsize - 1) // bsize


def partition(total: int, k: int, t: int) -> tuple[int, int]:
    """Task t's (batch_start, n_batches) in an even split of `total` over `k`.

    Deterministic in (total, k): the k slices tile [0, total) with no
    gaps/overlaps, so re-running the same (N_TRAJ, array size) reproduces them
    (resumable) and the union is exactly [0, total).
    """
    base, rem = divmod(total, k)
    n = base + (1 if t < rem else 0)
    start = t * base + min(t, rem)
    return start, n


def task_slice(n_traj: int, N: int, k: int, t: int, ncores: int = 16,
               batch_wall_ref: float = 1800.0, batch_size_override=None) -> dict:
    """One array task's slice (k = array size, t = 0-based task index).

    batch_size_override pins the batch size (passed by sc_plan.py); otherwise
    it is derived here.
    """
    bsize = int(batch_size_override) if batch_size_override else \
        batch_size(N, batch_wall_ref)
    tb = total_batches(n_traj, bsize)
    start, n = partition(tb, k, t)
    est = math.ceil(n / ncores) * bsize * per_traj(N) / 3600.0 if n > 0 else 0.0
    return {"batch_size": bsize, "total_batches": tb, "batch_start": start,
            "n_batches": n, "est_task_hours": est, "actual_traj": tb * bsize}


def plan(n_traj: int, N: int, target_hours: float = 4.0, ncores: int = 16,
         batch_wall_ref: float = 1800.0) -> dict:
    """Sizing for one N at fixed n_traj. Picks the array size k so each task
    runs ~target_hours on ncores; the trajectory count is independent of k.
    """
    bsize = batch_size(N, batch_wall_ref)
    tb = total_batches(n_traj, bsize)
    pt = per_traj(N)
    waves = max(1, int(target_hours * 3600.0 / (bsize * pt)))
    bpt = waves * ncores
    k = max(1, math.ceil(tb / bpt))
    nb_task = math.ceil(tb / k)
    est_task_h = math.ceil(nb_task / ncores) * bsize * pt / 3600.0
    return {
        "N": N, "n_traj_target": int(n_traj), "batch_size": bsize,
        "total_batches": tb, "actual_traj": tb * bsize, "array_size": k,
        "batches_per_task": nb_task, "est_task_hours": est_task_h,
        "core_hours": tb * bsize * pt / 3600.0, "per_traj": pt,
    }
