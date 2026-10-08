# Collective-spin (BTC) semiclassical QFI code — export 2026-08-29

Diffusive collective TWA for the boundary time crystal (Cabot–Carollo–Lesanovsky convention:
κ fixed, Ω = ratio·κ·S, S = N/2; defaults κ=1, ratio=2, T=100). QFI for sensing Ω:
F = 4·Var[X],  X(t) = ∫₀ᵗ (S_x/S) dt′ per trajectory; all outputs are F/S².

## Contents
btc/sde.py        stochastic integrators. USE scheme="diffusive": Stratonovich drift
                  (collective_diffusive_drift_cartesian) + global x/y rotation noise.
                  FIX 2026-08-28: the drift half-steps now rotate about the true axis
                  ω_n = B − A_n (collective_diffusive_axis / _axis_half_rotation). The
                  previous _half_rotation rotated about the tangential projection of ω,
                  which gave an O(Ω·dt_factor) spurious drift growing with N (−4.6% QFI at
                  N=256, dt_factor=1e-3). Legacy schemes ("drift", "diffusive_em") unchanged.
btc/qfi.py        trajectory QFI (power sums, checkpointing) + exact Lindblad / monitored
                  (Gammelmark–Mølmer) QFI reference (needs scipy, qutip).
btc/operators.py  collective operators, coherent initial states (needs qutip).
reduced_btc.py    REDUCED MODEL (numba): for Dicke Γ_mn=κ every sub-step is a rotation about
                  an axis common to all spins, so M=Σ s_n obeys a closed 3-vector SDE. Exact
                  rotations, |M| conserved to 1e-11, identical to the full-N stepper with the
                  same noise stream (1e-13), O(1) per step: 10^5 traj at N=256 in minutes.
                  Use this for the single collective spin; full-N code is for spatially
                  resolved decay matrices (2D arrays).
                    python reduced_btc.py N dt_factor n_traj up|down seed outdir
                  writes red_N*_seed*.npz with per-trajectory X at 200 times; pool with
                  reduced_csv.py / pool_100k.py (-> t,qfi,stderr,ntraj CSV).
sc_run_batches.py, sc_run_batches_cluster.sh, sc_plan.py, cost_model.py
                  full-N SLURM campaign runner (env: N, N_TRAJ, BATCH_SIZE, MASTER_SEED,
                  DT_FACTOR, CKPT, RUN_ID, STATE=up|down, STOP_AFTER). Default STATE=down
                  (ground) for backward compatibility; the exact reference uses "up".
sc_peek.py        reduce power-sum batches (+checkpoints) to QFI(t) CSV.
reduced_one.sh, reduced_dense.sh   SLURM one-core array wrappers for reduced_btc.py.
setup_env.sh      venv setup (numpy, scipy, tqdm; add numba for reduced_btc.py).

## Time step
With the exact-axis integrator dt_factor = 1e-2 (dt = 0.01/Ω) is converged: N=128/256 agree
with the exact monitored QFI to <1% at t=100 (10^4–10^5 trajectories). Small N (≤64) were run
at 1e-3 in the first pass and 1e-2 in the 100k pass; both agree.

## Initial state
"up" = |S,+S⟩ all excited (paper / exact reference); "down" = ground. Wigner-cone sampling
(arXiv:2305.19829 Eq. 6). Affects only an additive transient; the long-time rate is IC-independent.
