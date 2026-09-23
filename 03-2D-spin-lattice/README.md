# Driven atomic array — semiclassical QFI

Quantum Fisher information for sensing the Rabi frequency Ω of a coherently
driven, radiatively coupled 2-D atomic array (Sect. 6.1 of Mink & Fleischhauer,
*Collective Radiative Interactions in the DTWA*, arXiv:2305.19829).  The
semiclassical (truncated-Wigner) QFI scales to large lattices (10×10) where the
exact master equation is out of reach; it is benchmarked against exact / MCWF
references at small lattices (≤4×4).

This mirrors the boundary-time-crystal bundle in
`../02-boundary-time-crystal/N_convergence_cluster/`, split by **method**, each
with its own self-contained `.npz` format, producing raw data only (no plots in
the runners).

```
atomarray/               self-contained package (semiclassical + exact/MCWF QFI)
  geometry.py            lattice positions and Green's-tensor rate matrices Γ, J
  sde.py                 pole-safe Cartesian rotation SDE stepper (Eqs. 48/49/51)
  operators.py           sparse many-body operators for the MCWF reference
  qfi.py                 semiclassical QFI power sums + exact/MCWF monitored QFI
cost_model.py            shared cost model + batch sizing (single source of truth)
sc_plan.py               SEMICLASSICAL: plan a fixed-N_TRAJ campaign -> sbatch cmds
sc_run_batches.py        SEMICLASSICAL: compute+save one batch slice (power sums)
sc_run_batches.sh        SEMICLASSICAL: SLURM array (per-L, N_TRAJ-driven)
sc_aggregate.py          SEMICLASSICAL: reduce slices -> sc_array_L<L>.npz
mcwf_run.py              REFERENCE: exact (L≤3) / MCWF (4×4) monitored QFI per L
local_test.sh            end-to-end smoke test of the whole workflow (no SLURM)
```

Requirements: `numpy`, `scipy`, `tqdm` (`matplotlib` optional, only for the
diagnostic plots).  No QuTiP.

## Model and convention

`N = L²` two-level atoms on a square lattice (spacing `a = 0.8 λe`), circular
in-plane polarisation `e_p = (1, i, 0)/√2`, driven on resonance by a plane wave
perpendicular to the array so every Rabi frequency equals Ω.  Master equation
(Eqs. 44/45) with the free-space decay matrix Γ and dipole–dipole matrix J
(Eq. 47).  The atoms start in the collective ground state |↓…↓⟩.

**Sensed parameter:** the Rabi frequency Ω, generator ∂_Ω H = −Σ_n σ^x_n.
* Semiclassical: `F_sc(T) = 4·Var_traj(∂_Ω 𝒮)`, `∂_Ω 𝒮 = ∫₀ᵀ Σ_n s^x_n dt`.
* Quantum: the Gammelmark–Mølmer **monitored** QFI `F_mon(T)` (the proper
  counterpart of the trajectory-action QFI).

Both use the same generator, so **compare `F_sc ≈ F_mon` directly** (both
unnormalised).  Divide by `N` (or `S² = (N/2)²`) for a per-atom view.

Defaults: `a=0.8, Ω=2Γ0, Δ=0, Γ0=1, T=10, dt=0.02/max(2Ω, Γ0), n_store=200`.

## Test locally first

`bash local_test.sh` runs the entire pipeline (plan → emulated array of
`sc_run_batches.sh` → resume-skip → aggregate → exact reference) with a tiny
grid in seconds — no SLURM.  It stands in for the scheduler by exporting
`SLURM_ARRAY_TASK_MIN/MAX/ID` and `SLURM_CPUS_PER_TASK`; the run grid
(`T, OMEGA, DT_FACTOR, N_STORE, A_SPACING`) is env-overridable in
`sc_run_batches.sh`, and `module load` runs only where `module` exists.

## Semiclassical — the batchwise idea

Identical to the BTC bundle: the QFI is `4·Var(X)`, `X = ∂_Ω 𝒮` per trajectory,
from **additive power sums** `s1..s4`.  A *batch* is the unit of parallel work,
seeded from its **global index** alone
(`SeedSequence(master_seed, spawn_key=(g,))`), so disjoint slices never share
trajectories and summing the partial files reproduces exactly (to FP roundoff) a
single run over their union.  `N_TRAJ` (same for every L) is the knob.

> **The array is cheap.** Unlike the BTC (where Ω∝N drives `dt` and `n_time`),
> here Ω and `dt` are N-independent, so only the O(N²) collective field grows: a
> 10×10 campaign with 20 000 trajectories is **well under one core-hour**.  The
> whole semiclassical study can run locally; the SLURM machinery is kept for
> consistency and for pushing `N_TRAJ` very high.

### Run

```bash
python3 sc_plan.py --n-traj 20000 --L 2 3 4 6 8 10       # sizing + sbatch cmds
# locally, one L at a time (no SLURM needed):
python3 sc_run_batches.py 10 --n-traj 20000 --n-jobs -1 --outdir results
# or submit the printed sbatch line per L, then:
python3 sc_aggregate.py --dir results                    # -> sc_array_L<L>.npz
```

`sc_aggregate.py` writes one standalone `sc_array_L<L>.npz` per L (`times, qfi,
qfi_stderr, n_traj, …`) plus a quick `F_sc(t)/N` plot; it errors on overlapping
batch ranges and warns on gaps.  Relative QFI stderr falls as `1/√n_traj`.

## Reference — exact / MCWF monitored QFI (small L, local)

```bash
python3 mcwf_run.py 2 3            # exact density-matrix hierarchy (no MC/dt bias)
python3 mcwf_run.py 4 --n-traj-mcwf 1000 --T 6   # 4×4: statevector MCWF (see below)
```

`--method auto` (default) uses the **exact** engine for `N ≤ 9` (2×2, 3×3 — no
Monte-Carlo or time-step bias) and statevector **MCWF** above (the 4ᴺ density
matrix is infeasible at 4×4; only the 2ᴺ statevector is).  Writes one standalone
`mcwf_L<L>.npz` (`times, qfi, qfi_stderr, method, …`).

**4×4 (N=16) caveats.** The statevector MCWF is the *only* feasible reference —
budget a few core-hours to overnight (∝ `n_traj · T / dt`), so start with a
modest `--n-traj-mcwf 1000 --T 6`.  It carries an `O(dt)` unravelling bias
(default `dt = 0.005/max(2Ω,Γ0)` keeps it ≲2%; verify by halving `--dt-factor-mcwf`)
and the monitored-QFI integrand has heavy tails at late times, so its stderr
grows with T — increase `n_traj` if the band is wide.

## Validation (what to expect)

At 2×2 and 3×3 the exact monitored QFI is available with no MC/`dt` bias:

| L | N | F_sc vs exact F_mon at Γ0t≈3 |
|---|---|------------------------------|
| 2 | 4 | ~15–18 % (TWA truncation) |
| 3 | 9 | ~15 % |

The semiclassical QFI tracks the monitored QFI closely at early times and
overshoots by an `O(1/√N)` truncation error that shrinks with N and is smallest
at moderate-to-strong driving (`Ω/Γ0 ≳ 1`) — exactly the regime the method
targets.  The mean magnetisations ⟨S_a⟩(t) match the exact `mesolve` result to
the same order, confirming the drift/noise.

## Analysis

Load the two standalone formats side by side (e.g. in a notebook): compare
`sc_array_L<L>.npz` `qfi` against `mcwf_L<L>.npz` `qfi` **directly** (same
generator, both unnormalised), or both `/N` for a per-atom view.  Kept out of the
runners on purpose — they only produce raw data.
