# Collective-BTC QFI — N-convergence runners

> **This bundle carries the corrected diffusive drift.**
> Its `btc/sde.py` pairs the Stratonovich rotation noise with the matching
> Stratonovich drift (`collective_diffusive_drift_cartesian`): the full-Γ
> collective field plus the drive, and nothing else.  The pre-fix solver applied
> the Itô→Stratonovich conversion *twice* — once implicitly, by mapping the (θ,φ)
> Itô drift through the ordinary chain rule, once explicitly as
> `+½κ(sₓ, s_y, 2s_z)` — which injected a spurious tangential drift toward the
> equator.  `../N_convergence_cluster/` is the untouched pre-fix original, kept
> for comparison; **its semiclassical data is invalid and must be resampled here.**
> The deterministic/exact reference is solver-independent and was carried over
> verbatim, so `deterministic_qfi/` and `deterministic_qfi_ratio_4/` need no rerun.
> `convergence_check_fixed.ipynb` plots this bundle.
>
> The bundle carries **only** this scheme: the drift-only and `diffusive_em`
> collective variants (and the unused single-spin path) have been removed, so
> there is no `scheme=` argument any more. `../btc_fixed/` still has all three.
>
> Note: the `local_test.sh` mentioned below does not exist in the bundle.

Standalone bundle for the BTC QFI N-convergence study, split by **method**, each
with its own self-contained `.npz` format. Produces raw data only (no plots).

* **Semiclassical (diffusive)** — runs *batchwise & resumably* on the cluster.
  Each task computes a slice of trajectory batches and saves raw power sums; a
  reduce step combines them. You keep adding batches and re-reducing until the
  Monte-Carlo error is small enough — deciding "how far to go" dynamically.
* **Deterministic (monitored QFI)** — the exact quantum reference, run on the
  local workstation (it is single-threaded, so many cores do not help).

```
btc/                     trimmed btc package (semiclassical + quantum QFI)
cost_model.py            shared cost model + batch sizing (single source of truth)
sc_plan.py               SEMICLASSICAL: plan a fixed-N_TRAJ campaign -> sbatch commands
sc_run_batches.py        SEMICLASSICAL: compute+save one batch slice (power sums)
sc_run_batches.sh        SEMICLASSICAL: SLURM array (16-core tasks), N_TRAJ-driven
sc_aggregate.py          SEMICLASSICAL: reduce power-sum slices -> sc_diffusive_N<N>.npz
det_run.py               DETERMINISTIC: exact monitored QFI -> det_N<N>.npz
local_test.sh            end-to-end smoke test of the whole workflow (no SLURM)
convergence_check_fixed.ipynb   PLOTTING: dt- and N-convergence figures of this bundle
```

Requirements: `numpy`, `scipy`, `tqdm`. The deterministic runner also needs
`qutip`; the semiclassical path does not. `matplotlib` is optional (only for the
`sc_aggregate.py` diagnostic plot — aggregation still runs without it).

## Test locally first

`bash local_test.sh` runs the entire pipeline (plan -> emulated array of
sc_run_batches.sh -> resume-skip -> aggregate -> deterministic) with a tiny,
short-grid config in under a minute — no SLURM needed. It stands in for the
scheduler by exporting `SLURM_ARRAY_TASK_MIN/MAX/ID` and `SLURM_CPUS_PER_TASK`.
The run grid (`T`, `N_STORE`, `RATIO`, `KAPPA`, `DT_FACTOR`) is env-overridable
in `sc_run_batches.sh`, and `module load` runs only where `module` exists, so
the same script works locally and on the cluster.

## Physics / convention

Cabot–Carollo–Lesanovsky convention: `κ` fixed (N-independent), `Ω = ratio·κ·S`
with `S = N/2`, so `Ω/ω_c = ratio` stays fixed deep in the time-crystal phase.
Defaults: `κ=1, ratio=2, T=100, dt=0.01/Ω, n_store=200`.

## Semiclassical — the batchwise idea

The QFI is `4·Var(X)`, `X = ∂_ω𝒮` per trajectory, computed from **additive power
sums** `s1..s4`. A *batch* is the unit of parallel work (one whole batch per
core at a time); each batch is seeded from its **global index** alone
(`SeedSequence(master_seed, spawn_key=(g,))`). Therefore:

* disjoint batch slices never share trajectories — run them on any number of
  tasks/nodes, in any order;
* summing all the partial files reproduces *exactly* (to FP roundoff, verified
  ~1e-16) a single run over their union.

`batch_size` (trajectories per batch) is the RNG/reproducibility unit and
auto-scales *down* with N (a batch is the unit of parallel work — one per core
at a time — so it must stay small at expensive N: ~1 traj at N=1024, ~25 at
N=128). The **trajectory count is the knob** (`N_TRAJ`), not the batch size.

### Run on the cluster — fix the trajectory count

You choose `N_TRAJ` (the SAME for every N — the thesis-relevant quantity). For
each N it splits into `ceil(N_TRAJ/batch_size)` batches, partitioned evenly
across the SLURM array; the wall time is whatever it turns out to be. Let
`sc_plan.py` size the array and print the exact commands:

```bash
python3 sc_plan.py --n-traj 20000 --N 128 256 512 1024
```

```
     N  batch   total    actual  array  batches  est/task    core-h
   128     25     800     20000      7      115       3.9       394
   256     11    1819     20009     15      122       3.9       884
   512      3    6667     20001     53      126       4.0      3334
  1024      1   20000     20000    250       80       3.4     13667
```

It prints one `sbatch … --export=ALL,N=…,N_TRAJ=…,BATCH_SIZE=…` command per N —
copy them. The `core-h` column is the real cost (note N=1024 at 20k ≈ 13.7k
core-hours); pick `N_TRAJ` you can afford. Edit the `module load` line and set
`#SBATCH --time` to `sc_plan`'s `est/task` (tune with `--target-hours`).

The task→batch map is deterministic in `(N_TRAJ, array size, BATCH_SIZE)`, so a
re-submit **skips finished slices** (resumable) and the slices tile `[0,total)`
with no gaps/overlaps. Keep those three fixed for a campaign; to change `N_TRAJ`,
use a fresh `results/` dir. Partial files land in `results/` (override with
`RESULTS_DIR=…`); watch progress with `tail -f <job>.slurm.err`.

### Reduce & monitor convergence

Run anytime (even mid-campaign) to combine what has landed and print total
trajectories + relative QFI stderr at `t=T` per N:

```bash
python3 sc_aggregate.py --dir results
```

Writes one standalone `sc_diffusive_N<N>.npz` per N (`times, qfi, qfi_stderr,
n_traj, n_batches_total, …`) plus `sc_diffusive_qfi.png` — a quick
QFI(t)±stderr plot over all N for a swift eyeball check (`--no-plot` to skip,
`--show` to also open a window; needs matplotlib). It errors on overlapping
batch ranges (double-counting) and warns on gaps. If a finished campaign's
stderr is still too large, plan a larger `N_TRAJ` in a fresh `results/` dir.

#### Pooling several campaigns per N

By default each N must come from one campaign — a single `(master_seed,
batch_size)` — and anything else is an error. When a run instead varied the
*seed* per task (e.g. 16 tasks × 625 traj, `--master-seed 0…15`, each writing
batch 0), pass:

```bash
python3 sc_aggregate.py --dir results --combine-campaigns
```

Power sums are additive, so pooling independently seeded campaigns is exactly
one longer run. The safety argument is that a batch's trajectories come from
`SeedSequence(master_seed, spawn_key=(g,))` — `batch_size` does **not** enter
the seed — so a stream is identified by `(master_seed, g)` alone, and the
double-count check runs per master seed across all campaigns (two slices
sharing a seed must still hold disjoint batch ranges, even at different batch
sizes). Pooled slices must also agree on the time grid, on `(S, Ω, δ, κ, ratio,
T, dt)` and on `scheme/sampling/method/state`; mismatches are an error, since
the same N at two `dt` factors is two different simulations rather than more
statistics. The pooled file records `master_seeds`, `batch_sizes` and
`n_campaigns` (the scalar `master_seed`/`batch_size` are set to `-1`).

#### Importing reduced CSV tables

For data that arrives without its partials, `--from-csv` converts already-reduced
`t,qfi,stderr,ntraj` tables into the same `sc_diffusive_N<N>.npz` layout:

```bash
python3 sc_aggregate.py --from-csv --dir tables --csv-pattern '*dt1e-3*.csv'
```

`N` comes from the file name (`…N<N>…`) and the Ω dt factor from a `dt<factor>`
token if present; `Ω = ratio·κ·N/2` from `--ratio`/`--kappa` (defaults 2 and 1).
A reduced table carries no power sums, so it is converted one-to-one and can
never be summed with anything — use `--csv-pattern`/`--N` when several tables
map to the same N (they would collide on the output name).

### Choosing N_TRAJ

Relative QFI stderr ≈ `5%·√(1000/n_traj)` (so ~1% needs ~25k, ~0.5% needs ~100k).
Total core-time = `n_traj · per_traj` (per-traj: 71/159/600/2460 s for
N=128/256/512/1024); wall time = that ÷ (concurrent 16-core tasks). `sc_plan.py`
prints these `core-h` per N — e.g. N=1024 at 25k ≈ 17k core-hours — so you can
pick an `N_TRAJ` that is both statistically adequate and affordable.

## Deterministic — local workstation

Exact monitored-QFI solver; `qfi` is the **bare** F_Q (divide by S² to compare
with the semiclassical F_sc). Cost grows steeply with N — the largest sizes run
for days, which is what the per-N checkpointing is for.

```bash
python3 det_run.py 4 8 16 32 64 128 256 512
```

Writes standalone `det_N<N>.npz` (`times, qfi, qfi_stderr, N, S, Omega, kappa,
method, n_traj`; the last three are constant, kept for schema compatibility).
Resumable (skips existing unless `--overwrite`).

## Analysis

Load the two standalone formats side by side (e.g. in a notebook): compare
`sc_diffusive_N<N>.npz` `qfi` against `det_N<N>.npz` `qfi / S²`. Kept out of these
runners on purpose — they only produce raw data.
