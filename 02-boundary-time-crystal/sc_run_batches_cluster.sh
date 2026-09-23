#!/bin/bash
#SBATCH --job-name=btc_sc
#SBATCH --output=%x_%A_%a.slurm.out
#SBATCH --error=%x_%A_%a.slurm.err
#SBATCH --array=0-9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=05:00:00
#SBATCH --mem=50gb

# ============================================================================
#  CLUSTER version of sc_run_batches.sh, following the conventions of the
#  proven old_TWA_cluster/two_levelV.sh (module load, lustre workspace for
#  results, parameter log, %x_%A_%a log naming).
#
#  One SLURM array task = one batch slice for a single N, driven by a fixed
#  trajectory count N_TRAJ. Plan the array size + exact submit line with:
#     python3 sc_plan.py --n-traj 20000 --N 512 1024 --script sc_run_batches_cluster.sh
#  which prints one command per N, e.g.:
#     sbatch --job-name=btc_sc_N512  --array=0-52  --cpus-per-task=16 \
#            --export=ALL,N=512,N_TRAJ=20000,BATCH_SIZE=3,MASTER_SEED=42 sc_run_batches_cluster.sh
#     sbatch --job-name=btc_sc_N1024 --array=0-249 --cpus-per-task=16 \
#            --export=ALL,N=1024,N_TRAJ=20000,BATCH_SIZE=1,MASTER_SEED=42 sc_run_batches_cluster.sh
#  Submit both — the two N campaigns run as independent arrays in parallel.
#  Set --time from sc_plan's est/task column (add margin), e.g. --time=05:00:00.
#
#  Keep --array, N_TRAJ, BATCH_SIZE, MASTER_SEED fixed for a given N: the
#  task->batch map is deterministic, so re-submitting skips finished slices
#  (resumable). To change N_TRAJ, use a fresh RUN_ID / results dir.
#  Reduce afterwards (locally or on the cluster):
#     python3 sc_aggregate.py --dir <results dir>
# ============================================================================

# ---- Inputs (via --export; see sc_plan.py) -------------------------------
: "${N:?set N, e.g. --export=ALL,N=512,N_TRAJ=20000}"
: "${N_TRAJ:?set N_TRAJ (total trajectories for this N)}"
MASTER_SEED=${MASTER_SEED:-42}                # campaign seed
BATCH_WALL_REF=${BATCH_WALL_REF:-1800}        # sets batch_size if BATCH_SIZE unset
BATCH_SIZE=${BATCH_SIZE:-}                    # pinned by sc_plan; else derived below

# ---- Environment (cluster: same pattern as two_levelV.sh) -----------------
module load devel/python/3.12.11
# One-time setup before first submission: run  bash setup_env.sh  on a login node
# (creates $HOME/btc-venv with numpy/scipy/tqdm; PIP_TARGET-safe).
[ -f "$HOME/btc-venv/bin/activate" ] && source "$HOME/btc-venv/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PYTHON=${PYTHON:-python3}
"$PYTHON" -c "import numpy, scipy, tqdm" 2>/dev/null \
    || { echo "ERROR: numpy/scipy/tqdm not importable — run 'bash setup_env.sh' on a login node first" >&2; exit 1; }

# SLURM runs the batch script from a spool copy, so BASH_SOURCE is useless
# there; use the submit dir (submit from the code directory!) or CODE_DIR.
SCRIPT_DIR="${CODE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"
NCORES=${SLURM_CPUS_PER_TASK:-16}

# ---- This task's slice of the fixed-N_TRAJ campaign ----------------------
AMIN=${SLURM_ARRAY_TASK_MIN:-0}
AMAX=${SLURM_ARRAY_TASK_MAX:-0}
AID=${SLURM_ARRAY_TASK_ID:-0}
K=$(( AMAX - AMIN + 1 ))          # number of array tasks
TIDX=$(( AID - AMIN ))            # 0-based task index

SLICE=$("$PYTHON" -c "import sys; sys.path.insert(0,'$SCRIPT_DIR'); import cost_model as c
r = c.task_slice($N_TRAJ, $N, $K, $TIDX, $NCORES, $BATCH_WALL_REF, ${BATCH_SIZE:-None})
print(r['batch_size'], r['total_batches'], r['batch_start'], r['n_batches'],
      r['actual_traj'], f\"{r['est_task_hours']:.1f}\")") \
    || { echo "ERROR: cost model failed for N=$N (untabulated? add it to cost_model.py PER_TRAJ)" >&2; exit 1; }
read -r BATCH_SIZE TOTAL_BATCHES BATCH_START N_BATCHES ACTUAL_TRAJ EST_WALL_H <<< "$SLICE"

# Extending an existing campaign: cost_model always tiles batches from 0, so a
# second submission would recompute indices the first one already owns -- and
# since a batch's trajectories come from its global index alone, that is a
# silent duplicate, not a harmless redo. BATCH_OFFSET shifts this submission's
# whole slice to untouched indices. Set it to the finished campaign's
# total_batches; N_TRAJ then counts only the NEW trajectories.
BATCH_OFFSET=${BATCH_OFFSET:-0}
BATCH_START=$(( BATCH_START + BATCH_OFFSET ))

if [ "${N_BATCHES:-0}" -le 0 ]; then
    echo "N=$N task $TIDX/$K: no batches assigned (array larger than needed); nothing to do."
    exit 0
fi

# ---- Run grid (fixed across the campaign; overridable via --export) -------
KAPPA=${KAPPA:-1.0}; RATIO=${RATIO:-2.0}; T=${T:-100.0}
DT_FACTOR=${DT_FACTOR:-0.01}; N_STORE=${N_STORE:-200}
STATE=${STATE:-down}   # initial coherent state: up (all excited) / down (all ground)

# Mid-trajectory checkpointing. CKPT=1 lets a task killed at its --time limit
# resume where it stopped when the identical sbatch line is resubmitted, so a
# long batch can be run as several short jobs. Off by default (unchanged
# behaviour); resume state lives beside the results and is deleted on success.
CKPT=${CKPT:-0}; CKPT_EVERY=${CKPT_EVERY:-1000000}

# ---- Directories (results on the lustre workspace, like two_levelV.sh) ----
# Create the workspace once on a login node, e.g.:  ws_allocate BTC 60
# then set WORKBASE to the path it prints (ws_find BTC).
RUN_ID=${RUN_ID:-1}
WORKBASE=${WORKBASE:-/lustre/work/ws/ws1/tu_ptivn01-BTC}   # <-- EDIT: your workspace
RESULTS_DIR="${RESULTS_DIR:-$WORKBASE/${RUN_ID}/N_${N}}"
mkdir -p "$RESULTS_DIR"
cd "$RESULTS_DIR"

# ---- Parameter log (saved once per results dir, as in two_levelV.sh) ------
PARAMFILE="$RESULTS_DIR/parameters.txt"
if [ ! -f "$PARAMFILE" ]; then
    {
        echo "Run ID: $RUN_ID"
        echo "Date: $(date)"
        echo ""
        echo "N = $N"
        echo "N_TRAJ = $N_TRAJ"
        echo "BATCH_SIZE = $BATCH_SIZE"
        echo "MASTER_SEED = $MASTER_SEED"
        echo ""
        echo "KAPPA = $KAPPA"
        echo "RATIO = $RATIO   (Omega = ratio * kappa * N/2)"
        echo "T = $T"
        echo "DT_FACTOR = $DT_FACTOR"
        echo "N_STORE = $N_STORE"
        echo "STATE = $STATE"
        echo ""
        echo "array size K = $K"
        echo "total batches = $TOTAL_BATCHES (campaign = $ACTUAL_TRAJ traj)"
    } > "$PARAMFILE"
    echo "-> Wrote parameters to $PARAMFILE"
fi

echo "N=$N task $TIDX/$K: batch_size=$BATCH_SIZE, batches " \
     "[$BATCH_START,$((BATCH_START + N_BATCHES))) of $TOTAL_BATCHES " \
     "(campaign = $ACTUAL_TRAJ traj); this task $((N_BATCHES * BATCH_SIZE)) traj, " \
     "est ~${EST_WALL_H} h -> $RESULTS_DIR"

# ---- Run one checkpointed slice ------------------------------------------
# Resume state is keyed by array task index, and a resubmitted array gives task
# t the same batches, so task t always finds its own checkpoints.
CKPT_ARGS=()
if [ "$CKPT" != "0" ]; then
    CKPT_ARGS=(--ckpt-dir "$RESULTS_DIR/ckpt_t${TIDX}" --ckpt-every "$CKPT_EVERY")
    echo "checkpointing ON: every $CKPT_EVERY steps -> $RESULTS_DIR/ckpt_t${TIDX}"
fi

# Self-stop for dependency chains. Slurm's per-task chaining (--dependency=
# aftercorr:<jobid>) only fires when the parent task exits 0, and a TIMEOUT
# does not. With STOP_AFTER=<seconds> (choose ~30 min below --time) the python
# step is stopped cleanly before the limit and this task exits 0, so the next
# window in the chain starts right away and resumes from the checkpoints.
# Checkpoint files are written atomically (tmp + os.replace), so a kill
# between checkpoints loses at most CKPT_EVERY steps of work.
STOP_AFTER=${STOP_AFTER:-}
RUNNER=()
if [ -n "$STOP_AFTER" ]; then
    RUNNER=(timeout --signal=TERM --kill-after=60 "$STOP_AFTER")
    echo "self-stop ON: will stop after ${STOP_AFTER}s and exit 0 (resume via aftercorr chain)"
fi

"${RUNNER[@]}" "$PYTHON" "$SCRIPT_DIR/sc_run_batches.py" "$N" \
    --batch-start "$BATCH_START" --n-batches "$N_BATCHES" \
    --batch-size "$BATCH_SIZE" --master-seed "$MASTER_SEED" \
    --kappa "$KAPPA" --ratio "$RATIO" --T "$T" --dt-factor "$DT_FACTOR" \
    --n-store "$N_STORE" --state "$STATE" --n-jobs "$NCORES" --outdir "$RESULTS_DIR" \
    ${CKPT_ARGS[@]+"${CKPT_ARGS[@]}"}
RC=$?
if [ -n "$STOP_AFTER" ] && { [ "$RC" -eq 124 ] || [ "$RC" -eq 137 ]; }; then
    echo "Stopped by self-stop timer after ${STOP_AFTER}s (rc=$RC) at $(date); checkpoints kept, exiting 0 for the chain"
    exit 0
fi

echo "Job finished at $(date) (rc=$RC)"
exit "$RC"
