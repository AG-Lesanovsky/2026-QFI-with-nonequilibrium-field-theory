#!/bin/bash
#SBATCH --job-name=btc_red
#SBATCH --output=%x_%A_%a.slurm.out --error=%x_%A_%a.slurm.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=2G
module load devel/python/3.12.11; source $HOME/btc-venv/bin/activate; export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
cd $HOME/BTC_cluster; python -u reduced_btc.py $N $DTF 625 ${STATE:-down} $SLURM_ARRAY_TASK_ID $HOME/reduced_out
