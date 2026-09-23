#!/bin/bash
# One-time environment setup for the BTC semiclassical runs.
# Run ONCE on a LOGIN node:  bash setup_env.sh
# Rebuilds $HOME/btc-venv from scratch. Do NOT run inside a job.
set -e

module load devel/python/3.12.11   # keep identical to sc_run_batches_cluster.sh

# The cluster sets PIP_TARGET, which hijacks pip installs even inside a venv.
unset PIP_TARGET

rm -rf "$HOME/btc-venv"
python3 -m venv "$HOME/btc-venv"
"$HOME/btc-venv/bin/pip" install --upgrade pip
"$HOME/btc-venv/bin/pip" install numpy scipy tqdm
"$HOME/btc-venv/bin/python" -c "import numpy, scipy, tqdm; print('deps OK')"
echo "venv ready at $HOME/btc-venv"
