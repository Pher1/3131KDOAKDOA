#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTDIR="$ROOT/experiments/rlhf_bridge/results_paper"
mkdir -p "${OUTDIR}"

python "$ROOT/experiments/rlhf_bridge/run_shp_bridge.py" --task all --outdir "${OUTDIR}" --T 150 --n_seeds 3 --N 8 --m 16 --k_total 5 --max_features 2000 --eta0 1.0 --kappa 1.2

echo "[done] SHP bridge artifacts regenerated."
