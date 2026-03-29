#!/usr/bin/env bash
set -euo pipefail

# Reproduce the synthetic preference-bandit artifacts used in the paper.
# Can be run from any working directory.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# We fix a single random bandit instance (arm embeddings + true parameter)
# for all seeds to reduce instance-to-instance variance.
ENV_SEED=0

# Default hyperparameters for the paper.
BETA=1.0
KAPPA=0.3
M=32
N=20
ETA0=1.0

# Table horizons.
T_TABLE=2000
N_SEEDS_TABLE=8

# Scaling plot horizon (longer to make the √t trend easier to see).
T_SCALING=5000
N_SEEDS_SCALING=8

# Output directory for paper artifacts (kept separate from scratch runs).
OUTDIR="$ROOT/experiments/synthetic_bandit/results_paper"
mkdir -p "${OUTDIR}"

python "$ROOT/experiments/synthetic_bandit/run_synthetic.py" \
  --task scaling \
  --T ${T_SCALING} --n_seeds ${N_SEEDS_SCALING} \
  --env_seed ${ENV_SEED} \
  --beta ${BETA} --kappa ${KAPPA} --m ${M} --N ${N} --eta0 ${ETA0} \
  --outdir ${OUTDIR}

cp "${OUTDIR}/config.json" \
   ${OUTDIR}/config_scaling.json

python "$ROOT/experiments/synthetic_bandit/run_synthetic.py" \
  --task impl \
  --T ${T_TABLE} --n_seeds ${N_SEEDS_TABLE} \
  --env_seed ${ENV_SEED} \
  --beta ${BETA} --kappa ${KAPPA} --m ${M} --N ${N} --eta0 ${ETA0} \
  --outdir ${OUTDIR}

cp "${OUTDIR}/config.json" \
   ${OUTDIR}/config_impl.json

python "$ROOT/experiments/synthetic_bandit/run_synthetic.py" \
  --task minibatch \
  --T ${T_TABLE} --n_seeds ${N_SEEDS_TABLE} \
  --env_seed ${ENV_SEED} \
  --beta ${BETA} --kappa ${KAPPA} --m ${M} --N ${N} --eta0 ${ETA0} \
  --outdir ${OUTDIR}

cp "${OUTDIR}/config.json" \
   ${OUTDIR}/config_minibatch.json

python "$ROOT/experiments/synthetic_bandit/run_synthetic.py" \
  --task ablation \
  --T ${T_TABLE} --n_seeds ${N_SEEDS_TABLE} \
  --env_seed ${ENV_SEED} \
  --beta ${BETA} --kappa ${KAPPA} --m ${M} --N ${N} --eta0 ${ETA0} \
  --outdir ${OUTDIR}

cp "${OUTDIR}/config.json" \
   ${OUTDIR}/config_ablation.json

python "$ROOT/experiments/synthetic_bandit/run_synthetic.py" \
  --task baselines \
  --T ${T_TABLE} --n_seeds ${N_SEEDS_TABLE} \
  --env_seed ${ENV_SEED} \
  --beta ${BETA} --kappa ${KAPPA} --m ${M} --N ${N} --eta0 ${ETA0} \
  --outdir ${OUTDIR}

cp "${OUTDIR}/config.json" \
   ${OUTDIR}/config_baselines.json

python "$ROOT/experiments/synthetic_bandit/run_synthetic.py" \
  --task tradeoff \
  --T ${T_TABLE} --n_seeds 4 --tradeoff_n_seeds 4 --seed_offset 2 \
  --env_seed ${ENV_SEED} \
  --beta ${BETA} --kappa ${KAPPA} --m ${M} --N ${N} --eta0 ${ETA0} \
  --tradeoff_N_list 5,10,20,40 --tradeoff_m_list 8,32 \
  --tradeoff_include_laplace \
  --outdir ${OUTDIR}

cp "${OUTDIR}/config.json" \
   ${OUTDIR}/config_tradeoff.json

# Appendix hyperparameter grid and heatmap.
python "$ROOT/experiments/synthetic_bandit/run_synthetic.py" \
  --task grid \
  --T ${T_TABLE} --n_seeds ${N_SEEDS_TABLE} --grid_n_seeds 4 \
  --env_seed ${ENV_SEED} \
  --beta ${BETA} --kappa ${KAPPA} --m ${M} --N ${N} --eta0 ${ETA0} \
  --outdir ${OUTDIR}

cp "${OUTDIR}/config.json" \
   ${OUTDIR}/config_grid.json

python "$ROOT/experiments/synthetic_bandit/run_synthetic.py" \
  --task heatmap \
  --T ${T_TABLE} --n_seeds ${N_SEEDS_TABLE} \
  --env_seed ${ENV_SEED} \
  --beta ${BETA} --kappa ${KAPPA} --m ${M} --N ${N} --eta0 ${ETA0} \
  --outdir ${OUTDIR}

cp "${OUTDIR}/config.json" \
   ${OUTDIR}/config_heatmap.json

echo "[done] Synthetic artifacts regenerated."
