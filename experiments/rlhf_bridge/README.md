# RLHF-bridging experiment (lightweight)

This experiment bridges the paper's preference-bandit theory to an
RLHF-relevant public dataset **without** requiring large-scale model training.

We use the **Stanford Human Preferences (SHP)** dataset (an RLHF-style
preference dataset mined from Reddit) and a **frozen TF-IDF embedding** as the
primary feature extractor.

To keep the package self-contained, this repo ships the small SHP
``askphysics`` validation/test files used by the paper in
``experiments/rlhf_bridge/data/``. If you delete them, the script will
re-download them from Hugging Face.

## Setup

From the repository root:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r experiments/rlhf_bridge/requirements.txt
```

## Run

```bash
bash experiments/rlhf_bridge/reproduce_paper.sh

# or run the driver directly:
# python experiments/rlhf_bridge/run_shp_bridge.py --task all --outdir experiments/rlhf_bridge/results_custom
```

This will:

1. Load the SHP ``askphysics`` validation/test splits (downloaded only if missing).
2. Build a small multi-action preference-bandit environment by grouping
   responses by prompt and selecting a fixed baseline per prompt.
3. Run OLE and baselines (posterior mean, Thompson sampling) and save:
   - `figures/shp_bridge_regret.pdf`
   - `figures/shp_bridge_regret_gap.pdf`
   - `experiments/rlhf_bridge/results_paper/shp_bridge_metrics.json`

The paper uses the configuration in `reproduce_paper.sh` and records the exact hyperparameters in `results_paper/shp_bridge_metrics.json`.

## Optional: frozen MiniLM/SBERT features

Our theory only assumes a fixed feature map, so the experiment can swap
TF--IDF for a frozen sentence encoder.

To enable the optional MiniLM/SBERT encoder (no finetuning):

```bash
pip install transformers
python experiments/rlhf_bridge/run_shp_bridge.py --encoder sbert --tag sbert
```

The first run will download the model weights (Hugging Face cache). The script
also caches embeddings to ``experiments/rlhf_bridge/data/`` for speed.

## Notes

* The download step uses the Hugging Face-hosted SHP repository.
* SHP contains naturally occurring text and may include content that some users
  find sensitive. This script does not print dataset samples by default.
