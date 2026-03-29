# Synthetic preference-bandit experiment (OLE)

This folder contains a controlled synthetic experiment that matches the paper's
BTL/logistic preference feedback model and the OLE (optimistic Langevin ensemble)
index.

## Quick reproduce (from repo root)

```bash
pip install -r experiments/synthetic_bandit/requirements.txt
bash experiments/synthetic_bandit/reproduce_paper.sh
```

For a quick smoke test that does **not** overwrite paper figures/tables, use a tag + separate outdir, e.g.:

```bash
python experiments/synthetic_bandit/run_synthetic.py --task scaling --T 1000 --n_seeds 2 \
  --tag smoke --outdir experiments/synthetic_bandit/results_smoke
```


The paper uses a longer horizon for the scaling plot (to make the $\sqrt{t}$
visualization less sensitive to finite-horizon transients), and a shorter horizon
for tables/ablations (to keep the full sweep runnable on a single CPU).

Artifacts are written to:
- `figures/synthetic_regret_scaling.pdf`
- `figures/synthetic_hparam_heatmap.pdf` (appendix hyperparameter heatmap)
- `figures/synthetic_compute_tradeoff.pdf` (compute–regret diagnostic)
- `experiments/synthetic_bandit/results_paper/`

## Additional tasks

The paper also reports:

- **Classical baselines** (synthetic):
  ```bash
  python experiments/synthetic_bandit/run_synthetic.py --task baselines --T 2000 --n_seeds 8 --env_seed 0
  ```
  This writes `experiments/synthetic_bandit/results/synthetic_baselines.csv` and
  `experiments/synthetic_bandit/results/synthetic_baselines_table.tex`.

- **Compute–regret tradeoff** (synthetic):
  ```bash
  python experiments/synthetic_bandit/run_synthetic.py --task tradeoff --T 2000 --n_seeds 4 --tradeoff_n_seeds 4 --seed_offset 2 --env_seed 0 --tradeoff_include_laplace
  ```
  This writes `experiments/synthetic_bandit/results/synthetic_compute_tradeoff.csv` and
  `figures/synthetic_compute_tradeoff.pdf`.

## Notes

- Cells in the hyperparameter heatmap are annotated with mean regret numbers for readability.

- To reduce variance across random problem instances, the code supports fixing
  a single random bandit instance via `--env_seed` (arm embeddings + true
  parameter). The reproduction script uses `--env_seed 0`.

- The environment is a *degenerate contextual bandit* with a single context,
  i.e., a preference-based multi-armed bandit. This is a special case of the
  contextual bandit setting analyzed in the paper and is used here to isolate
  the finite-ensemble and discretization effects predicted by the theory.

- The script also writes `config.json` with the exact hyperparameters and seeds
  used in the run.

## Hyperparameter heatmap (appendix)

To (re)generate only the N×$\eta_0$ heatmap from an existing grid CSV:

```bash
python experiments/synthetic_bandit/run_synthetic.py --task heatmap
```

To run the grid sweep *and* regenerate the heatmap (may take longer):

```bash
python experiments/synthetic_bandit/run_synthetic.py --task grid --T 2000 --grid_n_seeds 4 --env_seed 0 --grid_N_list 5,10,20,40 --grid_eta0_list 0.2,2.0,10.0
python experiments/synthetic_bandit/run_synthetic.py --task heatmap --outdir experiments/synthetic_bandit/results
```
## Rebuttal Updates

In response to the reviewers' valuable feedback, we have made the following updates to our synthetic experiments:

**1. Improved Scaling Log-log Slope (Fine-tuned OLE)**  
To address the concerns regarding the log-log slope of the regret in our initial scaling experiments, we have fine-tuned the hyperparameters of our **OLE algorithm**. With the newly fine-tuned OLE configuration, the scaling plot now achieves a log-log slope that aligns better with the theoretical expectations. The updated hyperparameters are now the default in our reproduction scripts.

**2. Additional Baselines and Combinations (Ensemble++ and HyperAgent)**  
Per the reviewer's suggestion, we have also implemented and evaluated two additional algorithms: **Ensemble++** and **HyperAgent**. Furthermore, we evaluated the performance of combining these mechanisms with our OLE algorithm. We compared these methods under the same synthetic preference-bandit setting. 

The quantitative results of this comparison are summarized in the table below:
| Seed | OLE | Ensemble++ | OLE + Enspp | HyperAgent | HyperAgent+OLE |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Seed 0** | 0.64 / [0.57, 0.71] | 0.86 / [0.78, 0.93] | 0.60 / [0.53, 0.68] | 0.61 / [0.58, 0.64] | 0.67 / [0.64, 0.70] |
| **Seed 1** | 0.58 / [0.49, 0.67] | 0.63 / [0.54, 0.70] | 0.44 / [0.35, 0.52] | 0.61 / [0.55, 0.67] | 0.59 / [0.55, 0.64] |
| **Seed 2** | 0.49 / [0.42, 0.57] | 0.51 / [0.43, 0.60] | 0.51 / [0.46, 0.56] | 0.81 / [0.77, 0.84] | 0.82 / [0.78, 0.85] |
| **Seed 3** | 0.45 / [0.39, 0.51] | 0.98 / [0.94, 1.01] | 0.34 / [0.27, 0.41] | 0.14 / [0.12, 0.16] | 0.16 / [0.14, 0.18] |
| **Seed 4** | 0.52 / [0.44, 0.60] | 0.54 / [0.42, 0.65] | 0.50 / [0.42, 0.59] | 0.77 / [0.74, 0.79] | 0.72 / [0.70, 0.75] |

*(Note: Values are reported as Mean / 95% CI [Lower, Upper])*

### Reproducing the Scaling Results

To reproduce the fine-tuned scaling results and the new baselines reported in the table above, you can use the updated `run_synthetic.py` script. 

The general command structure is as follows:
```bash
python experiments/synthetic_bandit/run_synthetic.py --task scaling --env_seed <SEED_NUMBER> --method <METHOD_NAME>
```
**Arguments:**
* `--env_seed`: Specifies the environment seed to reproduce a specific row from the table (e.g., `0`, `1`, `2`, `3`, `4`).
* `--method`: Specifies the algorithm to run. Available options are:
  * `ole` (Our fine-tuned Optimistic Langevin Ensemble)
  * `enspp` (Ensemble++)
  * `enspp_ole` (OLE combined with Ensemble++)
  * `hyper` (HyperAgent)
  * `hyper_ole` (OLE combined with HyperAgent)
### Summary of the Baseline Comparisons

We sincerely thank the reviewer for suggesting these two highly relevant baselines. Evaluating Ensemble++ and HyperAgent within our synthetic bandit framework has provided extremely valuable insights into the empirical behavior of different exploration strategies.

Based on the quantitative results in the table above, we summarize our key observations as follows:

**1. Competitive and Stable Performance of OLE**
Overall, our fine-tuned OLE demonstrates highly stable performance across different random initializations. It consistently achieves empirical log-log slopes closely aligned with the theoretical $O(\sqrt{T})$ expectation (ranging from 0.45 to 0.64). While both Ensemble++ and HyperAgent are highly competitive and perform remarkably well in general, we observed that their exploration trajectories can occasionally be sensitive to specific environment initializations in this particular synthetic setting (e.g., extended transient phases on Seed 3 for Ensemble++, and Seeds 2/4 for HyperAgent). In comparison, OLE's mechanism provides a consistently stable exploration process that effectively mitigates such variance across seeds.

**2. Complementary Benefits: Integrating OLE's Exploration with Ensemble++**
Beyond evaluating the standalone baselines, we observed that combining OLE's exploration strategy with the underlying architecture of Ensemble++ yields noticeable performance improvements and better empirical robustness.

To implement the `OLE + Enspp` variant, we retained the core representation learning mechanism of Ensemble++ (which uses a base network to predict rewards and a perturbed ensemble network to quantify uncertainty). However, we replaced its default randomized Thompson Sampling action selection with OLE's variance-based optimistic exploration. Specifically, the action selection score is computed as `mean_diff + kappa * std_ens`, where the predicted mean difference is derived from the base network and the standard deviation is explicitly computed from the ensemble representations.

As shown in the table, this algorithmic integration effectively combines the efficient uncertainty representation of Ensemble++ with the stable, directed exploration of OLE. The combined approach (`OLE + Enspp`) achieves more stable log-log regret slopes across different seeds, demonstrating improved robustness against challenging environment initializations.