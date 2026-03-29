# Experiments

This directory contains three experiments. For detailed instructions on how to run each experiment, please refer to the `README.md` in the respective subdirectories.

## 1. Synthetic Bandit

**Directory:** [`synthetic_bandit/`](./synthetic_bandit)

This folder contains a controlled synthetic experiment that matches the paper's BTL/logistic preference feedback model and the OLE (optimistic Langevin ensemble) index.

Please see [`synthetic_bandit/README.md`](./synthetic_bandit/README.md) for setup and running instructions.

## 2. RLHF Bridge

**Directory:** [`rlhf_bridge/`](./rlhf_bridge)

This experiment bridges the paper's preference-bandit theory to an RLHF-relevant public dataset **without** requiring large-scale model training. It uses the **Stanford Human Preferences (SHP)** dataset and a **frozen TF-IDF embedding** as the primary feature extractor.

Please see [`rlhf_bridge/README.md`](./rlhf_bridge/README.md) for setup and running instructions.

## 3. LLM Case Study (OLE)

**Directory:** [`llm_case_study/`](./llm_case_study)

This directory contains the implementation for OLE, including environment setup (FlashAttention required), datasets (GSM8K, MATH, etc.), training recipes (DAPO, GRPO, GPG, OPO), and evaluation.

Please see [`llm_case_study/README.md`](./llm_case_study/README.md) for detailed documentation.
