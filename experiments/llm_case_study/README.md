
# OLE

## 📚 Contents

- [OLE](#ole)
  - [📚 Contents](#-contents)
  - [Quick Start ⚡](#quick-start-)
    - [Environment Setup](#environment-setup)
      - [FlashAttention (Required)](#flashattention-required)
    - [Datasets](#datasets)
    - [Training](#training)
    - [Key Hyperparameters](#key-hyperparameters)
    - [Evaluation](#evaluation)

---

## Quick Start ⚡

This section walks through the minimal steps to reproduce training and evaluation.

---

### Environment Setup

We recommend using **Python 3.10** with Conda.

```bash
conda create -n verl python=3.10 -y
conda activate verl
pip install -r requirements.txt
```

#### FlashAttention (Required)

This project relies on **FlashAttention 2.x** for efficient training.
Please install a **prebuilt wheel** matching your CUDA / PyTorch version.

```bash
# Download the wheel from:
# https://github.com/mjun0812/flash-attention-prebuild-wheels/releases

pip install flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

> ⚠️ Make sure your PyTorch and CUDA versions are compatible with the selected wheel.

---

### Datasets

All datasets are **preprocessed and ready to use** under:

```
./data
```

The training corpus is a mixture of:

* GSM8K
* MATH
* AIME 2024 / 2025
* AMC 2023

---

### Training

All training recipes are located under:

```
OLE/verl/recipe/dapo/
```

Example commands:

```bash
cd OLE/verl

# DAPO
bash verl/recipe/dapo/run_dapo_qwen2.5_ole_lora.sh

# GRPO
bash verl/recipe/dapo/run_grpo_qwen2.5_ole_lora.sh

# GPG
bash verl/recipe/dapo/run_gpg_qwen2.5_ole_lora.sh

# OPO
bash verl/recipe/dapo/run_opo_qwen2.5_ole_lora.sh
```

Each script supports toggling **OLE on/off** via environment variables.

---

### Key Hyperparameters

The following environment variables control core behavior:

```bash
# Enable OLE (1 = enabled, 0 = disabled)
export PPO_USE_OLE=1

# Optional experiment tracking
export SWANLAB_API_KEY=

# Ray data directory
export RAY_DATA_HOME=

# Base model path
export MODEL_PATH=
```

---

### Evaluation

Evaluation is performed using a LoRA-merged inference pipeline.

```bash
BASE_MODEL_PATH="PATH/TO/BASE/MODEL"
LORA_ROOT="PATH/TO/LORA/CHECKPOINTS"
STEP=XXX
```

Define the experiments to be evaluated:

```bash
EXPS=(
  "7b-dapo-with-ole-bz128"
  "7b-grpo-with-ole-bz128"
  "7b-gpg-with-ole-bz128"
  "7b-opo-with-ole-bz128"
  "7b-dapo-without-ole-bz128"
  "7b-grpo-without-ole-bz128"
  "7b-gpg-without-ole-bz128"
  "7b-opo-without-ole-bz128"
  #...
)
```

Run evaluation:

```bash
bash ./evaluation/eval.sh
```

Results will be saved to:

```
./results/{lora_name}/summary.json
```