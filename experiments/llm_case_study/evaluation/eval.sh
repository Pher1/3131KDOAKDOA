#!/usr/bin/env bash
set -euo pipefail

# =========================
# GPU plan
#   - Generation: vLLM server uses ALL 8 GPUs (TP=8)
#   - Eval: RM uses ONE GPU (e.g., GPU0)
# =========================
VLLM_GPUS="0,1,2,3"
TP_SIZE=4

RM_GPU="3"          # only one GPU visible for eval stage
RM_DEVICE="cuda:0"  # within eval stage, cuda:0 is that visible GPU

BASE_MODEL_PATH="PATH TO YOUR MODEL"
LORA_ROOT="PATH TO YOUR LORA CKPTS"
STEP=

TEST_PATH="./data/merged_eval.parquet"
MAX_NEW_TOKENS=8192
TEMPERATURE=1.0
TOP_P=0.7
N_SAMPLES=5
CONCURRENCY=512

VERIFY_WORKERS=64
RM_BATCH_SIZE=32
RM_MAX_LENGTH=8192

NUM_TEST=0  # use --num to modify NUM_TEST

REWARD_MODEL_PATH="Skywork-Reward-Llama-3.1-8B-v0.2"

HOST="127.0.0.1"
PORT="8000"

# name of loras
EXPS=(
  "7b-dapo-with-ole-bz128"
  "7b-grpo-with-ole-bz128"
  "7b-gpg-with-ole-bz128"
  "7b-opo-with-ole-bz128"
  "7b-dapo-without-ole-bz128"
  "7b-grpo-without-ole-bz128"
  "7b-gpg-without-ole-bz128"
  "7b-opo-without-ole-bz128"
)

mkdir -p results/_server
SERVER_LOG="results/_server/server.log"
SERVER_PID=""

is_running() { kill -0 "$1" >/dev/null 2>&1; }

stop_pid_graceful() {
  local pid="$1" name="$2" term_to="${3:-15}" kill_to="${4:-5}"
  [[ -z "${pid}" ]] && return 0
  ! is_running "${pid}" && return 0
  echo "[INFO] Stopping ${name} pid=${pid} (TERM ${term_to}s)..."
  kill -TERM "${pid}" >/dev/null 2>&1 || true
  for _ in $(seq 1 "${term_to}"); do
    ! is_running "${pid}" && { echo "[INFO] ${name} stopped."; return 0; }
    sleep 1
  done
  echo "[WARN] ${name} still alive; KILL ${kill_to}s..."
  kill -KILL "${pid}" >/dev/null 2>&1 || true
  for _ in $(seq 1 "${kill_to}"); do
    ! is_running "${pid}" && { echo "[INFO] ${name} killed."; return 0; }
    sleep 1
  done
  echo "[WARN] ${name} pid=${pid} still alive."
}

cleanup_server() {
  if [[ -n "${SERVER_PID}" ]]; then
    stop_pid_graceful "${SERVER_PID}" "vLLM server" 15 5 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
    SERVER_PID=""
  fi
}
trap cleanup_server EXIT INT TERM

# ---------- args ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --num) NUM_TEST="${2:-0}"; shift 2;;
    *) echo "[ERROR] Unknown argument: $1"; exit 1;;
  esac
done

echo "================================================================================"
echo "[INFO] vLLM GPUs: ${VLLM_GPUS} | TP=${TP_SIZE}"
echo "[INFO] RM GPU:   ${RM_GPU} | rm_batch_size=${RM_BATCH_SIZE}"
echo "[INFO] Base:     ${BASE_MODEL_PATH}"
echo "[INFO] Test:     ${TEST_PATH}"
echo "================================================================================"

for EXP_NAME in "${EXPS[@]}"; do
  OUT_DIR="results/${EXP_NAME}"
  mkdir -p "${OUT_DIR}"

  CKPT_DIR="${LORA_ROOT}/${EXP_NAME}/global_step_${STEP}/actor/lora_adapter"
  [[ -d "${CKPT_DIR}" ]] || { echo "[ERROR] LoRA dir not found: ${CKPT_DIR}"; exit 1; }

  GEN_JSON="${OUT_DIR}/generations.json"
  DETAIL_JSON="${OUT_DIR}/detail.json"
  SUMMARY_JSON="${OUT_DIR}/summary.json"
  EVAL_LOG="${OUT_DIR}/eval.log"

  echo "--------------------------------------------------------------------------------"
  echo "[EXP] ${EXP_NAME}"
  echo "[EXP] LoRA: ${CKPT_DIR}"
  echo "--------------------------------------------------------------------------------"

  # -------- 1) Generation (skip if exists) --------
  if [[ -f "${GEN_JSON}" ]]; then
    echo "[INFO] Found ${GEN_JSON}, skip generation."
  else
    echo "[INFO] Start vLLM server for generation..."
    : > "${SERVER_LOG}"

    CUDA_VISIBLE_DEVICES="${VLLM_GPUS}" \
    nohup vllm serve "${BASE_MODEL_PATH}" \
      --host "${HOST}" \
      --port "${PORT}" \
      --tensor-parallel-size "${TP_SIZE}" \
      --max-model-len 32768 \
      --gpu-memory-utilization 0.80 \
      --enable-lora \
      --max-loras 1 \
      --lora-modules "${EXP_NAME}=${CKPT_DIR}" \
      1>/dev/null 2>>"${SERVER_LOG}" < /dev/null &

    SERVER_PID=$!
    echo "[INFO] vLLM server started pid=${SERVER_PID}."

    echo "[INFO] Waiting server ready..."
    READY=0
    for i in $(seq 1 300); do
      if curl -fsS "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; then READY=1; break; fi
      if ! is_running "${SERVER_PID}"; then echo "[ERROR] server died. Check ${SERVER_LOG}"; exit 1; fi
      sleep 2
    done
    [[ "${READY}" -eq 1 ]] || { echo "[ERROR] server not ready. Check ${SERVER_LOG}"; exit 1; }

    NUM_TEST_ARGS=()
    [[ "${NUM_TEST}" -gt 0 ]] && NUM_TEST_ARGS=(--num "${NUM_TEST}")

    echo "[INFO] generate_only -> ${GEN_JSON}"
    CUDA_VISIBLE_DEVICES="" \
    python evaluation/generate_only.py \
      --base_url "http://${HOST}:${PORT}/v1" \
      --api_key "NOT A REAL KEY" \
      --temperature "${TEMPERATURE}" \
      --top_p "${TOP_P}" \
      --max_tokens "${MAX_NEW_TOKENS}" \
      --n_samples "${N_SAMPLES}" \
      --concurrency "${CONCURRENCY}" \
      "${NUM_TEST_ARGS[@]}" \
      --model "${EXP_NAME}" \
      --test_file "${TEST_PATH}" \
      --out_json "${GEN_JSON}"

    echo "[INFO] Stop server after generation."
    cleanup_server
  fi

  # -------- 2) Eval (single-GPU RM) --------
  : > "${EVAL_LOG}"
  echo "[INFO] eval_only -> ${DETAIL_JSON}, ${SUMMARY_JSON}"

  CUDA_VISIBLE_DEVICES="${RM_GPU}" \
  python evaluation/eval_only.py \
    --gen_json "${GEN_JSON}" \
    --verify_workers "${VERIFY_WORKERS}" \
    --rm_path "${REWARD_MODEL_PATH}" \
    --rm_device "${RM_DEVICE}" \
    --rm_batch_size "${RM_BATCH_SIZE}" \
    --rm_max_length "${RM_MAX_LENGTH}" \
    --detail_json "${DETAIL_JSON}" \
    --summary_json "${SUMMARY_JSON}" \
    >> "${EVAL_LOG}" 2>&1

  echo "[INFO] Done ${EXP_NAME} | log=${EVAL_LOG}"
done

echo "[DONE] All experiments finished."
