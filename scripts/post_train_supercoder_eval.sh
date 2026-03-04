#!/usr/bin/env bash
set -euo pipefail

# Run SuperCoder protocol evaluation for:
# 1) model under test
# 2) SuperCoder paper model baseline
#
# Usage:
#   bash scripts/post_train_supercoder_eval.sh "<model_under_test>" ["paper_model"]
#
# Example:
#   bash scripts/post_train_supercoder_eval.sh "mistral-hackaton-2026/codeforge"

MODEL_UNDER_TEST="${1:-}"
PAPER_MODEL="${2:-LLM4Code/Superoptimizer_Qwen7B}"
ROOT_DIR="${ROOT_DIR:-/content}"
SUPERCODER_DIR="${ROOT_DIR}/SuperCoder"

if [[ -z "${MODEL_UNDER_TEST}" ]]; then
  echo "ERROR: missing model_under_test"
  echo 'Usage: bash scripts/post_train_supercoder_eval.sh "<model_under_test>" ["paper_model"]'
  exit 1
fi

export HF_CACHE="${HF_CACHE:-${ROOT_DIR}/.cache/huggingface}"

echo "[1/5] Clone/update SuperCoder"
if [[ -d "${SUPERCODER_DIR}/.git" ]]; then
  git -C "${SUPERCODER_DIR}" pull --ff-only
else
  git clone https://github.com/Anjiang-Wei/SuperCoder "${SUPERCODER_DIR}"
fi

cd "${SUPERCODER_DIR}"

echo "[2/5] Install system deps"
apt-get update -y
apt-get install -y hyperfine build-essential

echo "[3/5] Install Python deps"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install datasets transformers sglang psutil tqdm tabulate

echo "[4/5] Evaluate model_under_test=${MODEL_UNDER_TEST}"
python src/evaluate.py \
  --model_name "${MODEL_UNDER_TEST}" \
  --inference_engine sglang \
  --split val \
  --num_workers 2 \
  --num_iterations 1 \
  --best_of 1 \
  --temperature 0.0 \
  --max_new_tokens 2000

echo "[4b/5] Evaluate paper_model=${PAPER_MODEL}"
python src/evaluate.py \
  --model_name "${PAPER_MODEL}" \
  --inference_engine sglang \
  --split val \
  --num_workers 2 \
  --num_iterations 1 \
  --best_of 1 \
  --temperature 0.0 \
  --max_new_tokens 2000

echo "[5/5] Show summaries"
python print_results.py results

MODEL_TAG="$(basename "${MODEL_UNDER_TEST}")"
PAPER_TAG="$(basename "${PAPER_MODEL}")"
BASE_DIR="${SUPERCODER_DIR}/results/main/llm_superoptimizer_ds_val"
MODEL_JSON="${BASE_DIR}/${MODEL_TAG}/num_iterations_1/0-shot/best_of_1/problem_results.json"
PAPER_JSON="${BASE_DIR}/${PAPER_TAG}/num_iterations_1/0-shot/best_of_1/problem_results.json"

echo "model_under_test_json=${MODEL_JSON}"
echo "paper_model_json=${PAPER_JSON}"

