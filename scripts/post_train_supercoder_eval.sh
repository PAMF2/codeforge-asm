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
SUPERCODER_VENV_DIR="${SUPERCODER_VENV_DIR:-${ROOT_DIR}/.venv-supercoder}"
USE_VENV="${USE_VENV:-1}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"
INFERENCE_ENGINE="${INFERENCE_ENGINE:-sglang}"
NUM_WORKERS="${NUM_WORKERS:-2}"
SUPERCODER_BASE_MODEL="${SUPERCODER_BASE_MODEL:-}"

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
if [[ "${USE_VENV}" == "1" ]]; then
  if [[ ! -d "${SUPERCODER_VENV_DIR}" ]]; then
    python -m venv "${SUPERCODER_VENV_DIR}"
  fi
  # shellcheck disable=SC1090
  source "${SUPERCODER_VENV_DIR}/bin/activate"
fi

if [[ "${SKIP_INSTALL}" != "1" ]]; then
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
  python -m pip install datasets transformers sglang psutil tqdm tabulate peft accelerate
fi

echo "[3b/5] Resolve eval model (supports full model or LoRA adapter)"
MODEL_UNDER_TEST_RESOLVED="$(python - "${MODEL_UNDER_TEST}" "${ROOT_DIR}" "${SUPERCODER_BASE_MODEL}" <<'PY'
import os
import sys
from pathlib import Path

model_id = sys.argv[1]
root_dir = sys.argv[2]
base_override = sys.argv[3].strip()
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

def has_architectures(target: str) -> bool:
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(target, token=token, trust_remote_code=True)
        arch = getattr(cfg, "architectures", None)
        return bool(arch)
    except Exception:
        return False

if has_architectures(model_id):
    print(model_id)
    sys.exit(0)

try:
    from peft import PeftConfig
except Exception as exc:
    raise SystemExit(f"Cannot load peft to inspect adapter '{model_id}': {exc}")

try:
    peft_cfg = PeftConfig.from_pretrained(model_id, token=token)
except Exception as exc:
    raise SystemExit(
        f"Model '{model_id}' has no architectures and is not a readable PEFT adapter: {exc}"
    )

base_model = base_override or getattr(peft_cfg, "base_model_name_or_path", "")
if not base_model:
    raise SystemExit(
        "Could not infer base model from adapter. Set SUPERCODER_BASE_MODEL to force it."
    )

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

safe_tag = model_id.replace("/", "__").replace(":", "_")
out_dir = Path(root_dir) / "merged_models" / safe_tag
if (out_dir / "config.json").exists() and has_architectures(str(out_dir)):
    print(str(out_dir))
    sys.exit(0)

out_dir.mkdir(parents=True, exist_ok=True)

base = AutoModelForCausalLM.from_pretrained(
    base_model,
    token=token,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)
adapted = PeftModel.from_pretrained(base, model_id, token=token)
merged = adapted.merge_and_unload()
merged.save_pretrained(out_dir, safe_serialization=True)

try:
    tok = AutoTokenizer.from_pretrained(model_id, token=token, trust_remote_code=True)
except Exception:
    tok = AutoTokenizer.from_pretrained(base_model, token=token, trust_remote_code=True)
tok.save_pretrained(out_dir)

print(str(out_dir))
PY
)"
echo "model_under_test_resolved=${MODEL_UNDER_TEST_RESOLVED}"

echo "[4/5] Evaluate model_under_test=${MODEL_UNDER_TEST_RESOLVED}"
python src/evaluate.py \
  --model_name "${MODEL_UNDER_TEST_RESOLVED}" \
  --inference_engine "${INFERENCE_ENGINE}" \
  --split val \
  --num_workers "${NUM_WORKERS}" \
  --num_iterations 1 \
  --best_of 1 \
  --temperature 0.0 \
  --max_new_tokens 2000

echo "[4b/5] Evaluate paper_model=${PAPER_MODEL}"
python src/evaluate.py \
  --model_name "${PAPER_MODEL}" \
  --inference_engine "${INFERENCE_ENGINE}" \
  --split val \
  --num_workers "${NUM_WORKERS}" \
  --num_iterations 1 \
  --best_of 1 \
  --temperature 0.0 \
  --max_new_tokens 2000

echo "[5/5] Show summaries"
python print_results.py results

MODEL_TAG="$(basename "${MODEL_UNDER_TEST_RESOLVED}")"
PAPER_TAG="$(basename "${PAPER_MODEL}")"
BASE_DIR="${SUPERCODER_DIR}/results/main/llm_superoptimizer_ds_val"
MODEL_JSON="${BASE_DIR}/${MODEL_TAG}/num_iterations_1/0-shot/best_of_1/problem_results.json"
PAPER_JSON="${BASE_DIR}/${PAPER_TAG}/num_iterations_1/0-shot/best_of_1/problem_results.json"

echo "model_under_test_json=${MODEL_JSON}"
echo "paper_model_json=${PAPER_JSON}"

