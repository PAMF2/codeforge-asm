# HF Jobs Runbook

## 1) Auth
```bash
hf auth whoami
```
Expected: user + org `mistral-hackaton-2026`.

## 2) Preflight
```bash
python scripts/preflight.py
```

## 3) Submit training job
```bash
hf jobs uv run train.py \
  --flavor a10g-small \
  --namespace mistral-hackaton-2026 \
  --repo mistral-hackaton-2026/<existing-job-dataset-repo> \
  --timeout 6h \
  --secrets HF_TOKEN \
  --secrets WANDB_API_KEY \
  --secrets MISTRAL_API_KEY \
  --env PYTHONUTF8=1 \
  -- --config configs/grpo_config.yaml --ensure-system-deps
```

## 4) Monitor
```bash
hf jobs ps
hf jobs logs <JOB_ID>
hf jobs inspect <JOB_ID>
```

## 5) Model Push
`src/trainer.py` creates `training.hub_repo_id` if missing and uploads each checkpoint to:
`checkpoints/iter_<N>`

## Notes
- Validated base model id: `mistralai/Ministral-8B-Instruct-2410`.
- Current account check returned `403` for creating repos under `mistral-hackaton-2026` namespace.
- Current account check returned `402` for running Jobs on personal namespace due insufficient prepaid credits.
- Trainer is configured with fallback push target: `PAMF2/Hackathon-model`.
- Local Windows machine may not have `nasm/ld`; HF Linux job can auto-install with `--ensure-system-deps`.
