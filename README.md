# CodeForge ASM

Treino de modelo Ministral para geração de assembly NASM x86-64 Linux com RL e verificação objetiva (`nasm` + `ld` + execução).

## Prioridade do Hackathon
1. Colocar de pé o loop **Best-of-N + Reward Pipeline + W&B**.
2. Integrar **GRPO real** com LoRA/4-bit.
3. Ligar **MCTS line-by-line** como upgrade de Tier 3/4.

## Estrutura do Repositório
```text
codeforge-asm/
  configs/
    grpo_config.yaml
  demo/
    app.py
  docs/
    ARCHITECTURE.md
  eval/
    evaluate.py
  notebooks/
    train.ipynb
  prompts/
    dataset.json
  src/
    __init__.py
    best_of_n.py
    mcts.py
    prompt_engine.py
    reward.py
    trainer.py
    utils.py
  .editorconfig
  .env.example
  .gitignore
  Makefile
  README.md
  requirements.txt
```

## Requisitos
- Python 3.10+
- Linux para validação real de assembly (NASM + GNU ld)
- GPU 16 GB VRAM (recomendado para 3B com quantização)

### Dependências de sistema (Linux)
```bash
sudo apt update
sudo apt install -y nasm binutils
```

## Setup
```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## Variáveis de ambiente
Copie `.env.example` para `.env` e preencha:
```bash
HF_TOKEN=...
HUGGINGFACE_HUB_TOKEN=...
WANDB_API_KEY=...
MISTRAL_API_KEY=...
WANDB_PROJECT=codeforge-asm
```

Notas:
- `HF_TOKEN` e `HUGGINGFACE_HUB_TOKEN` são aliases no projeto (pode usar um deles).
- `MISTRAL_API_KEY` é opcional para o treino principal em `src.trainer`; use quando rodar scripts que chamam a API da Mistral.

## Execução
Treino MVP:
```bash
python -m src.trainer --config configs/grpo_config.yaml
```

Backends de treino:
- `training.grpo_backend: manual` -> update group-relative custom (padrão).
- `training.grpo_backend: trl` -> usa `trl.GRPOTrainer` nativo.
- Push padrão de modelo: `mistral-hackaton-2026/codeforge`.

Preflight local:
```bash
python scripts/preflight.py
```

HF Jobs (PowerShell):
```powershell
.\scripts\submit_hf_job.ps1 -Flavor a10g-small -Timeout 6h -Detach
```

Demo:
```bash
python demo/app.py
```

Avaliação (predições externas):
```bash
python eval/evaluate.py --predictions artifacts/sample_predictions.json
```

## Estado Atual
- Reward pipeline em 4 estágios implementado.
- Best-of-N implementado.
- Loop de treino com logging e artefatos implementado.
- Treino group-relative (GRPO-inspired) implementado em `src/trainer.py`.
- Backend `trl` integrado com fallback por configuração.
- Suporte a carregamento real de modelo HF + LoRA + checkpoints por iteração.
- Push automático de checkpoints para HF Hub (`training.push_to_hub`).
  - Se não houver permissão na org, usa `hub_fallback_repo_id`.
- MCTS está separado como módulo de upgrade.

## Próxima Integração Crítica
Trocar objetivo atual por `TRL GRPOTrainer` nativo quando ambiente estiver estável:
- manter o mesmo buffer `(prompt, asm, reward)`,
- conectar penalidade KL explícita,
- comparar curva com baseline atual.
