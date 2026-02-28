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
HUGGINGFACE_HUB_TOKEN=...
WANDB_API_KEY=...
MISTRAL_API_KEY=...
WANDB_PROJECT=codeforge-asm
```

## Execução
Treino MVP:
```bash
python -m src.trainer --config configs/grpo_config.yaml
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
- GRPO está com stub (`run_grpo_update`) pronto para integração TRL.
- MCTS está separado como módulo de upgrade.

## Próxima Integração Crítica
Implementar GRPO real em `src/trainer.py`:
- carregar Ministral (3B) com 4-bit,
- aplicar LoRA,
- usar TRL para atualização por reward relativo,
- salvar adapters/checkpoints por iteração.
