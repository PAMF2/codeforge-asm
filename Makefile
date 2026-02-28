PYTHON ?= python
VENV ?= .venv
PIP := $(VENV)/Scripts/pip
PY := $(VENV)/Scripts/python

.PHONY: setup train demo eval preflight hf-job tree clean

setup:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -r requirements.txt

train:
	$(PY) -m src.trainer --config configs/grpo_config.yaml

demo:
	$(PY) demo/app.py

eval:
	$(PY) eval/evaluate.py --predictions artifacts/sample_predictions.json

preflight:
	$(PY) scripts/preflight.py

hf-job:
	powershell -ExecutionPolicy Bypass -File scripts/submit_hf_job.ps1 -Flavor a10g-small -Timeout 6h -Detach

tree:
	tree /F

clean:
	if exist artifacts rmdir /S /Q artifacts
	if exist __pycache__ rmdir /S /Q __pycache__
	for /d /r . %%d in (__pycache__) do @if exist "%%d" rmdir /S /Q "%%d"
