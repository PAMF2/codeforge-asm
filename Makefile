PYTHON ?= python
VENV ?= .venv
PIP := $(VENV)/Scripts/pip
PY := $(VENV)/Scripts/python

.PHONY: setup train demo eval tree clean

setup:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -r requirements.txt

train:
	$(PY) -m src.trainer --config configs/grpo_config.yaml

demo:
	$(PY) demo/app.py

eval:
	$(PY) eval/evaluate.py --predictions artifacts/sample_predictions.json

tree:
	tree /F

clean:
	if exist artifacts rmdir /S /Q artifacts
	if exist __pycache__ rmdir /S /Q __pycache__
	for /d /r . %%d in (__pycache__) do @if exist "%%d" rmdir /S /Q "%%d"
