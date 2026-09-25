PY := .venv/bin/python

.PHONY: synth train threshold eval evals serve bench docker-build docker-run smoke corpus pretrain scratch

synth:
	$(PY) synth.py --per-template 10

train:  ## trains a NEW model into its own directory; the shipped artifact is tracked and is never a default target
	$(PY) train.py --train data/train.jsonl --out models/eot-distilbert-retrain

corpus:  ## one-time: the data the scratch pretrain reads (installs the datasets package, not a serving dep)
	uv pip install -q --python $(PY) datasets
	$(PY) synth_scale.py --per-template 60
	$(PY) synth_es.py --per-template 30
	$(PY) fetch_pretrain_corpus.py

pretrain:  ## ~15 min on MPS: masked-language-model base for the from-scratch lane, run corpus first
	$(PY) pretrain_scratch.py

scratch:  ## fine-tune the from-scratch base on the task, run pretrain first
	$(PY) train_scratch.py --init-from models/eot-scratch-base --lr 1e-4 --out models/eot-scratch-pre $(if $(wildcard data/ood_train.jsonl),--real data/ood_train.jsonl,)

tier1:  ## rebuild the twelve guardrail rows from the committed data
	$(PY) make_tier1_probes.py

threshold:
	$(PY) pick_threshold.py --model-dir models/eot-distilbert-onnx-int8 --labels data/dev_set.json --constraints data/tier1_probes.jsonl --cost-ratio 5

eval:
	$(PY) evaluate.py --model models/eot-distilbert-onnx-int8 --data data/gold_set.json --report eval_report.json

evals: eval

serve:
	$(PY) -m uvicorn serve:app --host 127.0.0.1 --port 8000

bench:
	$(PY) bench.py --url http://127.0.0.1:8000/predict --n 1500 --concurrency 1,8,32 --report bench_report.json

docker-build:
	docker build -t eot-detector .

docker-run:
	docker run --rm -p 8000:8000 eot-detector

smoke:
	curl -s http://127.0.0.1:8000/healthz && curl -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"context":"","text":"Okay, got it."}'

figures:  ## regenerate the README figures from their constants
	$(PY) draw_figures.py --write

figures-check:  ## fail if a committed figure drifted from its constants or the 75% type floor
	$(PY) draw_figures.py --check