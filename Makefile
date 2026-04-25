PY ?= python
PORT ?= 7870

.PHONY: demo demo-fast install install-train serve test smoke train train-league evaluate clean docker-build docker-run

demo:
	$(PY) run_demo.py

demo-fast:
	$(PY) run_demo.py --no-train --no-upload --episodes 6 --max-steps 30 --grid-episodes 1


install:
	$(PY) -m pip install -e .

install-train:
	$(PY) -m pip install -e .[train]

serve:
	$(PY) -m server.app --port $(PORT)

test smoke:
	$(PY) -m pytest tests/ -q

train:
	$(PY) train/colab_trl_selfplay.py

train-league:
	TRAIN_LEAGUE_ROUNDS=2 $(PY) train/colab_trl_selfplay.py

train-blue:
	$(PY) train/train_blue_vs_pool.py --rounds 2 --episodes 8

train-red:
	$(PY) train/train_red_vs_pool.py --rounds 2 --episodes 6

evaluate:
	$(PY) train/evaluate_league.py --episodes 3

clean:
	rm -rf artifacts/ outputs/ .runtime/ .pytest_cache/ __pycache__/ */__pycache__/ */*/__pycache__/

docker-build:
	docker build -t cyber-selfplay:latest .

docker-run:
	docker run --rm -p $(PORT):$(PORT) -e PORT=$(PORT) cyber-selfplay:latest
