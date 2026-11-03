# make eval: score every evaluation in eval/plan.json (see backend/eval/report.py) and
# write eval/results.json, eval/results.md and the SVG charts.
.PHONY: eval test lint
eval:
	python -m backend.cli eval --plan eval/plan.json --out eval

test:
	python -m pytest

lint:
	ruff check .
