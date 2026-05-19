install:
	@echo "--- 🚀 Installing project dependencies ---"
	uv sync --all-groups

lint:
	@echo "--- 🧹 Running linters ---"
	uv run ruff format . 	# running ruff formatting
	uv run ruff check . --fix --exit-non-zero-on-fix  	# running ruff linting # --exit-non-zero-on-fix is used for the pre-commit hook to work
	uv run ty check . --error=all

reproduce-full:
    @echo "--- 🔬 Reproducing all results ---"
	@echo "Reproducing RQ1:"
	uv run scripts/alpha_distance_plot.py --experiments policy gov-ai
	@echo "Reproducing RQ2:"
	uv run scripts/neural_distance_plot.py --experiments policy gov-ai --cache
	uv run scripts/fairness_tables.py
	uv run scripts/lexical_baselines_table.py
	@echo "Reproducing RQ3:"
	uv run scripts/clustering.py
	uv run scripts/k_sensitivity_table.py
	@echo "--- Done! ---"
