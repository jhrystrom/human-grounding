install:
    @echo "--- 🚀 Installing project dependencies ---"
    uv sync --all-groups

lint:
    @echo "--- 🧹 Running linters ---"
    uv run ruff format .  # running ruff formatting
    uv run ruff check . --fix --exit-non-zero-on-fix  # running ruff linting
    # --exit-non-zero-on-fix is used for the pre-commit hook to work
    uv run ty check . --error=all

samples := "30"
rq1:
    @echo "--- 🔬 Reproducing RQ1 results ---"
    uv run scripts/alpha_distance_plot.py --experiment gov-ai --samples={{samples}}
    uv run scripts/alpha_distance_plot.py --experiment policy --samples={{samples}}

rq2:
    @echo "Reproducing RQ2:"
    uv run scripts/neural_alignment_plots.py --experiments policy gov-ai --cache
    uv run scripts/fairness_tables.py
    uv run scripts/lexical_baselines_table.py

rq3:
    @echo "Reproducing RQ3:"
    uv run scripts/clustering.py
    uv run scripts/k_sensitivity_table.py

main-values:
    @echo "Reproducing full values"
    uv run scripts/report_canonical_values.py

reproduce-full:
    @echo "--- 🔬 Reproducing all results ---"
    just rq1
    just rq2
    just rq3
    just main-values
    @echo "--- Done! ---"