install:
	@echo "--- 🚀 Installing project dependencies ---"
	uv sync --all-groups

lint:
	@echo "--- 🧹 Running linters ---"
	uv run ruff format . 	# running ruff formatting
	uv run ruff check . --fix --exit-non-zero-on-fix  	# running ruff linting # --exit-non-zero-on-fix is used for the pre-commit hook to work
	uv run ty check . --error=all