.PHONY: install run debug lint lint-strict clean

install:
	uv sync

run:
	uv run python -m call_me_maybe $(ARGS)

debug:
	uv run python -m pdb -m call_me_maybe $(ARGS)

lint:
	uv run flake8 call_me_maybe
	uv run mypy call_me_maybe --warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 call_me_maybe
	uv run mypy call_me_maybe --strict

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	rm -rf .mypy_cache
