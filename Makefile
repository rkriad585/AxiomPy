.PHONY: build clean install lint test

build:
	uv build

install:
	uv sync

clean:
	rm -rf dist/ build/ *.egg-info/ __pycache__/ .venv/
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true

lint:
	@echo "No linter configured yet"
	@true

test:
	@echo "No test framework configured yet"
	@true

.PHONY: run-examples
run-examples:
	for f in examples/*.py; do uv run python "$$f"; echo "---"; done
