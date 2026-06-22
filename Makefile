.PHONY: lint
lint: ruff-check mypy-core lint-imports

.PHONY: ruff-check
ruff-check:
	ruff check .

.PHONY: ruff-format
ruff-format:
	ruff format .

.PHONY: mypy-core
mypy-core:
	mypy packages/core

.PHONY: lint-imports
lint-imports:
	lint-imports

.PHONY: install-dev
install-dev:
	pip install -e ".[dev]"

.PHONY: precommit-install
precommit-install:
	pre-commit install

.PHONY: precommit-run
precommit-run:
	pre-commit run --all-files

.PHONY: test
test:
	pytest

.PHONY: test-integration
test-integration:
	DATABASE_URL=postgresql://rc:rc@localhost:5432/research_copilot \
	pytest tests/test_integration.py -v --timeout=120

.PHONY: eval
eval:
	python -m eval.run

.PHONY: docker-up
docker-up:
	docker compose up --build -d

.PHONY: docker-down
docker-down:
	docker compose down

.PHONY: docker-smoke
docker-smoke:
	tests/smoke_docker.sh

.PHONY: release
release: eval
	@echo "=== Gate PASS — creating v0.1.0 tag ==="
	git tag -a v0.1.0 -m "v0.1.0 — Research Copilot MVP with PDF ingest, QA pipeline, eval gate, and React frontend"
	@echo "Tagged v0.1.0. Push with: git push origin v0.1.0"
