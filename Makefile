.PHONY: install test notebooks verify docs build clean

install:
	python -m pip install -e '.[test,docs,notebooks]'

test:
	pytest

notebooks:
	python scripts/execute_notebooks.py

verify:
	python scripts/verify_release.py

docs:
	mkdocs build --strict

build:
	python -m build

clean:
	python -c "from pathlib import Path; import shutil; [shutil.rmtree(p, ignore_errors=True) for p in map(Path, ('build','dist','site','.pytest_cache','.ruff_cache'))]"
