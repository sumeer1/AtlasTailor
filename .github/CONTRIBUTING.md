# Contributing

Contributions should preserve the information-isolation contract and frozen scientific defaults. Open an issue before changing registration, panel eligibility, predictor fitting, calibration, the guard or MAP-T semantics.

```bash
python -m pip install -e '.[test,docs,notebooks]'
ruff check src tests
pytest
mkdocs build --strict
python -m build
```

New scientific behavior must be clearly separated from manuscript-frozen behavior, tested against leakage, documented and accompanied by a regression fixture. Do not commit manuscript datasets, credentials, local paths or generated run directories.

