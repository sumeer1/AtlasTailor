# Release validation record

Validation was repeated on 17 September 2026 using Python 3.11.11 after the public-example and repository-layout refinement. No manuscript prediction or metric was recomputed during these checks.

| Check | Result |
|---|---|
| Non-regression package tests | PASS — 13 passed, 3 manuscript-regression tests deselected |
| Public real-data notebook execution | PASS — 3 of 3 notebooks |
| Ruff static checks | PASS |
| MkDocs strict build | PASS |
| Source distribution and wheel build | PASS |
| Clean wheel import outside repository | PASS — `hyperspatial.__version__ == 0.1.0` |
| Release SHA-256 verification | PASS — manifest stored under `provenance/` |
| Manuscript-scale workflows/tables/source panels bundled | NO — moved to the companion reproducibility repository |
| Real manuscript evidence used by public notebooks | YES — compact frozen tables for Figures 2, 5 and 6 |
| Simulated fixtures presented as scientific examples | NO — internal fixtures are isolated under `tests/fixtures/` |
| Manual combined-figure masters bundled | NO; the author-supplied final Figure 1 PDF and preview remain available under `docs/assets/` |

The three deselected regression tests require external immutable manuscript data that are not redistributed through GitHub. Their accessions and artifact policy are documented under `data/`; the complete manuscript evidence record is in `AtlasTailor-reproducibility`.
