# Release validation record

Validation was performed on 15 September 2026 using Python 3.11.11. No manuscript prediction or metric was recomputed during these checks.

| Check | Result |
|---|---|
| Non-regression package tests | PASS — 13 passed, 3 manuscript-regression tests deselected |
| Tutorial execution | PASS — 7 of 7 notebooks, including the compact Figure 2 walkthrough |
| Ruff static checks | PASS |
| MkDocs strict build | PASS |
| Source distribution and wheel build | PASS |
| Clean wheel import outside repository | PASS — `hyperspatial.__version__ == 0.1.0` |
| Release SHA-256 verification | Regenerated for the separated software repository |
| Manuscript-scale workflows/tables/source panels bundled | NO — moved to the companion reproducibility repository |
| Manual combined-figure masters bundled | NO; the author-supplied final Figure 1 PDF and preview are included for orientation |

The three deselected regression tests require external immutable manuscript data that are not redistributed through GitHub. Their accessions and artifact policy are documented under `data/`; the complete manuscript evidence record is in `HyperSpatial-MAP-reproducibility`.
