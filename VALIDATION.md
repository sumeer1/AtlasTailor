# Release validation record

Validation was performed on 14 September 2026 using Python 3.11.11. No manuscript prediction or metric was recomputed during these checks.

| Check | Result |
|---|---|
| Non-regression package tests | PASS — 13 passed, 3 manuscript-regression tests deselected |
| Tutorial execution | PASS — 7 of 7 notebooks |
| Ruff static checks | PASS |
| MkDocs strict build | PASS |
| Source distribution and wheel build | PASS |
| Clean wheel import outside repository | PASS — `hyperspatial.__version__ == 0.1.0` |
| Release SHA-256 verification | PASS — 402 files |
| Executed-script identity | PASS — 114 of 114 byte-identical matches |
| Generated source-SVG identity | PASS — 27 of 27 byte-identical matches |
| Manual final manuscript masters bundled | NO, by design |

The three deselected regression tests require external immutable manuscript data that are not redistributed through GitHub. Their accessions and artifact policy are documented under `data/`.
