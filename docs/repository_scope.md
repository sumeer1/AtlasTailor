# Repository scope

This repository is focused on reusable AtlasTailor software and concise public examples.

## Included

- installable Python package and CLI;
- reusable 2D/3D input and preprocessing contracts;
- compact real-data result summaries;
- deterministic fixtures isolated under `tests/` for automated validation;
- unit, API, leakage-boundary, and notebook tests;
- user documentation;
- compact versioned evidence for the Figure 2 analyses;
- the final Figure 1 overview for orientation.

## Not included

- manuscript-scale provider data or prediction arrays;
- the complete collection of frozen evidence tables;
- historical executed workflow snapshots and prediction locks;
- generated manuscript source panels;
- final manuscript composite artwork.

Keeping provider-scale datasets and manuscript artwork outside the software package makes installation and review concise. Public dataset records are listed under `data/`, and versioned summary tables used by the reported-results notebooks are bundled under `examples/`.
