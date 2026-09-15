# Repository scope

HyperSpatial-MAP uses two linked repositories with distinct responsibilities.

## This repository: reusable software

Included:

- installable Python package and CLI;
- reusable 2D/3D input and preprocessing contracts;
- deterministic synthetic examples;
- unit, API, leakage-boundary, and notebook tests;
- user documentation;
- a compact frozen Figure 2 evidence walkthrough;
- the final Figure 1 overview for orientation.

Not included:

- manuscript-scale provider data or prediction arrays;
- the complete collection of frozen evidence tables;
- historical executed workflow snapshots and prediction locks;
- generated manuscript source panels;
- final manual Inkscape assemblies.

## Companion repository: manuscript reproducibility

[HyperSpatial-MAP-reproducibility](https://github.com/sumeer1/HyperSpatial-MAP-reproducibility) preserves executed workflow snapshots, exact frozen configurations and locks, manuscript evidence tables, generated source-panel SVGs, dataset provenance, and integrity manifests.

Keeping these layers separate makes the software repository approachable while retaining a complete, auditable computational record. The repositories are version-linked by Git commit identifiers and SHA-256 manifests.
