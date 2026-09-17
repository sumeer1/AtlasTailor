# AtlasTailor lab toolkit architecture

## Evidence boundary

The toolkit wraps the executed, frozen implementation; it does not redefine the
model. Frozen manuscript result trees are read-only demo and regression inputs.
All new runs write to a caller-selected run directory. The public package imports
the functions below from their authoritative modules, and regression tests compare
the wrappers with frozen tables or arrays.

The product vocabulary is **Design → Adapt → Explore → Forecast**. “Support” means
the frozen source-validation predictability guard. It is not predictive
uncertainty. Target non-anchor expression is excluded from fitting, panel
selection, calibration and prediction and can be opened only by an explicit
post-prediction validation operation.

## Scientific code map

| Product operation | Public entry point | Executed scientific implementation reused | Frozen evidence | Boundary |
|---|---|---|---|---|
| Source normalization | `hyperspatial.io.read_reference` | `run_wemerfish_temporal_holdout.measured_counts`, `normalize_counts`; DLPFC adapter `analysis.hyperspatial_map_2d_dlpfc.dlpfc_common.normalized_source` | 3D MAP manifests; DLPFC protocol lock | AnnData adapter is new tool I/O; formulas are frozen |
| Coordinate canonicalization | `hyperspatial.core.canonicalize` | `run_wemerfish_temporal_holdout.canonicalize`; dimension-agnostic equivalent in `dlpfc_common.canonicalize` | registration manifests | unchanged median center / 99th-percentile radius |
| Anchor eligibility and coverage selection | `hyperspatial.design_panel` | `run_hyperspatial_map_stage1b_20260803.select_non_axial_anchors`; human adapter `dlpfc_common.select_anchors` | frozen 16-gene panel tables | custom budgets are tool-facing, source-only experimental design |
| Panel utility and redundancy | `hyperspatial.panel.analyze_design` | `analysis.panel_design_extension.run_panel_design_extension`: definitions of `pseudo_residual`, `cv_panel`, LOAO and conditional utility; frozen centered-ridge helpers | panel-design extension outputs | post-hoc/source-only; never universal optimality; generic-AnnData adapter reports source-CV rather than claiming target validation |
| Geometry registration | `hyperspatial.registration.register_geometry` | similarity/reflection search in `analysis.hyperspatial_map_2d_dlpfc.predict_direction`; non-rigid `hyperspatial_molecular_hologram.fit_geometry_field` | 3D registration manifests and DLPFC run | target coordinates only; no molecular input |
| Registered atlas prior | `hyperspatial.core.registered_prior` | `run_hyperspatial_map_stage1_20260803.idw` | frozen IDW arrays | k=12 inverse-square IDW |
| Anchor residual features | `hyperspatial.core.residual_features` | `run_hyperspatial_map_stage1_20260803.features` | frozen model/prediction manifests | raw, k=12 and k=32 target-neighbour means |
| Centered low-rank predictors | `hyperspatial.workflows._source_fit` | `run_hyperspatial_map_stage1b_20260803.fit_centered_ridge_rank`, `apply_centered` | frozen source models | rank 16; ridge/gain grids frozen by default |
| Fusion and source-only guard | `hyperspatial.core.source_gate` | `run_hyperspatial_map_stage1b_20260803.candidate_fusion`, `source_gate` | guard tables/manifests | MSE tolerance 1.05 and Pearson margin 0.03; fallback is IDW |
| Adaptation orchestration | `hyperspatial.adapt` | composition of the preceding executed functions | frozen 3D and isolated 2D predictions | new orchestration and artifact schema; no new model |
| Metrics | `hyperspatial.validate` | frozen evaluators and `dlpfc_common.pearson_columns`; localization implementations in `evaluate_phase3.py` | metric tables | truth is accepted only after prediction artifacts are sealed |
| Residual exploration | `AdaptationResult.gene`, `hyperspatial.explore` | read-only computations/formulas in `analysis.residual_landscape` | immutable residual-landscape run | descriptive; residuals are not automatically biological |
| Lineage transport | `hyperspatial.forecast` | `hyperspatial_map_t_core_20260804.lineage_transport` | MAP-T decomposition/predictions | compatible ancestor mappings required |
| Temporal predictors/guard | `hyperspatial.forecast` | centered ridge and `temporal_guard` in `hyperspatial_map_t_core_20260804` | frozen MAP-T models and controls | tracking-map-conditioned forecasting only |

## Runtime layers

1. `hyperspatial/` is the single scientific and artifact core used by Python,
   CLI, REST and web routes.
2. `hyperspatial.cli` supplies the `hyperspatial` executable.
3. `hyperspatial.api` exposes versioned FastAPI routes and a bounded local
   background-job executor. A separately developed web client can consume these
   routes; no browser implementation is duplicated in this release repository.
4. A run directory is append-only after completion and contains configuration,
   hashes, software metadata, selected genes, assignments, scientific arrays,
   logs, manifest and reproduction command.

## Information isolation

Adaptation receives a `TargetSpecimen` object whose molecular matrix is restricted
to the declared measured panel. The orchestrator validates this condition before
fitting. Held-out truth is a separate object accepted only by `validate()` after
the prediction manifest and hashes exist. REST jobs preserve the same separation:
`/adapt` cannot accept a truth layer and `/validate` references a completed run.

## Frozen versus tool-only behavior

Frozen defaults are k=12 atlas transfer, residual neighbourhoods 12 and 32,
rank 16, ridge grid 0.1/1/10, gains 0.25/0.5/0.75/1, fusion weights
0/0.25/0.5/0.75/1, MSE tolerance 1.05, Pearson margin 0.03 and seeds 42–44.
Dimension-agnostic AnnData loading, custom panel budgets, asynchronous local jobs,
HTML reports and interactive visualization are tool-only conveniences and are
labelled as such. They do not alter the frozen scientific defaults.

## Deliberately excluded from v1

No target-trained tuning, formal uncertainty estimates, new neural predictors,
cell–cell communication inference, niche embeddings, perturbation prediction,
autonomous developmental simulation or automatic claims that residual variation
is biological are provided.
