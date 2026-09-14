# Manuscript computational record

This directory connects the final manuscript narrative to independently generated computational panels. A Figure 1 display PDF is included for convenience; manually assembled Inkscape masters for the remaining figures are not distributed here.

## What is included

- `FIGURE_PROVENANCE.tsv`: final Figure 1–6 panel mapping.
- `SUPPLEMENTARY_PROVENANCE.tsv`: supplementary-figure mapping and renderer status.
- `SOURCE_PANEL_PROVENANCE.tsv`: SHA-256 and byte-identical original path(s) for every bundled SVG.
- `frozen_tables/`: exact compact tables copied from completed analyses.
- `source_panels/`: independently generated SVG assets used during manuscript assembly.
- repository-level `SHA256SUMS.tsv`: checksums for every bundled release artifact.

## What is intentionally excluded

- manually assembled combined main-figure SVG/PDF/PNG files other than the explicitly supplied Figure 1 display PDF;
- historical layout variants;
- large raw provider datasets;
- manuscript-scale prediction arrays;
- browser sessions, credentials, caches, and third-party source trees.

The manuscript figures were assembled manually in Inkscape. Manual operations were restricted to extracting panels from generated SVGs, positioning, proportional resizing, alignment, whitespace adjustment, and relabelling. The computation, values, axes, normalization, colour scales, and underlying scientific content were not altered during assembly.

## Reproduction levels

1. **Integrity verification** checks every bundled artifact against `SHA256SUMS.tsv`.
2. **Source-panel reproduction** runs a preserved plotting workflow against downloaded frozen outputs where an exact renderer exists.
3. **Scientific reproduction** runs the applicable frozen model workflow from provider data into a new output directory. Prediction locks must be created before withheld target expression is opened.

Rows marked `renderer_not_serialized` retain the authoritative SVG and frozen table but do not claim exact programmatic regeneration of the historical rendering.
