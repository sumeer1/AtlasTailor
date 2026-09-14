# Figure provenance

Every manuscript result is represented at three levels:

1. frozen numerical or spatial output;
2. independently generated source-panel SVG;
3. panel-level mapping to the final manuscript figure.

The final combined main figures are excluded because they were assembled manually in Inkscape. This does not affect computational reproducibility: every plotted value and imported source panel remains traceable.

Use:

```bash
python scripts/verify_release.py
```

to validate the bundled checksums. Consult `manuscript/FIGURE_PROVENANCE.tsv` for main figures, `manuscript/SUPPLEMENTARY_PROVENANCE.tsv` for supplementary figures, and `manuscript/SOURCE_PANEL_PROVENANCE.tsv` for the byte-identical source path and hash of every bundled SVG.

The `assembly_status` field states whether a source figure was programmatic, imported into the external manual Inkscape assembly, or manually authored. Renderer limitations are stated directly in the relevant row; the release does not invent missing historical renderers.

Executed workflow copies are separately matched by SHA-256 in `reproducibility/provenance/EXECUTED_SCRIPT_PROVENANCE.tsv`.
