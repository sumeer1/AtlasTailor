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

to validate the software-repository checksums. Consult the companion [AtlasTailor reproducibility repository](https://github.com/sumeer1/AtlasTailor-reproducibility) for `manuscript/FIGURE_PROVENANCE.tsv`, `manuscript/SUPPLEMENTARY_PROVENANCE.tsv`, and `manuscript/SOURCE_PANEL_PROVENANCE.tsv`.

The `assembly_status` field states whether a source figure was programmatic, imported into the external manual Inkscape assembly, or manually authored. Renderer limitations are stated directly in the relevant row; the release does not invent missing historical renderers.

Executed workflow copies are separately matched by SHA-256 in the companion repository's `reproducibility/provenance/EXECUTED_SCRIPT_PROVENANCE.tsv`.
