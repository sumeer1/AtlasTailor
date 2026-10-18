# Figure provenance

Every manuscript result is represented at three levels:

1. frozen numerical or spatial output;
2. independently generated source-panel SVG;
3. panel-level mapping to the final manuscript figure.

Final manuscript composite artwork is maintained separately. The software repository provides the Figure 1 overview and versioned result tables used by the reported-results notebooks.

Use:

```bash
python scripts/verify_release.py
```

to validate the software-repository checksums.

The package generates analysis outputs and source-level visualizations; final manuscript composition is outside the software workflow.
