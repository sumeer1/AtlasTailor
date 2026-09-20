# Preprocessing boundary

AtlasTailor expects prepared AnnData objects rather than prescribing a universal preprocessing procedure across technologies.

The reusable sequence is:

1. verify the authoritative provider files;
2. assign explicit specimen, patient, section and platform identifiers;
3. retain raw counts where available;
4. place finite 2D or 3D coordinates in an AnnData `obsm` field;
5. use unique, stable gene identifiers;
6. identify the source–target gene intersection without examining target outcomes;
7. select the measurement panel from source information only;
8. expose only the declared target measurements during prediction;
9. open hidden target expression only after prediction locking when retrospective validation is intended.

Dataset-specific manuscript preprocessing is preserved in the companion reproducibility repository. The small DLPFC coordinate exporter in this repository supports the public software tutorial and is not presented as the frozen Figure 5 workflow.

See [data conventions](data_formats.md) for the exact AnnData contract and [scientific guardrails](scientific_guardrails.md) for prohibited information flow.
