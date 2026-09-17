# External-artifact policy

Git is used for code, compact result tables, configurations, manifests, documentation, and small CI fixtures. Raw data, protected data, large provider archives, model checkpoints, and manuscript-scale prediction arrays belong in their authoritative repository or a versioned archival bundle such as Zenodo. Generated manuscript source panels are maintained in the companion reproducibility repository.

Each external artifact should have:

- an accession or stable URL;
- provider and citation;
- license or access terms;
- expected filename and size;
- MD5 or SHA256 when available;
- the workflow that consumes it;
- a statement of whether it is raw, processed, or derived.

Do not use Git LFS as a substitute for an appropriate public data archive for human molecular data.
