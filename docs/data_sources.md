# Public data sources

AtlasTailor does not redistribute provider-scale spatial expression matrices. The authoritative study records, accessions, platforms and usage notes are maintained in [`data/DATASETS.tsv`](https://github.com/sumeer1/AtlasTailor/blob/main/data/DATASETS.tsv).

## Data-handling policy

1. Obtain data from the official repository or publication-linked record.
2. Preserve the provider archive unchanged.
3. Verify file size and checksum where recorded.
4. Extract external inputs outside the Git repository or under the ignored `data/external/` directory.
5. Write every new analysis into a new output directory.
6. Respect the provider's licence and citation requirements.

The end-to-end tutorials currently use:

- whole-embryo zebrafish weMERFISH from [Dryad 10.5061/dryad.j0zpc86v9](https://doi.org/10.5061/dryad.j0zpc86v9);
- human DLPFC Visium from [spatialLIBD / Figshare 10.6084/m9.figshare.13623902.v1](https://doi.org/10.6084/m9.figshare.13623902.v1).

Additional manuscript datasets and their publication records are listed in the dataset catalogue. Dataset licences remain with their providers and are not extended by AtlasTailor's BSD-3-Clause software licence.
