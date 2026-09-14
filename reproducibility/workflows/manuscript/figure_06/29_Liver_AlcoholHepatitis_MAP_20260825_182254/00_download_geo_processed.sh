#!/usr/bin/env bash
set -euo pipefail
out="PI_revision/29_Liver_AlcoholHepatitis_MAP_20260825_182254/data/GSE278662"
mkdir -p "$out"
base="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE278nnn/GSE278662/suppl"
for sample in Sah73 Sah80; do
  for suffix in barcodes.tsv.gz features.tsv.gz matrix.mtx.gz tissue_positions.csv.gz; do
    file="GSE278662_${sample}_${suffix}"
    wget -c -O "$out/$file" "$base/$file"
  done
done
