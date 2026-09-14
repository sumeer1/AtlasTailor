#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "$0")/../../.." && pwd)"
out="$project_root/PI_revision/28_Liver_PSC_MAP_20260825_175802/data/GSE245620"
mkdir -p "$out"

for record in \
  "GSM7845914 PSC011_A1_VISIUM" \
  "GSM7845915 PSC011_B1_VISIUM" \
  "GSM7845916 PSC011_C1_VISIUM" \
  "GSM7845917 PSC011_D1_VISIUM"
do
  read -r gsm stem <<< "$record"
  prefix="https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM7845nnn/$gsm/suppl/${gsm}_${stem}"
  for suffix in barcodes.tsv.gz features.tsv.gz matrix.mtx.gz tissue_positions_list.csv.gz
  do
    wget -c -O "$out/${gsm}_${stem}_${suffix}" "$prefix"_"$suffix"
  done
done

curl -fsSL "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE245nnn/GSE245620/suppl/filelist.txt" \
  -o "$out/GSE245620_filelist.txt"

