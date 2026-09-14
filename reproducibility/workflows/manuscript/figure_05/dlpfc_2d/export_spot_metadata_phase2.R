#!/usr/bin/env Rscript

# Export geometry/identifiers/annotations only from the authoritative spatialLIBD
# object. Expression assays are not dereferenced by this script.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3L) {
    stop("usage: export_spot_metadata_phase2.R <spatialLIBD.Rdata> <scale_dir> <output.tsv>")
}

loaded <- new.env(parent = emptyenv())
load(normalizePath(args[[1]], mustWork = TRUE), envir = loaded)
col_data <- attributes(attributes(loaded$sce)$colData)$listData

read_lowres_scale <- function(sample_id) {
    path <- file.path(args[[2]], paste0(sample_id, "_scalefactors_json.json"))
    text <- paste(readLines(path, warn = FALSE), collapse = "")
    pattern <- '.*"tissue_lowres_scalef"[[:space:]]*:[[:space:]]*([0-9.eE+-]+).*'
    as.numeric(sub(pattern, "\\1", text))
}

sample_ids <- as.character(col_data$sample_name)
scales <- vapply(sample_ids, read_lowres_scale, numeric(1))
layer <- as.character(col_data$layer_guess)

metadata <- data.frame(
    section = sample_ids,
    barcode = as.character(col_data$barcode),
    donor = as.character(col_data$subject),
    adjacent_pair = as.character(col_data$subject_position),
    position_um = as.character(col_data$position),
    replicate_within_pair = as.character(col_data$replicate),
    x_fullres_px = as.numeric(col_data$imagecol) / scales,
    y_fullres_px = as.numeric(col_data$imagerow) / scales,
    cortical_layer = ifelse(is.na(layer), "NA", layer),
    stringsAsFactors = FALSE
)

write.table(metadata, args[[3]], sep = "\t", row.names = FALSE, quote = FALSE, na = "NA")
cat("Wrote ", nrow(metadata), " geometry/metadata rows.\n", sep = "")
