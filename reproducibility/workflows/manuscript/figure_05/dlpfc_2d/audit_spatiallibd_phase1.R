#!/usr/bin/env Rscript

# Phase-1-only metadata audit for the authoritative spatialLIBD DLPFC object.
# This deliberately does not inspect expression values beyond assay metadata.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3L) {
    stop("usage: audit_spatiallibd_phase1.R <spatialLIBD.Rdata> <scalefactor_dir> <output_dir>")
}

input_path <- normalizePath(args[[1]], mustWork = TRUE)
scale_dir <- normalizePath(args[[2]], mustWork = TRUE)
output_dir <- normalizePath(args[[3]], mustWork = TRUE)

loaded <- new.env(parent = emptyenv())
objects <- load(input_path, envir = loaded)
if (!identical(objects, "sce")) stop("Expected the authoritative object named 'sce'.")

# Access serialized S4 slots without mutating or converting the source object.
top <- attributes(loaded$sce)
col_data <- attributes(top$colData)$listData
row_data <- attributes(attributes(top$rowRanges)$elementMetadata)$listData
assays <- attributes(attributes(top$assays)$data)$listData

required_cols <- c(
    "sample_name", "subject", "position", "replicate", "subject_position",
    "row", "col", "imagerow", "imagecol", "layer_guess"
)
stopifnot(all(required_cols %in% names(col_data)))
stopifnot(all(c("counts", "logcounts") %in% names(assays)))

assay_dim <- attributes(assays$counts)$Dim
if (!identical(assay_dim, attributes(assays$logcounts)$Dim)) {
    stop("counts and logcounts dimensions differ")
}

read_scale <- function(sample_id) {
    path <- file.path(scale_dir, paste0(sample_id, "_scalefactors_json.json"))
    text <- paste(readLines(path, warn = FALSE), collapse = "")
    extract <- function(key) {
        pattern <- paste0('.*"', key, '"[[:space:]]*:[[:space:]]*([0-9.eE+-]+).*')
        as.numeric(sub(pattern, "\\1", text))
    }
    c(
        tissue_lowres_scalef = extract("tissue_lowres_scalef"),
        tissue_hires_scalef = extract("tissue_hires_scalef"),
        spot_diameter_fullres = extract("spot_diameter_fullres")
    )
}

sample_ids <- sort(unique(col_data$sample_name))
records <- lapply(sample_ids, function(sample_id) {
    take <- which(col_data$sample_name == sample_id)
    scale <- read_scale(sample_id)
    layers <- sort(unique(as.character(col_data$layer_guess[take][!is.na(col_data$layer_guess[take])])))
    data.frame(
        donor = unique(col_data$subject[take]),
        section = as.character(sample_id),
        adjacent_pair = unique(col_data$subject_position[take]),
        position_um = unique(col_data$position[take]),
        replicate_within_pair = unique(col_data$replicate[take]),
        n_spots = length(take),
        n_genes = assay_dim[[1]],
        raw_counts_available = TRUE,
        normalized_log_expression_available = TRUE,
        stored_xy_fields = "imagecol;imagerow",
        selected_xy_fields = "pxl_col_in_fullres;pxl_row_in_fullres",
        selected_xy_units = "full-resolution image pixels",
        tissue_lowres_scalef = unname(scale[["tissue_lowres_scalef"]]),
        fullres_x_min = min(col_data$imagecol[take]) / scale[["tissue_lowres_scalef"]],
        fullres_x_max = max(col_data$imagecol[take]) / scale[["tissue_lowres_scalef"]],
        fullres_y_min = min(col_data$imagerow[take]) / scale[["tissue_lowres_scalef"]],
        fullres_y_max = max(col_data$imagerow[take]) / scale[["tissue_lowres_scalef"]],
        cortical_layer_annotation = "layer_guess",
        n_spots_with_layer = sum(!is.na(col_data$layer_guess[take])),
        observed_layer_labels = paste(layers, collapse = ";"),
        stringsAsFactors = FALSE
    )
})
sections <- do.call(rbind, records)
write.table(
    sections,
    file.path(output_dir, "01_data_manifest", "sections.tsv"),
    sep = "\t", row.names = FALSE, quote = FALSE, na = "NA"
)

pairs <- unique(sections[c("donor", "adjacent_pair", "position_um")])
pairs <- pairs[order(pairs$donor, as.numeric(pairs$position_um)), ]
pairs$section_1 <- vapply(pairs$adjacent_pair, function(pair) {
    as.character(sections$section[sections$adjacent_pair == pair & sections$replicate_within_pair == "1"])
}, character(1))
pairs$section_2 <- vapply(pairs$adjacent_pair, function(pair) {
    as.character(sections$section[sections$adjacent_pair == pair & sections$replicate_within_pair == "2"])
}, character(1))
pairs$n_directions_phase2 <- 2L
write.table(
    pairs,
    file.path(output_dir, "01_data_manifest", "adjacent_pairs.tsv"),
    sep = "\t", row.names = FALSE, quote = FALSE
)

scales <- do.call(rbind, lapply(sample_ids, function(sample_id) {
    scale <- read_scale(sample_id)
    data.frame(
        section = sample_id,
        tissue_lowres_scalef = scale[["tissue_lowres_scalef"]],
        tissue_hires_scalef = scale[["tissue_hires_scalef"]],
        spot_diameter_fullres_pixels = scale[["spot_diameter_fullres"]],
        stored_image_coordinates = "low-resolution image pixels",
        selected_registration_coordinates = "imagecol/lowres_scale;imagerow/lowres_scale",
        selected_units = "full-resolution image pixels",
        stringsAsFactors = FALSE
    )
}))
write.table(
    scales,
    file.path(output_dir, "01_data_manifest", "coordinate_scaling.tsv"),
    sep = "\t", row.names = FALSE, quote = FALSE
)

assay_manifest <- data.frame(
    assay = c("counts", "logcounts"),
    class = c(attributes(assays$counts)$class, attributes(assays$logcounts)$class),
    n_genes = assay_dim[[1]],
    n_spots = assay_dim[[2]],
    nonzero_entries = c(length(attributes(assays$counts)$x), length(attributes(assays$logcounts)$x)),
    representation = c("raw integer UMI counts", "scran-normalized log-expression"),
    stringsAsFactors = FALSE
)
write.table(
    assay_manifest,
    file.path(output_dir, "01_data_manifest", "expression_assays.tsv"),
    sep = "\t", row.names = FALSE, quote = FALSE
)

feature_manifest <- data.frame(
    field = c("feature_identity", "display_symbol", "gene_biotype"),
    spatialLIBD_rowData_field = c("gene_id", "gene_name", "gene_biotype"),
    n_values = c(length(row_data$gene_id), length(row_data$gene_name), length(row_data$gene_biotype)),
    example = c(row_data$gene_id[[1]], row_data$gene_name[[1]], as.character(row_data$gene_biotype[[1]])),
    stringsAsFactors = FALSE
)
write.table(
    feature_manifest,
    file.path(output_dir, "01_data_manifest", "feature_identity.tsv"),
    sep = "\t", row.names = FALSE, quote = FALSE
)

cat("Phase 1 metadata audit complete: ", nrow(sections), " sections; ", nrow(pairs), " adjacent pairs.\n", sep = "")
