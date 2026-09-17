#!/usr/bin/env Rscript

# Export only public geometry and identifiers for the two-section DLPFC
# AtlasTailor tutorial. Expression is read from the official 10x matrices in
# Python; this script never exports or transforms molecular measurements.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) {
    stop("usage: export_spatiallibd_coordinates.R <output.tsv>")
}

if (!requireNamespace("spatialLIBD", quietly = TRUE)) {
    stop(
        "Install spatialLIBD first: ",
        "BiocManager::install('spatialLIBD')"
    )
}

sce <- spatialLIBD::fetch_data(type = "sce")
fields <- as.data.frame(SummarizedExperiment::colData(sce))
keep <- fields$sample_name %in% c("151507", "151508")
fields <- fields[keep, , drop = FALSE]

required <- c("sample_name", "barcode", "imagecol", "imagerow")
if (!all(required %in% colnames(fields))) {
    stop("The spatialLIBD object does not contain the expected coordinate fields")
}

output <- data.frame(
    section = as.character(fields$sample_name),
    barcode = as.character(fields$barcode),
    x = as.numeric(fields$imagecol),
    y = as.numeric(fields$imagerow),
    stringsAsFactors = FALSE
)

write.table(output, args[[1]], sep = "\t", row.names = FALSE, quote = FALSE)
cat("Wrote ", nrow(output), " public DLPFC spot-coordinate records.\n", sep = "")
