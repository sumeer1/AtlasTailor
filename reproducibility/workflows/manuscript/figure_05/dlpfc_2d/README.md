# HyperSpatial-MAP 2D DLPFC extension

This directory contains new, isolated code for the post-freeze external 2D audit. It does not import or modify an existing 3D result directory.

`audit_spatiallibd_phase1.R` reads the authoritative spatialLIBD processed DLPFC object and writes metadata-only Phase 1 manifests. It does not fit registration, select anchors, read target sections directionally, or make molecular predictions.
