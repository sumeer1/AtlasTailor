"""Biological audit of the HyperSpatial-MAP target-specific residual landscape.

The frozen cross-stage run established that HyperSpatial-MAP recovers a
target-specific residual (truth minus registered atlas transfer) that registered
IDW does not. It did not establish *what that residual is*. This module asks
whether the recovered residual is biologically structured tissue variation or
technical/geometric variation, using annotations the frozen protocol never
opened.

Everything under ``results_v10/hyperspatial_map_cross_stage_frozen_20260803_151726``
and the two 75%-epiboly prediction directories is read-only input. All output
goes to a fresh ``results_v10/hyperspatial_map_residual_landscape_<timestamp>``
directory.
"""
