# FAQ

## Is HyperSpatial-MAP generic imputation?

No. It begins with a geometry-registered atlas expectation and predicts a target-specific residual only when source validation supports correction.

## Are support scores uncertainty estimates?

No. They are source pseudo-holdout performance quantities used by a deterministic guard.

## Can I tune on my target?

Not in the v1 workflow. Target non-anchor expression is evaluation-only.

## Can I use two-dimensional coordinates?

Yes. Coordinate canonicalization, distance calculations, IDW, residual neighborhoods and the deformation field are dimension-aware. The source pseudo-holdout scheme uses four quadrants in 2D and eight octants in 3D.

## Why did a gene retain registered IDW?

The frozen source guard did not support a correction for that gene. This fallback is part of the method.

