# Explore target-specific residuals

A target-specific residual is the deviation from the registered atlas expectation. The predicted residual is zero for genes assigned to registered-IDW fallback. Source support is a predictability decision, not an uncertainty score.

`hyperspatial.explore` ranks genes using support and spatial structure of the predicted correction. Held-out measured residuals, reconstruction effects and annotations can be added only through explicit post-prediction validation. Spatial structure beyond registered anatomy should not automatically be interpreted as biological variation; registration quality, molecular depth and other technical effects remain relevant.

