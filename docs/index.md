# AtlasTailor

AtlasTailor adapts a deeply profiled spatial reference to an individual target using target geometry and sparse molecular measurements. The reusable package follows four workflows:

```text
DESIGN:   reference → source-only measurement panel
ADAPT:    reference + target geometry + measured anchors → adapted target
EXPLORE:  registered prior ↔ target-specific residual ↔ adapted field
FORECAST: adapted early state + tracking map → later molecular forecast
```

The software is a thin, regression-tested interface over the frozen manuscript implementation. It does not introduce a separate scientific model.
