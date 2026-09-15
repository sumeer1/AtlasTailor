"""Public HyperSpatial-MAP lab API.

The implementation composes the executed manuscript functions. Target truth is
deliberately accepted only by :func:`validate`, never by :func:`adapt`.
"""
from .config import AdaptConfig, DesignConfig, ForecastConfig
from .models import (AdaptationResult, ForecastResult, GeneResult, PanelDesign,
                     ReferenceAtlas, RunManifest, TargetSpecimen)
from .workflows import adapt, design_panel, explore, forecast, inspect_run, validate

__all__ = [
    "AdaptConfig", "DesignConfig", "ForecastConfig", "ReferenceAtlas",
    "TargetSpecimen", "PanelDesign", "AdaptationResult", "GeneResult", "ForecastResult",
    "RunManifest", "design_panel", "adapt", "explore", "forecast",
    "validate", "inspect_run",
]
__version__ = "0.1.0"
