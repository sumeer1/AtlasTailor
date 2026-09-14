from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class GeneResult:
    """One gene's registered prior, supported correction and adapted field."""
    gene: str
    registered_atlas: np.ndarray
    predicted_residual: np.ndarray
    adapted: np.ndarray
    support: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"gene": self.gene, "registered_atlas": self.registered_atlas,
                "predicted_residual": self.predicted_residual, "adapted": self.adapted,
                **self.support}

@dataclass
class ReferenceAtlas:
    expression: np.ndarray
    coordinates: np.ndarray
    genes: np.ndarray
    obs_names: np.ndarray | None = None
    source_path: Path | None = None
    counts: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class TargetSpecimen:
    coordinates: np.ndarray
    measured_expression: np.ndarray
    measured_genes: np.ndarray
    obs_names: np.ndarray | None = None
    source_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.measured_expression.shape != (len(self.coordinates), len(self.measured_genes)):
            raise ValueError("measured expression must be points × measured genes")

@dataclass
class RunManifest:
    run_id: str
    workflow: str
    status: str
    run_dir: Path
    data: dict[str, Any]

@dataclass
class PanelDesign:
    genes: list[str]
    ordered_panel: pd.DataFrame
    marker_utility: pd.DataFrame
    conditional_utility: pd.DataFrame
    budget_curve: pd.DataFrame
    coverage: pd.DataFrame
    manifest: RunManifest | None = None

    def export(self, out: str | Path):
        from .artifacts import export_panel
        return export_panel(self, Path(out))

    def report(self, out: str | Path):
        """Export the complete panel-design artifact bundle, including HTML/PDF."""
        return self.export(out)

@dataclass
class AdaptationResult:
    genes: np.ndarray
    coordinates: np.ndarray
    adapted: np.ndarray
    registered_atlas: np.ndarray
    predicted_residual: np.ndarray
    support: pd.DataFrame
    registered_source_coordinates: np.ndarray
    panel: list[str]
    manifest: RunManifest | None = None

    def gene(self, name: str) -> GeneResult:
        hit = np.flatnonzero(self.genes.astype(str) == str(name))
        if not len(hit): raise KeyError(name)
        j=int(hit[0]); row=self.support.iloc[j].to_dict()
        return GeneResult(str(name), self.registered_atlas[:,j],
                          self.predicted_residual[:,j], self.adapted[:,j], row)

    def export(self, out: str | Path):
        from .artifacts import export_adaptation
        return export_adaptation(self, Path(out))

    def report(self, path: str | Path):
        from .artifacts import write_report
        return write_report(self, Path(path))

@dataclass
class ForecastResult:
    genes: np.ndarray
    coordinates: np.ndarray
    transported_early: np.ndarray
    temporal_residual: np.ndarray
    forecast: np.ndarray
    support: pd.DataFrame
    trajectory_summary: pd.DataFrame
    manifest: RunManifest | None = None

    def export(self, out: str | Path):
        from .artifacts import export_forecast
        return export_forecast(self, Path(out))

    def gene(self, name: str) -> dict[str, Any]:
        hit=np.flatnonzero(self.genes.astype(str)==str(name))
        if not len(hit):raise KeyError(name)
        j=int(hit[0]);row=self.support.iloc[j].to_dict()
        return {'gene':str(name),'transported_early':self.transported_early[:,j],
                'temporal_residual':self.temporal_residual[:,j],'forecast':self.forecast[:,j],**row}
