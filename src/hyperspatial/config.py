from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Literal

@dataclass(frozen=True)
class DesignConfig:
    budget: int = 16
    budgets: tuple[int, ...] = (8, 16, 32)
    prevalence_min: float = 0.05
    prevalence_max: float = 0.98
    variance_min: float = 1e-5
    seed: int = 42116
    max_source_points: int = 5000
    exclusions: tuple[str, ...] = ()
    candidates: tuple[str, ...] = ()
    ridge: float = 1.0
    gain: float = 1.0

    def __post_init__(self):
        object.__setattr__(self,"budgets",tuple(self.budgets));object.__setattr__(self,"exclusions",tuple(self.exclusions));object.__setattr__(self,"candidates",tuple(self.candidates))
        if self.budget < 1: raise ValueError("budget must be positive")
        if self.ridge != 1.0 or self.gain != 1.0:
            raise ValueError("generic panel design uses prespecified frozen-grid ridge=1 and gain=1; target tuning is disabled")
    def to_dict(self): return asdict(self)

@dataclass(frozen=True)
class AdaptConfig:
    k_atlas: int = 12
    residual_neighbors: tuple[int, int] = (12, 32)
    rank: int = 16
    ridge_grid: tuple[float, ...] = (0.1, 1.0, 10.0)
    gain_grid: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    fusion_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    guard_mse_tolerance: float = 1.05
    guard_pearson_margin: float = 0.03
    seeds: tuple[int, ...] = (42, 43, 44)
    registration_epochs: int = 30
    registration: Literal["geometry", "pre_registered"] = "geometry"
    device: str = "cpu"

    def __post_init__(self):
        for name in ('residual_neighbors','ridge_grid','gain_grid','fusion_grid','seeds'):
            object.__setattr__(self,name,tuple(getattr(self,name)))
        frozen=(self.k_atlas,self.residual_neighbors,self.rank,self.ridge_grid,self.gain_grid,self.fusion_grid,self.guard_mse_tolerance,self.guard_pearson_margin)
        expected=(12,(12,32),16,(0.1,1.0,10.0),(0.25,0.5,0.75,1.0),(0.0,0.25,0.5,0.75,1.0),1.05,0.03)
        if frozen!=expected: raise ValueError("v1 scientific hyperparameters are frozen; target-specific tuning is disabled")
        if not self.seeds: raise ValueError("at least one algorithmic seed is required")
    def to_dict(self): return asdict(self)

@dataclass(frozen=True)
class ForecastConfig:
    rank: int = 16
    ridge_grid: tuple[float, ...] = (0.1, 1.0, 10.0)
    gain_grid: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    guard_mse_tolerance: float = 1.05
    guard_pearson_margin: float = 0.03
    seed: int = 42

    def __post_init__(self):
        object.__setattr__(self,"ridge_grid",tuple(self.ridge_grid));object.__setattr__(self,"gain_grid",tuple(self.gain_grid))
    def to_dict(self): return asdict(self)
