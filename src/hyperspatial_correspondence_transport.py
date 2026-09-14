"""Correspondence-aware local molecular transport for HyperSpatial.

This module deliberately keeps the molecular predictor local.  Geometry
registration is performed upstream; each observed stage then acts as a local
expert.  Expert visibility is inferred from source-only sampling density and
distance, so a source is not forced to explain target tissue outside its
support.  A zero-initialized low-rank residual is provided for source-only
pseudo-holdout training, but is inert until validation supports it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.spatial import cKDTree
import torch
from torch import nn


@dataclass(frozen=True)
class TransportMixtureConfig:
    k: int = 12
    distance_power: float = 2.0
    visibility_scale: float = 3.0
    minimum_visibility: float = 1e-4


@dataclass
class LocalExpertPrediction:
    values: np.ndarray
    mean_distance: np.ndarray
    support_ratio: np.ndarray
    visibility: np.ndarray
    effective_neighbors: np.ndarray


def _source_spacing(coordinates: np.ndarray, k: int) -> float:
    """Robust source-only estimate of local sampling distance."""
    tree = cKDTree(coordinates)
    probe = coordinates
    if len(probe) > 20_000:
        probe = probe[np.linspace(0, len(probe) - 1, 20_000, dtype=np.int64)]
    distances = tree.query(probe, k=min(max(2, k), len(coordinates)))[0]
    edge = distances[:, -1] if distances.ndim == 2 else distances
    positive = edge[np.isfinite(edge) & (edge > 0)]
    return max(float(np.median(positive)), 1e-8)


def local_source_expert(
    source_coordinates: np.ndarray,
    source_values: np.ndarray,
    query_coordinates: np.ndarray,
    config: TransportMixtureConfig = TransportMixtureConfig(),
) -> LocalExpertPrediction:
    """Predict one registered source stage at query points using adaptive IDW."""
    if len(source_coordinates) != len(source_values):
        raise ValueError("source coordinates and values must have equal rows")
    if source_coordinates.shape[1] != query_coordinates.shape[1]:
        raise ValueError("source and query coordinate dimensions differ")
    k = min(config.k, len(source_coordinates))
    distances, indices = cKDTree(source_coordinates).query(query_coordinates, k=k)
    if distances.ndim == 1:
        distances, indices = distances[:, None], indices[:, None]
    weights = np.maximum(distances, 1e-7) ** (-config.distance_power)
    weights /= np.maximum(weights.sum(1, keepdims=True), 1e-12)
    prediction = np.einsum("qk,qkg->qg", weights, source_values[indices]).astype(np.float32)

    spacing = _source_spacing(source_coordinates, config.k)
    mean_distance = np.sum(weights * distances, axis=1)
    support_ratio = distances[:, -1] / spacing
    visibility = np.exp(-0.5 * np.square(support_ratio / config.visibility_scale))
    visibility = np.maximum(visibility, config.minimum_visibility)
    effective = 1.0 / np.maximum(np.square(weights).sum(1), 1e-12)
    return LocalExpertPrediction(
        values=prediction,
        mean_distance=mean_distance.astype(np.float32),
        support_ratio=support_ratio.astype(np.float32),
        visibility=visibility.astype(np.float32),
        effective_neighbors=effective.astype(np.float32),
    )


def temporal_barycentric_priors(source_times: Sequence[float], query_time: float) -> np.ndarray:
    """Piecewise-linear temporal priors; extrapolation uses the nearest stage."""
    times = np.asarray(source_times, dtype=np.float64)
    if len(times) == 0 or len(np.unique(times)) != len(times):
        raise ValueError("source times must be non-empty and unique")
    order = np.argsort(times)
    times = times[order]
    weights = np.zeros(len(times), dtype=np.float64)
    if query_time <= times[0]:
        weights[0] = 1.0
    elif query_time >= times[-1]:
        weights[-1] = 1.0
    else:
        right = int(np.searchsorted(times, query_time))
        left = right - 1
        fraction = (query_time - times[left]) / (times[right] - times[left])
        weights[left], weights[right] = 1.0 - fraction, fraction
    restored = np.empty_like(weights)
    restored[order] = weights
    return restored.astype(np.float32)


def mix_local_experts(
    experts: Sequence[LocalExpertPrediction],
    temporal_priors: Sequence[float],
    source_fallback: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mix experts without forcing correspondence outside observed support.

    Returns prediction, normalized expert weights, and total support confidence.
    If all experts have weak support, the mixture conservatively shrinks toward
    a source-derived fallback profile (normally the training median profile).
    """
    if not experts:
        raise ValueError("at least one expert is required")
    priors = np.asarray(temporal_priors, dtype=np.float32)
    if priors.shape != (len(experts),) or np.any(priors < 0) or priors.sum() <= 0:
        raise ValueError("invalid temporal priors")
    raw = np.stack([p.visibility for p in experts], axis=1) * priors[None, :]
    support = np.clip(raw.sum(1), 0.0, 1.0)
    weights = raw / np.maximum(raw.sum(1, keepdims=True), 1e-12)
    values = np.stack([p.values for p in experts], axis=1)
    prediction = np.einsum("qe,qeg->qg", weights, values).astype(np.float32)
    if source_fallback is not None:
        fallback = np.asarray(source_fallback, dtype=np.float32)
        if fallback.shape != (prediction.shape[1],):
            raise ValueError("fallback must contain one source-derived value per gene")
        prediction = support[:, None] * prediction + (1.0 - support[:, None]) * fallback[None, :]
    return prediction.astype(np.float32), weights.astype(np.float32), support.astype(np.float32)


class LowRankTransportResidual(nn.Module):
    """Small correction to a transport baseline, initialized exactly at zero."""

    def __init__(self, n_genes: int, n_experts: int = 2, rank: int = 64, hidden: int = 128):
        super().__init__()
        self.expert_encoder = nn.Linear(n_genes, rank, bias=False)
        self.network = nn.Sequential(
            nn.Linear(3 + n_experts * (rank + 3), hidden),
            nn.SiLU(),
            nn.Linear(hidden, rank),
            nn.SiLU(),
        )
        self.decoder = nn.Linear(rank, n_genes, bias=False)
        self.alpha = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        coordinates: torch.Tensor,
        baseline: torch.Tensor,
        expert_values: torch.Tensor,
        expert_diagnostics: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if expert_values.ndim != 3 or expert_diagnostics.shape[-1] != 3:
            raise ValueError("expected [batch, experts, genes] and three diagnostics per expert")
        encoded = torch.tanh(self.expert_encoder(expert_values)).flatten(1)
        features = torch.cat((coordinates, encoded, expert_diagnostics.flatten(1)), dim=1)
        residual = self.decoder(self.network(features))
        prediction = torch.clamp(baseline + torch.tanh(self.alpha) * residual, min=0)
        return prediction, residual


class LocalCrossAttentionResidual(nn.Module):
    """Local temporal operator around a fixed molecular-transport baseline.

    Source expression is projected to a source-fitted rank-``rank`` basis
    before entering this module. Attention is restricted to the supplied local
    neighbours; therefore the model cannot invent a global coordinate field.
    The scalar residual gate is initialized at zero, making the initial output
    exactly equal to the transport baseline.
    """

    def __init__(self, n_genes: int, rank: int = 64, hidden: int = 128):
        super().__init__()
        self.rank = rank
        self.relative_encoder = nn.Sequential(
            nn.Linear(4, rank),
            nn.SiLU(),
            nn.Linear(rank, rank),
        )
        self.query_encoder = nn.Sequential(
            nn.Linear(4, rank),
            nn.SiLU(),
            nn.Linear(rank, rank),
        )
        self.key_projection = nn.Linear(rank, rank, bias=False)
        self.value_projection = nn.Linear(rank, rank, bias=False)
        self.residual_network = nn.Sequential(
            nn.Linear(2 * rank, hidden),
            nn.SiLU(),
            nn.Linear(hidden, rank),
            nn.SiLU(),
        )
        self.decoder = nn.Linear(rank, n_genes, bias=False)
        self.alpha = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        query_features: torch.Tensor,
        baseline: torch.Tensor,
        neighbour_latent: torch.Tensor,
        relative_space_time: torch.Tensor,
        log_attention_prior: torch.Tensor,
        neighbour_values: torch.Tensor | None = None,
        support: torch.Tensor | None = None,
        fallback: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return normalized-count prediction, residual and local attention.

        Shapes are ``[B,4]``, ``[B,G]``, ``[B,E,K,R]``,
        ``[B,E,K,4]`` and ``[B,E,K]`` respectively.
        """
        if query_features.ndim != 2 or query_features.shape[1] != 4:
            raise ValueError("query_features must be [batch, 4]")
        if neighbour_latent.ndim != 4 or neighbour_latent.shape[-1] != self.rank:
            raise ValueError("neighbour_latent must be [batch, experts, k, rank]")
        if relative_space_time.shape != (*neighbour_latent.shape[:-1], 4):
            raise ValueError("relative_space_time shape is inconsistent")
        if log_attention_prior.shape != neighbour_latent.shape[:-1]:
            raise ValueError("log_attention_prior shape is inconsistent")

        query = self.query_encoder(query_features)
        local = neighbour_latent + self.relative_encoder(relative_space_time)
        keys = self.key_projection(torch.tanh(local))
        values = self.value_projection(torch.tanh(neighbour_latent))
        scores = (keys * query[:, None, None, :]).sum(-1) / np.sqrt(self.rank)
        scores = scores + log_attention_prior
        attention = torch.softmax(scores.flatten(1), dim=1).reshape_as(scores)
        context = (attention[..., None] * values).sum(dim=(1, 2))
        residual = self.decoder(self.residual_network(torch.cat((query, context), dim=1)))
        molecular_baseline = baseline
        if neighbour_values is not None:
            if neighbour_values.shape[:-1] != attention.shape:
                raise ValueError("neighbour_values shape is inconsistent with attention")
            if neighbour_values.shape[-1] != baseline.shape[-1]:
                raise ValueError("neighbour_values and baseline gene dimensions differ")
            molecular_baseline = (
                attention[..., None] * neighbour_values
            ).sum(dim=(1, 2))
            if support is not None or fallback is not None:
                if support is None or fallback is None:
                    raise ValueError("support and fallback must be supplied together")
                molecular_baseline = (
                    support[:, None] * molecular_baseline
                    + (1 - support[:, None]) * fallback[None, :]
                )
        prediction_log = torch.clamp(
            torch.log1p(torch.clamp(molecular_baseline, min=0))
            + torch.tanh(self.alpha) * residual,
            min=0,
            max=12,
        )
        return torch.expm1(prediction_log), residual, attention
