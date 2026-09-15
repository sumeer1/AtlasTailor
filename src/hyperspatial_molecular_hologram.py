"""Minimal continuous geometry and molecular-field components for HyperSpatial.

This module is independent of Spateo.  Spateo is used only by benchmark runners
as an external comparator.  The geometry transform is a stationary neural
velocity field integrated in small steps, which makes the forward and inverse
maps share parameters and encourages a smooth, fold-free deformation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class FourierFeatures(nn.Module):
    def __init__(self, in_dim: int, n_frequencies: int = 6) -> None:
        super().__init__()
        frequencies = 2.0 ** torch.arange(n_frequencies, dtype=torch.float32)
        self.register_buffer("frequencies", frequencies)
        self.out_dim = in_dim * (1 + 2 * n_frequencies)

    def forward(self, x: Tensor) -> Tensor:
        phase = x[..., None] * self.frequencies
        return torch.cat((x, torch.sin(torch.pi * phase).flatten(-2), torch.cos(torch.pi * phase).flatten(-2)), -1)


class StationaryVelocityField(nn.Module):
    """Near-identity continuous deformation with a shared numerical inverse."""

    def __init__(self, dim: int = 3, hidden: int = 96, n_frequencies: int = 5, steps: int = 6) -> None:
        super().__init__()
        self.dim = dim
        self.steps = steps
        self.features = FourierFeatures(dim, n_frequencies)
        self.velocity_net = nn.Sequential(
            nn.Linear(self.features.out_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )
        nn.init.zeros_(self.velocity_net[-1].weight)
        nn.init.zeros_(self.velocity_net[-1].bias)
        self.log_scale = nn.Parameter(torch.zeros(()))
        self.rotation = nn.Parameter(torch.zeros(3 if dim == 3 else 1))
        self.translation = nn.Parameter(torch.zeros(dim))

    def _rotation_matrix(self) -> Tensor:
        if self.dim == 2:
            a = self.rotation[0]
            return torch.stack((torch.stack((torch.cos(a), -torch.sin(a))), torch.stack((torch.sin(a), torch.cos(a)))))
        w = self.rotation
        theta = torch.linalg.vector_norm(w).clamp_min(1e-8)
        k = w / theta
        zero = torch.zeros((), device=w.device, dtype=w.dtype)
        K = torch.stack(
            (
                torch.stack((zero, -k[2], k[1])),
                torch.stack((k[2], zero, -k[0])),
                torch.stack((-k[1], k[0], zero)),
            )
        )
        eye = torch.eye(3, device=w.device, dtype=w.dtype)
        return eye + torch.sin(theta) * K + (1.0 - torch.cos(theta)) * (K @ K)

    def affine(self, x: Tensor, inverse: bool = False) -> Tensor:
        r = self._rotation_matrix()
        scale = torch.exp(self.log_scale)
        if inverse:
            return ((x - self.translation) @ r) / scale
        return scale * (x @ r.T) + self.translation

    def velocity(self, x: Tensor) -> Tensor:
        return 0.15 * torch.tanh(self.velocity_net(self.features(x)))

    def integrate(self, x: Tensor, inverse: bool = False) -> Tensor:
        y = self.affine(x, inverse=inverse) if not inverse else x
        sign = -1.0 if inverse else 1.0
        dt = sign / float(self.steps)
        for _ in range(self.steps):
            y = y + dt * self.velocity(y)
        return y if not inverse else self.affine(y, inverse=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.integrate(x)

    def inverse(self, x: Tensor) -> Tensor:
        return self.integrate(x, inverse=True)


def trimmed_chamfer(x: Tensor, y: Tensor, keep: float = 0.9) -> Tensor:
    d2 = torch.cdist(x, y).square()
    dx, dy = d2.min(1).values, d2.min(0).values

    def trimmed_mean(v: Tensor) -> Tensor:
        n = max(1, int(keep * v.numel()))
        return torch.topk(v, n, largest=False).values.mean()

    return 0.5 * (trimmed_mean(dx) + trimmed_mean(dy))


def jacobian_determinants(field: StationaryVelocityField, points: Tensor) -> Tensor:
    """Jacobian determinants of the integrated transform at supplied points."""
    x = points.detach().requires_grad_(True)
    y = field(x)
    rows = []
    for j in range(field.dim):
        grad = torch.autograd.grad(y[:, j].sum(), x, create_graph=True, retain_graph=True)[0]
        rows.append(grad)
    return torch.linalg.det(torch.stack(rows, dim=1))


@dataclass
class RegistrationConfig:
    epochs: int = 30
    lr: float = 2e-3
    batch_size: int = 768
    chamfer_keep: float = 0.85
    lambda_cycle: float = 0.2
    lambda_smooth: float = 0.02
    lambda_fold: float = 0.2
    seed: int = 42


def fit_geometry_field(
    source: np.ndarray,
    target: np.ndarray,
    device: str,
    config: RegistrationConfig,
) -> Tuple[StationaryVelocityField, Dict[str, list]]:
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    dim = int(source.shape[1])
    field = StationaryVelocityField(dim=dim).to(device)
    optimizer = torch.optim.AdamW(field.parameters(), lr=config.lr, weight_decay=1e-6)
    src = torch.as_tensor(source, dtype=torch.float32, device=device)
    tgt = torch.as_tensor(target, dtype=torch.float32, device=device)
    history: Dict[str, list] = {"loss": [], "chamfer": [], "cycle": [], "smooth": [], "fold": []}
    for _ in range(config.epochs):
        si = rng.choice(len(src), min(config.batch_size, len(src)), replace=False)
        ti = rng.choice(len(tgt), min(config.batch_size, len(tgt)), replace=False)
        x, y = src[si], tgt[ti]
        mapped = field(x)
        chamfer = trimmed_chamfer(mapped, y, config.chamfer_keep)
        cycle = F.mse_loss(field.inverse(mapped), x)
        jitter = 0.01 * torch.randn_like(x)
        smooth = ((field.velocity(x + jitter) - field.velocity(x)) / 0.01).square().mean()
        ji = rng.choice(len(x), min(64, len(x)), replace=False)
        determinants = jacobian_determinants(field, x[ji])
        fold = F.relu(0.05 - determinants).square().mean()
        loss = chamfer + config.lambda_cycle * cycle + config.lambda_smooth * smooth + config.lambda_fold * fold
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(field.parameters(), 5.0)
        optimizer.step()
        for key, value in (("loss", loss), ("chamfer", chamfer), ("cycle", cycle), ("smooth", smooth), ("fold", fold)):
            history[key].append(float(value.detach().cpu()))
    return field, history


class MolecularField(nn.Module):
    """Count-aware low-rank molecular field over canonical position and time."""

    def __init__(self, n_genes: int, dim: int = 3, rank: int = 64, hidden: int = 128) -> None:
        super().__init__()
        self.dim = dim
        self.features = FourierFeatures(dim + 1, 6)
        self.encoder = nn.Sequential(
            nn.Linear(self.features.out_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, rank),
        )
        self.decoder = nn.Linear(rank, n_genes)
        self.reaction = nn.Sequential(nn.Linear(rank, rank), nn.Tanh(), nn.Linear(rank, rank))
        self.log_diffusion = nn.Parameter(torch.full((rank,), -4.0))

    def latent(self, x: Tensor, time: Tensor) -> Tensor:
        return self.encoder(self.features(torch.cat((x, time[:, None]), dim=-1)))

    def forward(self, x: Tensor, time: Tensor, log_depth: Optional[Tensor] = None) -> Tensor:
        log_rate = self.decoder(self.latent(x, time))
        if log_depth is not None:
            log_rate = log_rate + log_depth[:, None]
        return F.softplus(log_rate)

    def pde_residual(self, x: Tensor, time: Tensor, velocity: Optional[Tensor] = None) -> Tensor:
        x = x.requires_grad_(True)
        time = time.requires_grad_(True)
        z = self.latent(x, time)
        velocity = torch.zeros_like(x) if velocity is None else velocity
        residuals = []
        diffusion = F.softplus(self.log_diffusion)
        reaction = self.reaction(z)
        for k in range(z.shape[1]):
            grad_x = torch.autograd.grad(z[:, k].sum(), x, create_graph=True, retain_graph=True)[0]
            grad_t = torch.autograd.grad(z[:, k].sum(), time, create_graph=True, retain_graph=True)[0]
            laplacian = 0.0
            for d in range(self.dim):
                laplacian = laplacian + torch.autograd.grad(
                    grad_x[:, d].sum(), x, create_graph=True, retain_graph=True
                )[0][:, d]
            residuals.append(grad_t + (velocity * grad_x).sum(1) - diffusion[k] * laplacian - reaction[:, k])
        return torch.stack(residuals, 1)


def poisson_deviance_loss(observed: Tensor, rate: Tensor) -> Tensor:
    return (rate - observed * torch.log(rate.clamp_min(1e-8))).mean()

