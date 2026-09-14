#!/usr/bin/env python3
"""
HyperSpatial v7.1 — Merged Production Model
=============================================

Merges best of v7 + v2.1:

From v7 (kept):
  - 4-layer implicit field with skip connection
  - Bilinear gene-conditioned decoder (emb=128)
  - ZINB likelihood for spot-level reconstruction
  - Per-timepoint x,y normalization
  - Per-timepoint z normalization (NOT global — v2.1 was wrong here)
  - Full gene set support (13615)

From v2.1 (adopted):
  ✓ Continuous hours as time coordinate (not ordinal integers)
  ✓ Separate time_id (integer) for batching vs time coordinate (continuous)
  ✓ log1p_mse / log1p_cosine for niche & temporal auxiliary losses
  ✓ CyclicBalancedTimeBatchSampler (wraps small groups, uses all data)
  ✓ Early stopping with validation split + best weight restoration
  ✓ Gene variance loss in log1p space: var(log1p(x))
  ✓ Lower auxiliary loss weights (0.05-0.10)
"""

from __future__ import annotations
import math, random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import anndata as ad
import scanpy as sc
import scipy.sparse as sp
from torch.utils.data import Dataset, DataLoader, Sampler, Subset


# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _to_dense_float32(X) -> np.ndarray:
    if sp.issparse(X): X = X.toarray()
    elif not isinstance(X, np.ndarray): X = np.asarray(X)
    return X.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Preprocessing
# ══════════════════════════════════════════════════════════════════════════════

class Preprocessor:
    @staticmethod
    def preprocess_counts_then_hvg(adata, n_top_genes=2000, target_sum=1e4,
                                    min_counts=1, min_cells=3,
                                    hvg_flavor="seurat_v3", counts_layer=None):
        adata = adata.copy()
        X = (adata.layers[counts_layer]
             if counts_layer is not None and counts_layer in adata.layers
             else adata.X)
        X_counts = X.copy() if sp.issparse(X) else np.asarray(X, dtype=np.float32).copy()

        tmp = ad.AnnData(X=X_counts, obs=adata.obs.copy(), var=adata.var.copy())
        sc.pp.filter_genes(tmp, min_counts=min_counts)
        sc.pp.filter_genes(tmp, min_cells=min_cells)
        keep = tmp.var_names
        orig_idx = adata.var_names.get_indexer(keep)
        assert (orig_idx != -1).all()

        adata = adata[:, keep].copy()
        adata.layers["counts"] = X_counts[:, orig_idx].copy()

        hvg = ad.AnnData(X=adata.layers["counts"].copy(), obs=adata.obs.copy(), var=adata.var.copy())
        sc.pp.normalize_total(hvg, target_sum=target_sum)
        sc.pp.log1p(hvg)
        sc.pp.highly_variable_genes(hvg, n_top_genes=min(n_top_genes, hvg.n_vars), flavor=hvg_flavor)
        return adata[:, hvg.var["highly_variable"].to_numpy()].copy()

    @staticmethod
    def get_coords_per_timepoint_norm(adata, spatial_key="spatial", depth_key="z",
                                       time_key="time_hours"):
        """
        Build 4D coords with:
          - x,y normalized PER timepoint (consistent spatial semantics)
          - z normalized PER timepoint (different # slices per time — v7 original, NOT v2.1)
          - t = continuous hours normalized to [0,1] (from v2.1)
        """
        xy = np.asarray(adata.obsm[spatial_key], dtype=np.float32)
        z = (np.asarray(adata.obs[depth_key].values, dtype=np.float32)
             if depth_key in adata.obs else xy[:, 2].astype(np.float32))

        has_time = time_key in adata.obs
        if not has_time:
            c3 = np.column_stack([xy[:, :2], z]) if xy.shape[1] == 2 else xy[:, :3].copy()
            mn, mx = c3.min(0, keepdims=True), c3.max(0, keepdims=True)
            return ((c3 - mn) / (mx - mn + 1e-8)).astype(np.float32), False

        t_hours = np.asarray(adata.obs[time_key].values, dtype=np.float32)

        # We need a discrete key for per-timepoint grouping
        # Use time_id if available, else discretize hours
        if "time_id" in adata.obs:
            t_discrete = adata.obs["time_id"].values.astype(int)
        else:
            unique_h = np.unique(t_hours)
            t_discrete = np.searchsorted(unique_h, t_hours)

        xy_n = np.zeros_like(xy[:, :2])
        z_n = np.zeros_like(z)

        for tid in np.unique(t_discrete):
            m = t_discrete == tid
            # x,y per-timepoint
            mn, mx = xy[m, :2].min(0, keepdims=True), xy[m, :2].max(0, keepdims=True)
            xy_n[m] = (xy[m, :2] - mn) / (mx - mn + 1e-8)
            # z per-timepoint (CRITICAL: kept from v7, not global like v2.1)
            z_m = z[m]
            z_n[m] = (z_m - z_m.min()) / (z_m.max() - z_m.min() + 1e-8)

        # t normalized to [0,1] using continuous hours (from v2.1)
        t_n = (t_hours - t_hours.min()) / (t_hours.max() - t_hours.min() + 1e-8)

        out = np.column_stack([xy_n, z_n, t_n]).astype(np.float32)
        assert np.all(np.isfinite(out))
        return out, True


# ══════════════════════════════════════════════════════════════════════════════
# Encoders
# ══════════════════════════════════════════════════════════════════════════════

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, input_dim=3, n_freqs=16):
        super().__init__()
        self.W = nn.Parameter(torch.randn(input_dim, n_freqs) * 2.0)
        self.b = nn.Parameter(torch.zeros(n_freqs))

    def forward(self, coords):
        a = 2 * math.pi * (coords @ self.W + self.b)
        return torch.cat([torch.sin(a), torch.cos(a)], dim=-1)


class NeuralImplicitField(nn.Module):
    def __init__(self, input_dim=3, latent_dim=256, n_pos_freqs=16,
                 hidden_layers=(512, 512, 512, 512)):
        super().__init__()
        self.pos_enc = LearnedPositionalEncoding(input_dim, n_pos_freqs)
        enc_dim = input_dim + 2 * n_pos_freqs
        mid = len(hidden_layers) // 2

        layers1 = []
        prev = enc_dim
        for h in hidden_layers[:mid]:
            layers1 += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(0.05)]
            prev = h
        self.block1 = nn.Sequential(*layers1)

        layers2 = []
        prev = prev + enc_dim  # skip
        for h in hidden_layers[mid:]:
            layers2 += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(0.05)]
            prev = h
        layers2.append(nn.Linear(prev, latent_dim))
        self.block2 = nn.Sequential(*layers2)

    def forward(self, coords):
        x = torch.cat([coords, self.pos_enc(coords)], dim=-1)
        h = self.block1(x)
        return self.block2(torch.cat([h, x], dim=-1))


# ══════════════════════════════════════════════════════════════════════════════
# Decoders
# ══════════════════════════════════════════════════════════════════════════════

class GeneBilinearDecoder(nn.Module):
    def __init__(self, n_genes, latent_dim=256, gene_emb_dim=128,
                 proj_hidden=256, dropout=0.05):
        super().__init__()
        self.gene_emb = nn.Parameter(torch.randn(n_genes, gene_emb_dim) * 0.02)
        self.gene_bias = nn.Parameter(torch.zeros(n_genes))
        self.proj = nn.Sequential(
            nn.Linear(latent_dim, proj_hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(proj_hidden, gene_emb_dim))

    def forward(self, latent):
        return F.softplus(self.proj(latent) @ self.gene_emb.t() + self.gene_bias)


class PiBilinearDecoder(nn.Module):
    def __init__(self, n_genes, latent_dim=256, gene_emb_dim=128,
                 proj_hidden=256, dropout=0.05):
        super().__init__()
        self.gene_emb = nn.Parameter(torch.randn(n_genes, gene_emb_dim) * 0.02)
        self.gene_bias = nn.Parameter(torch.zeros(n_genes))
        self.proj = nn.Sequential(
            nn.Linear(latent_dim, proj_hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(proj_hidden, gene_emb_dim))

    def forward(self, latent):
        return torch.sigmoid(self.proj(latent) @ self.gene_emb.t() + self.gene_bias)


# ══════════════════════════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════════════════════════

class HyperSpatialModel(nn.Module):
    def __init__(self, n_genes, n_cell_types=10, latent_dim=256, n_pos_freqs=16,
                 gene_emb_dim=128, use_zinb=True, shared_dispersion=False,
                 pi_bilinear=True, has_temporal=False):
        super().__init__()
        self.n_genes = n_genes
        self.n_cell_types = n_cell_types
        self.latent_dim = latent_dim
        self.use_zinb = use_zinb
        self.shared_dispersion = shared_dispersion
        self.has_temporal = has_temporal

        inp = 4 if has_temporal else 3
        self.implicit_field = NeuralImplicitField(
            inp, latent_dim, n_pos_freqs, (512, 512, 512, 512))
        self.cell_type_decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(256, n_cell_types))
        self.mu_decoder = GeneBilinearDecoder(
            n_genes, latent_dim, gene_emb_dim, proj_hidden=256)
        self.log_theta = (nn.Parameter(torch.tensor(-2.0)) if shared_dispersion
                          else nn.Parameter(torch.full((n_genes,), -2.0)))
        self.pi_bilinear = bool(pi_bilinear)
        if use_zinb:
            self.pi_decoder = (PiBilinearDecoder(n_genes, latent_dim, gene_emb_dim, 256)
                               if pi_bilinear else
                               nn.Sequential(nn.Linear(latent_dim, 256), nn.GELU(),
                                             nn.Linear(256, n_genes)))

    def forward(self, coords, return_latent=False):
        latent = self.implicit_field(coords)
        mu = torch.clamp(self.mu_decoder(latent), min=1e-6, max=1e6)
        theta = torch.exp(torch.clamp(self.log_theta, -6, 6))
        out = {"cell_type_logits": self.cell_type_decoder(latent),
               "mu": mu, "theta": theta}
        if self.use_zinb:
            pi = (self.pi_decoder(latent) if self.pi_bilinear
                  else torch.sigmoid(self.pi_decoder(latent)))
            out["pi"] = torch.clamp(pi, 1e-6, 1 - 1e-6)
        if return_latent:
            out["latent"] = latent
        return out

    @staticmethod
    def _f32(*ts):
        return tuple(t.float() if t.dtype != torch.float32 else t for t in ts)

    def nb_loss(self, counts, mu, theta, eps=1e-8):
        if theta.dim() == 0: theta = theta.expand(mu.shape[1])
        theta = theta.unsqueeze(0)
        counts, mu, theta = self._f32(counts, mu, theta)
        mu, theta = mu.clamp(min=eps, max=1e12), theta.clamp(min=eps, max=1e12)
        lrm = torch.log(theta + mu)
        lp = (torch.lgamma(counts + theta) - torch.lgamma(theta) - torch.lgamma(counts + 1)
              + theta * (torch.log(theta) - lrm) + counts * (torch.log(mu) - lrm))
        return -torch.where(torch.isfinite(lp), lp, torch.zeros_like(lp)).mean()

    def zinb_loss(self, counts, mu, theta, pi, eps=1e-8):
        if theta.dim() == 0: theta = theta.expand(mu.shape[1])
        theta = theta.unsqueeze(0)
        counts, mu, theta, pi = self._f32(counts, mu, theta, pi)
        mu = mu.clamp(min=eps, max=1e12)
        theta = theta.clamp(min=eps, max=1e12)
        pi = pi.clamp(min=eps, max=1 - eps)
        lrm = torch.log(theta + mu)
        nb_lp = (torch.lgamma(counts + theta) - torch.lgamma(theta) - torch.lgamma(counts + 1)
                 + theta * (torch.log(theta) - lrm) + counts * (torch.log(mu) - lrm))
        nb_case = torch.log1p(-pi + eps) + nb_lp
        zero_case = torch.logsumexp(
            torch.stack([torch.log(pi + eps), torch.log1p(-pi + eps) + nb_lp]), dim=0)
        zm = (counts == 0).float()
        lp = zm * zero_case + (1 - zm) * nb_case
        return -torch.where(torch.isfinite(lp), lp, torch.zeros_like(lp)).mean()


# ══════════════════════════════════════════════════════════════════════════════
# Niche utilities — use time_ids (integer) for grouping, not continuous coords
# ══════════════════════════════════════════════════════════════════════════════

def build_batch_knn_niches_same_time(coords, time_ids=None, k=10,
                                      n_anchors=128, seed=None):
    """time_ids: integer tensor for same-timepoint grouping."""
    B = coords.shape[0]
    if B == 0:
        return (torch.empty(0, dtype=torch.long, device=coords.device),
                torch.empty(0, 0, dtype=torch.long, device=coords.device))
    A = min(n_anchors, B)
    g = torch.Generator(device=coords.device)
    if seed is not None: g.manual_seed(seed)
    pool = torch.randperm(B, generator=g, device=coords.device)[:A]
    xyz = coords[:, :3]
    anc, nch = [], []
    for a in pool:
        if time_ids is not None:
            cand = torch.where(time_ids == time_ids[a])[0]
        else:
            cand = torch.arange(B, device=coords.device)
        if cand.numel() <= 1: continue
        d = torch.cdist(xyz[a:a+1], xyz[cand]).squeeze(0)
        nn_l = torch.topk(d, min(k+1, cand.numel()), largest=False).indices
        nn_idx = cand[nn_l]
        nn_idx = nn_idx[nn_idx != a]
        if nn_idx.numel() == 0: continue
        if nn_idx.numel() >= k: nn_idx = nn_idx[:k]
        else: nn_idx = torch.cat([nn_idx, nn_idx[-1].expand(k - nn_idx.numel())])
        anc.append(a); nch.append(nn_idx)
    if not nch:
        return (torch.empty(0, dtype=torch.long, device=coords.device),
                torch.empty(0, 0, dtype=torch.long, device=coords.device))
    return torch.stack(anc), torch.stack(nch)


def build_temporal_niches(coords, time_ids, k=6, max_delta=1,
                           n_anchors=128, seed=None):
    """time_ids: integer tensor for cross-timepoint neighbors."""
    B = coords.shape[0]
    if B == 0 or time_ids is None: return None, None
    xyz = coords[:, :3]
    A = min(n_anchors, B)
    g = torch.Generator(device=coords.device)
    if seed is not None: g.manual_seed(seed)
    pool = torch.randperm(B, generator=g, device=coords.device)[:A]
    ut = torch.unique(time_ids, sorted=True)
    if ut.numel() <= 1: return None, None
    anc, nch = [], []
    for a in pool:
        ta = time_ids[a]
        ot = ut[ut != ta]
        if ot.numel() == 0: continue
        delta = torch.abs(ot - ta)
        chosen = ot[delta <= max_delta]
        if chosen.numel() == 0: chosen = ot[torch.argsort(delta)[:1]]
        mask = torch.isin(time_ids, chosen)
        if mask.sum() == 0: continue
        d = torch.cdist(xyz[a:a+1], xyz[mask]).squeeze(0)
        ke = min(k, int(mask.sum().item()))
        nn = torch.topk(d, ke, largest=False).indices
        idx = torch.where(mask)[0][nn]
        if idx.numel() < k: idx = torch.cat([idx, idx[-1].expand(k - idx.numel())])
        anc.append(a); nch.append(idx[:k])
    if not nch: return None, None
    return torch.stack(anc), torch.stack(nch)


def build_spot_knn_same_time(coords, time_ids=None, k=10):
    B = coords.shape[0]
    if B <= 1: return torch.empty(B, 0, dtype=torch.long, device=coords.device)
    xyz = coords[:, :3]
    out = torch.full((B, k), 0, dtype=torch.long, device=coords.device)
    if time_ids is not None:
        for tv in torch.unique(time_ids):
            m = time_ids == tv; idx = torch.where(m)[0]; nt = idx.shape[0]
            if nt <= 1: out[idx[0]] = idx[0].expand(k); continue
            d = torch.cdist(xyz[idx], xyz[idx]); d.fill_diagonal_(float('inf'))
            ke = min(k, nt-1)
            _, tl = torch.topk(d, ke, largest=False, dim=1)
            kg = idx[tl]
            if ke < k: kg = torch.cat([kg, kg[:, -1:].expand(-1, k-ke)], 1)
            out[idx] = kg
    else:
        d = torch.cdist(xyz, xyz); d.fill_diagonal_(float('inf'))
        ke = min(k, B-1)
        _, ti = torch.topk(d, ke, largest=False, dim=1)
        if ke < k: ti = torch.cat([ti, ti[:, -1:].expand(-1, k-ke)], 1)
        out = ti
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Dataset — returns (coords, counts, sf, ct, time_id)
# ══════════════════════════════════════════════════════════════════════════════

class SpatialDataset(Dataset):
    def __init__(self, adata, coords, sf_min=0.2, sf_max=5.0,
                 celltype_key="leiden", counts_layer="counts",
                 time_id_key="time_id", keep_sparse=True):
        self.coords = coords.astype(np.float32)
        X = adata.layers[counts_layer] if counts_layer in adata.layers else adata.X
        self.is_sparse = sp.issparse(X) and keep_sparse
        self.counts = X.tocsr() if self.is_sparse else _to_dense_float32(X)

        lib = (np.asarray(self.counts.sum(1)).ravel().astype(np.float32)
               if self.is_sparse else self.counts.sum(1).astype(np.float32))
        sf = lib / (np.median(lib) + 1e-8)
        self.sf = np.clip(sf, sf_min, sf_max).astype(np.float32)

        self.has_ct = celltype_key in adata.obs
        if self.has_ct:
            ct = adata.obs[celltype_key]
            if hasattr(ct, "cat"): self.ct = ct.cat.codes.to_numpy().astype(np.int64)
            else:
                v = ct.to_numpy()
                self.ct = (v.astype(np.int64) if np.issubdtype(v.dtype, np.integer)
                           else np.unique(v, return_inverse=True)[1].astype(np.int64))
        else:
            self.ct = None

        self.time_ids = (adata.obs[time_id_key].to_numpy().astype(np.int64)
                         if time_id_key in adata.obs else None)
        self.n_spots, self.n_genes = adata.n_obs, adata.n_vars

    def __len__(self): return self.n_spots

    def __getitem__(self, i):
        c = torch.from_numpy(self.coords[i])
        x = (torch.from_numpy(self.counts.getrow(i).toarray().ravel().astype(np.float32))
             if self.is_sparse else torch.from_numpy(self.counts[i]))
        ct = torch.tensor(-1 if self.ct is None else self.ct[i], dtype=torch.long)
        tid = torch.tensor(-1 if self.time_ids is None else self.time_ids[i], dtype=torch.long)
        return c, x, torch.tensor(self.sf[i], dtype=torch.float32), ct, tid


# ══════════════════════════════════════════════════════════════════════════════
# CyclicBalancedTimeBatchSampler (from v2.1 — wraps small groups)
# ══════════════════════════════════════════════════════════════════════════════

class CyclicBalancedTimeBatchSampler(Sampler):
    """Uses ALL data from every timepoint by cycling smaller groups."""
    def __init__(self, time_ids, batch_size, seed=42):
        self.time_ids = np.asarray(time_ids).astype(int)
        self.rng = np.random.default_rng(seed)
        self.times = np.unique(self.time_ids)
        self.n_times = len(self.times)
        assert self.n_times >= 2
        self.per_time = max(1, batch_size // self.n_times)
        self.batch_size = self.per_time * self.n_times
        self.by_time = {t: np.where(self.time_ids == t)[0] for t in self.times}
        # n_batches: use median group size — balances coverage vs speed
        # Small groups wrap (see all data), large groups subsample per epoch
        sizes = [len(v) for v in self.by_time.values()]
        self.n_batches = max(1, int(np.ceil(np.median(sizes) / self.per_time)))

    def __iter__(self):
        pools = {t: self.rng.permutation(idx) for t, idx in self.by_time.items()}
        ptrs = {t: 0 for t in self.times}
        for _ in range(self.n_batches):
            batch = []
            for t in self.times:
                pool = pools[t]
                need = self.per_time
                take = []
                while need > 0:
                    rem = len(pool) - ptrs[t]
                    if rem == 0:
                        pool = self.rng.permutation(self.by_time[t])
                        pools[t] = pool
                        ptrs[t] = 0
                        rem = len(pool)
                    k = min(need, rem)
                    take.extend(pool[ptrs[t]:ptrs[t]+k].tolist())
                    ptrs[t] += k
                    need -= k
                batch.extend(take)
            self.rng.shuffle(batch)
            yield batch

    def __len__(self): return self.n_batches


# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrainerConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    betas: tuple = (0.9, 0.999)
    adam_eps: float = 1e-8

    # Lower auxiliary weights (from v2.1)
    lambda_niche: float = 0.10    #0.10, 0.05, 0.03 #zebra 0.10, try 0.05 and 0.10, 0.05 drosp
    lambda_temporal: float = 0.10 #0.10, 0.05 drosp
    lambda_celltype: float = 0.05
    lambda_niche_ct: float = 0.05
    lambda_gene_var: float = 0.10   #0.05, 0.10, 0.15, 0,15,0.30 drosp

    niche_k: int = 10
    niche_anchors: int = 128
    temporal_k: int = 6
    temporal_dt: int = 1
    soft_niche_k: int = 10

    grad_clip: float = 1.0
    use_amp: bool = True
    celltype_ignore_index: int = -1
    label_smoothing: float = 0.05
    t_max: int = 200
    eta_min: float = 1e-6
    warmup_epochs: int = 10

    # From v2.1: auxiliary loss type
    niche_loss_type: str = "log1p_mse"     # log1p_mse | log1p_cosine | none
    temporal_loss_type: str = "log1p_mse"

    # From v2.1: early stopping
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 1e-4
    restore_best_weights: bool = True


# ══════════════════════════════════════════════════════════════════════════════
# Trainer — unified _run_epoch for train and val
# ══════════════════════════════════════════════════════════════════════════════

class HyperSpatialTrainer:
    def __init__(self, model, config, device="cuda", class_counts=None):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.device_type = "cuda" if str(device).startswith("cuda") else "cpu"

        self.class_weights = None
        if class_counts is not None:
            c = np.maximum(np.array(class_counts, dtype=np.float64), 1.0)
            w = np.sqrt(np.median(c) / c); w /= w.mean()
            self.class_weights = torch.tensor(w, dtype=torch.float32).to(device)

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.lr, betas=config.betas,
            weight_decay=config.weight_decay, eps=config.adam_eps)

        wu = max(config.warmup_epochs, 1)
        warmup = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=0.01, total_iters=wu)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(config.t_max - wu, 1), eta_min=config.eta_min)
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer, [warmup, cosine], [wu])
        self.scaler = torch.amp.GradScaler(
            self.device_type,
            enabled=bool(config.use_amp and self.device_type == "cuda"))

    def _nll(self, counts, mu, theta, pi=None):
        return (self.model.nb_loss(counts, mu, theta) if pi is None
                else self.model.zinb_loss(counts, mu, theta, pi))

    def _consistency_loss(self, pred_mu, target_counts, loss_type):
        """From v2.1: log1p MSE or cosine for auxiliary niche losses."""
        if loss_type == "none":
            return torch.zeros((), device=pred_mu.device)
        pred = torch.log1p(pred_mu)
        target = torch.log1p(target_counts)
        if loss_type == "log1p_mse":
            return F.mse_loss(pred, target)
        if loss_type == "log1p_cosine":
            return 1.0 - F.cosine_similarity(pred, target, dim=1).mean()
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    def _spatial_gradient_loss(self, pred_mu, target_counts, coords, time_ids, k=8):
        """
        Preserve spatial contrast instead of smoothing it away.
        Matches anchor-neighbor differences in log1p space.
        """
        knn = build_spot_knn_same_time(coords, time_ids=time_ids, k=k)
        if knn.numel() == 0:
            return torch.zeros((), device=pred_mu.device)

        log_pred = torch.log1p(pred_mu)
        log_true = torch.log1p(target_counts.float())

        pred_diff = log_pred.unsqueeze(1) - log_pred[knn]
        true_diff = log_true.unsqueeze(1) - log_true[knn]

        pred_diff = torch.clamp(pred_diff, -5, 5)
        true_diff = torch.clamp(true_diff, -5, 5)

        mag_loss = F.mse_loss(pred_diff, true_diff)

        sign_loss = 1.0 - F.cosine_similarity(
            pred_diff.reshape(-1, pred_diff.shape[-1]),
            true_diff.reshape(-1, true_diff.shape[-1]),
            dim=-1
        ).mean()

        return mag_loss + 0.5 * sign_loss
    
    def _gene_moment_loss(self, mu, counts, sf):
        """
        Match gene-wise first and second moments on library-size-normalized rates.
        Compute in fp32 for numerical stability.
        """
        with torch.amp.autocast(self.device_type, enabled=False):
            mu_f = mu.float()
            c_f = counts.float()
            sf_f = sf.float().unsqueeze(1).clamp_min(1e-6)

            rate_pred = mu_f / sf_f
            rate_true = c_f / sf_f

            log_p = torch.log1p(rate_pred)
            log_t = torch.log1p(rate_true)

            mean_p = log_p.mean(0)
            mean_t = log_t.mean(0)

            std_p = log_p.std(0, unbiased=False)
            std_t = log_t.std(0, unbiased=False)

            mean_loss = F.mse_loss(mean_p, mean_t)
            std_loss = F.mse_loss(std_p, std_t)

            cv_pred = std_p / mean_p.clamp_min(1e-3)
            cv_true = std_t / mean_t.clamp_min(1e-3)
            cv_loss = F.mse_loss(cv_pred, cv_true)

        return 0.5 * mean_loss + std_loss + 0.5 * cv_loss

    def _soft_niche_ct(self, coords, logits, ct, time_ids):
        knn = build_spot_knn_same_time(coords, time_ids=time_ids,
                                        k=self.config.soft_niche_k)
        if knn.numel() == 0: return torch.zeros((), device=coords.device)
        probs = F.softmax(logits, -1)
        np_ = probs[knn].mean(1)
        valid = ct != self.config.celltype_ignore_index
        if valid.sum() == 0: return torch.zeros((), device=coords.device)
        oh = F.one_hot(ct.clamp(min=0), logits.shape[1]).float() * valid.unsqueeze(1).float()
        nt = oh[knn].mean(1)
        d = ((np_ - nt) ** 2).sum(1)
        return (d * valid.float()).sum() / valid.float().sum().clamp_min(1)

    def _run_epoch(self, loader, epoch, train=True):
        self.model.train(train)
        cfg = self.config
        stats = {k: [] for k in
                 ["loss","nll_spot","nll_spatial","nll_temporal","ce","niche_ct","gene_var"]}
        correct, total, skipped = 0, 0, 0

        for bi, (coords, counts, sf, ct, time_ids) in enumerate(loader):
            coords = coords.to(self.device, non_blocking=True)
            counts = counts.to(self.device, non_blocking=True)
            sf = sf.to(self.device, non_blocking=True).float()
            ct = ct.to(self.device, non_blocking=True).long()
            time_ids = time_ids.to(self.device, non_blocking=True).long()
            if train: self.optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(train):
                with torch.amp.autocast(self.device_type,
                                        enabled=bool(cfg.use_amp and self.device_type == "cuda")):
                    out = self.model(coords, return_latent=True)
                    mu = out["mu"] * sf.unsqueeze(1)
                    pi = out.get("pi") if self.model.use_zinb else None
                    nll_spot = self._nll(counts, mu, out["theta"], pi)

                    # Niche loss: log1p_mse (from v2.1)
                    niche_loss = torch.zeros((), device=self.device)
                    if cfg.lambda_niche > 0 and coords.shape[0] >= 2 and cfg.niche_loss_type != "none":
                        tid_arg = time_ids if self.model.has_temporal else None
                        ai, ni = build_batch_knn_niches_same_time(
                            coords, time_ids=tid_arg, k=cfg.niche_k,
                            n_anchors=cfg.niche_anchors, seed=epoch*100000+bi)
                        if ni.numel() > 0:
                            niche_loss = self._consistency_loss(
                                mu[ai], counts[ni].mean(1), cfg.niche_loss_type)
                    # niche_loss = torch.zeros((), device=self.device)
                    # if cfg.lambda_niche > 0 and coords.shape[0] >= 2:
                    #     tid_arg = time_ids if self.model.has_temporal else None

                    #     grad_loss = self._spatial_gradient_loss(mu, counts, coords, tid_arg, k=cfg.niche_k)

                    #     ai, ni = build_batch_knn_niches_same_time(
                    #         coords, time_ids=tid_arg, k=cfg.niche_k,
                    #         n_anchors=cfg.niche_anchors, seed=epoch*100000+bi
                    #     )
                    #     smooth_loss = torch.zeros((), device=self.device)
                    #     if ni.numel() > 0:
                    #         smooth_loss = self._consistency_loss(mu[ai], counts[ni].mean(1), cfg.niche_loss_type)

                    #     niche_loss = 0.5 * grad_loss + 0.5 * smooth_loss

                    # Temporal loss: log1p_mse (from v2.1)
                    temp_loss = torch.zeros((), device=self.device)
                    if (self.model.has_temporal and cfg.lambda_temporal > 0
                            and coords.shape[0] >= 2 and cfg.temporal_loss_type != "none"):
                        ait, nit = build_temporal_niches(
                            coords, time_ids=time_ids, k=cfg.temporal_k,
                            max_delta=cfg.temporal_dt, n_anchors=cfg.niche_anchors,
                            seed=epoch*100000+bi+1)
                        if nit is not None and nit.numel() > 0:
                            temp_loss = self._consistency_loss(
                                mu[ait], counts[nit].mean(1), cfg.temporal_loss_type)

                    ce = (F.cross_entropy(out["cell_type_logits"], ct,
                                          weight=self.class_weights,
                                          ignore_index=cfg.celltype_ignore_index,
                                          label_smoothing=cfg.label_smoothing)
                          if cfg.lambda_celltype > 0
                          else torch.zeros((), device=self.device))

                    with torch.no_grad():
                        v = ct != cfg.celltype_ignore_index
                        if v.sum() > 0:
                            correct += (out["cell_type_logits"][v].argmax(1) == ct[v]).sum().item()
                            total += v.sum().item()

                    niche_ct = (self._soft_niche_ct(coords, out["cell_type_logits"], ct, time_ids)
                                if cfg.lambda_niche_ct > 0
                                else torch.zeros((), device=self.device))

                    # Gene var in log1p space (from v2.1)
                    gv = torch.zeros((), device=self.device)
                    if cfg.lambda_gene_var > 0 and counts.shape[0] >= 16:
                        gv = F.mse_loss(torch.var(torch.log1p(mu), dim=0),
                                        torch.var(torch.log1p(counts), dim=0))
                    # gv = torch.zeros((), device=self.device)
                    # if cfg.lambda_gene_var > 0 and counts.shape[0] >= 64:
                    #     gv = self._gene_moment_loss(mu, counts, sf)

                    loss = (nll_spot
                            + cfg.lambda_niche * niche_loss
                            + cfg.lambda_temporal * temp_loss
                            + cfg.lambda_celltype * ce
                            + cfg.lambda_niche_ct * niche_ct
                            + cfg.lambda_gene_var * gv)

                if not torch.isfinite(loss): skipped += 1; continue
                if train:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    if cfg.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

            for k, v in [("loss",loss),("nll_spot",nll_spot),("nll_spatial",niche_loss),
                         ("nll_temporal",temp_loss),("ce",ce),("niche_ct",niche_ct),("gene_var",gv)]:
                stats[k].append(float(v.detach().cpu()))

        if train: self.scheduler.step()
        if not stats["loss"]:
            return {k: float("nan") for k in stats} | {"lr":0,"skipped":0,"acc":0}
        return {**{k: float(np.mean(v)) for k, v in stats.items()},
                "lr": float(self.optimizer.param_groups[0]["lr"]),
                "skipped": float(skipped), "acc": correct / max(total, 1)}

    def train_epoch(self, loader, epoch):
        return self._run_epoch(loader, epoch, train=True)

    @torch.no_grad()
    def validate_epoch(self, loader, epoch):
        return self._run_epoch(loader, epoch, train=False)

    def train(self, train_loader, val_loader=None, n_epochs=100, print_every=10):
        cfg = self.config
        history = []
        best_metric, best_state, bad_epochs = float("inf"), None, 0

        print(f"\n{'='*72}\nHyperSpatial v7.1 Training\n{'='*72}")
        print(f"{'4D' if self.model.has_temporal else '3D'} | {self.device} | "
              f"niche={cfg.niche_loss_type}:{cfg.lambda_niche} | "
              f"temp={cfg.temporal_loss_type}:{cfg.lambda_temporal}")
        if val_loader: print(f"Early stopping: patience={cfg.early_stopping_patience}")

        for ep in range(1, n_epochs + 1):
            tr = self.train_epoch(train_loader, ep)
            rec = {"epoch": ep, "train": tr}

            if val_loader is not None:
                va = self.validate_epoch(val_loader, ep)
                rec["val"] = va
                metric = va["loss"]
                if np.isfinite(metric) and metric < (best_metric - cfg.early_stopping_min_delta):
                    best_metric = metric; bad_epochs = 0
                    if cfg.restore_best_weights:
                        best_state = {k: v.detach().cpu().clone()
                                      for k, v in self.model.state_dict().items()}
                else:
                    bad_epochs += 1

            history.append(rec)
            if ep % print_every == 0:
                msg = f"  {ep:03d}/{n_epochs} | train={tr['loss']:.4f}"
                if val_loader: msg += f" val={rec['val']['loss']:.4f}"
                msg += f" acc={tr['acc']:.3f} lr={tr['lr']:.1e}"
                print(msg)

            if val_loader and bad_epochs >= cfg.early_stopping_patience:
                print(f"  Early stopping at epoch {ep}")
                break

        if best_state is not None and cfg.restore_best_weights:
            self.model.load_state_dict(best_state)
            print(f"  Restored best weights (val_loss={best_metric:.4f})")

        print(f"{'='*72}\n✓ Done\n{'='*72}\n")
        return history


# ══════════════════════════════════════════════════════════════════════════════
# Train/val split helper (from v2.1)
# ══════════════════════════════════════════════════════════════════════════════

def make_train_val_loaders(dataset, batch_size, seed=42, val_fraction=0.1,
                            num_workers=0, use_balanced_time=True):
    n = len(dataset)
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_val = max(1, int(round(n * val_fraction))) if n > 10 else 0
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    train_ds = Subset(dataset, train_idx.tolist())
    val_ds = Subset(dataset, val_idx.tolist()) if n_val > 0 else None
    pw = num_workers > 0

    if (use_balanced_time and dataset.time_ids is not None
            and len(np.unique(dataset.time_ids[train_idx])) >= 2):
        sampler = CyclicBalancedTimeBatchSampler(
            dataset.time_ids[train_idx], batch_size, seed)
        train_loader = DataLoader(train_ds, batch_sampler=sampler,
                                  num_workers=num_workers, pin_memory=True,
                                  persistent_workers=pw)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True,
                                  persistent_workers=pw)

    val_loader = (DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True,
                             persistent_workers=pw)
                  if val_ds else None)
    return train_loader, val_loader, train_idx, val_idx


# ══════════════════════════════════════════════════════════════════════════════
# Prediction
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict(model, coords, size_factors=None, batch_size=4096,
            device="cuda", return_theta=False):
    model.eval()
    ct = torch.from_numpy(coords.astype(np.float32))
    sf = None if size_factors is None else torch.from_numpy(size_factors.astype(np.float32))
    N = ct.shape[0]; mu_l, p_l = [], []; th_out = None
    for i in range(0, N, batch_size):
        b = ct[i:i+batch_size].to(device)
        o = model(b)
        mu = o["mu"].float()
        if sf is not None:
            mu = mu * sf[i:i+batch_size].to(device).float().unsqueeze(1)
        mu_l.append(mu.cpu().numpy())
        p_l.append(F.softmax(o["cell_type_logits"], -1).float().cpu().numpy())
        if return_theta and th_out is None:
            th = o["theta"]
            th_out = (np.array([float(th.cpu())]) if th.dim() == 0
                      else th.cpu().numpy().astype(np.float32))
    return np.concatenate(mu_l), np.concatenate(p_l), th_out


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_metrics(counts_true, mu_pred, gene_names=None):
    from scipy.stats import pearsonr
    ct = _to_dense_float32(counts_true); mp = mu_pred.astype(np.float32)
    def _r(a, b):
        return float("nan") if np.std(a)<1e-12 or np.std(b)<1e-12 else float(pearsonr(a,b)[0])
    gm_t, gm_p = ct.mean(0), mp.mean(0)
    gv_t, gv_p = ct.var(0), mp.var(0)
    pgr = [_r(ct[:,g], mp[:,g]) for g in range(ct.shape[1])]
    pgr = np.array([r for r in pgr if np.isfinite(r)])
    psr = [_r(ct[i,:], mp[i,:]) for i in range(ct.shape[0])]
    psr = np.array([r for r in psr if np.isfinite(r)])
    return {
        "gene_mean_pearson": _r(gm_t, gm_p),
        "gene_var_pearson": _r(gv_t, gv_p),
        "lib_size_pearson": _r(ct.sum(1), mp.sum(1)),
        "per_gene_pearson_median": float(np.median(pgr)) if len(pgr) else float("nan"),
        "per_spot_pearson_median": float(np.median(psr)) if len(psr) else float("nan"),
        "rmse_log": float(np.sqrt(np.mean((np.log1p(ct) - np.log1p(mp))**2))),
        "zero_frac_true": float((ct==0).mean()),
        "zero_frac_pred": float((mp<0.5).mean()),
    }

def evaluate_celltype_metrics(true_labels, pred_probs, ignore_index=-1):
    from sklearn.metrics import accuracy_score, f1_score
    v = true_labels != ignore_index
    if v.sum() == 0: return {"ct_acc": float("nan"), "ct_f1": float("nan")}
    y, yp = true_labels[v], pred_probs[v].argmax(1)
    return {"ct_acc": float(accuracy_score(y, yp)),
            "ct_f1": float(f1_score(y, yp, average="macro", zero_division=0))}

def compute_morans_i(values, coords, k=15):
    from scipy.spatial import cKDTree
    xyz = coords[:, :3] if coords.shape[1] >= 3 else coords
    N = values.shape[0]
    if N < 10: return float("nan")
    tree = cKDTree(xyz)
    _, idx = tree.query(xyz, k=min(k+1, N)); idx = idx[:, 1:]
    d = values - values.mean(); denom = np.sum(d**2)
    if denom < 1e-12: return float("nan")
    numer = sum(d[i] * d[idx[i][idx[i] != i]].sum() for i in range(N))
    w_sum = sum((idx[i] != i).sum() for i in range(N))
    return float((N / w_sum) * (numer / denom)) if w_sum > 0 else float("nan")

def evaluate_spatial_metrics(counts_true, mu_pred, coords, n_top=50, k=15):
    from scipy.stats import pearsonr
    ct = _to_dense_float32(counts_true); mp = mu_pred.astype(np.float32)
    top = np.argsort(ct.var(0))[-min(n_top, ct.shape[1]):]
    mt, ms = [], []
    for g in top:
        a, b = compute_morans_i(ct[:,g], coords, k), compute_morans_i(mp[:,g], coords, k)
        if np.isfinite(a) and np.isfinite(b): mt.append(a); ms.append(b)
    mt, ms = np.array(mt), np.array(ms)
    r = float(pearsonr(mt, ms)[0]) if len(mt) > 2 else float("nan")
    return {"morans_i_pearson": r,
            "morans_i_true_mean": float(mt.mean()) if len(mt) else float("nan"),
            "morans_i_pred_mean": float(ms.mean()) if len(ms) else float("nan"),
            "n_genes_morans": len(mt)}

def compute_class_counts(ds, n_classes, ignore=-1):
    if ds.ct is None: return np.ones(n_classes)
    return np.array([np.sum(ds.ct == c) for c in range(n_classes)], dtype=np.float64)