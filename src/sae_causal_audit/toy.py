"""Reference toy-regime implementations of the audit interfaces.

These reproduce the originating article's setting — the Elhage et al.
(2022) toy model and a TopK SAE — as a *known-ground-truth harness* for
the audit itself. They serve three purposes:

1. **Executable documentation**: the smallest complete example of
   satisfying :class:`~sae_causal_audit.interfaces.SparseAutoencoder`,
   ``DownstreamFn`` and ``FeatureProbe``.
2. **Integration-test fixture**: the test suite trains these end-to-end
   and asserts the audit's qualitative findings (good SAE → high
   specificity; an atom that never fires → flagged inert), so the
   pipeline is validated against a setting where the right answer is
   known — the same epistemic move as the original piece.
3. **Calibration**: users can sanity-check a modified pipeline here
   before spending GPU-hours on a production SAE.

Deliberately compact; production knobs (aux losses for dead atoms,
schedulers, checkpointing) belong to real training code, not a fixture.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class ToyConfig:
    n_features: int = 32
    n_hidden: int = 8
    sparsity: float = 0.95
    importance_decay: float = 0.9
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_hidden <= 0 or self.n_features <= 0:
            raise ValueError("dimensions must be positive")
        if not (0.0 <= self.sparsity < 1.0):
            raise ValueError("sparsity must be in [0, 1)")


class ToyModel(nn.Module):
    """``h = W x``; ``x_hat = ReLU(Wᵀ h + b)`` — the bottleneck model."""

    def __init__(self, cfg: ToyConfig) -> None:
        super().__init__()
        self.cfg = cfg
        gen = torch.Generator().manual_seed(cfg.seed)
        self.W = nn.Parameter(
            torch.randn(cfg.n_hidden, cfg.n_features, generator=gen) * 0.1
        )
        self.b = nn.Parameter(torch.zeros(cfg.n_features))

    def sample_batch(self, n: int, generator: torch.Generator) -> torch.Tensor:
        x = torch.rand(n, self.cfg.n_features, generator=generator)
        mask = torch.rand(n, self.cfg.n_features, generator=generator) >= self.cfg.sparsity
        return x * mask

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W.T

    def output(self, h: torch.Tensor) -> torch.Tensor:
        return torch.relu(h @ self.W + self.b)

    def importance(self) -> torch.Tensor:
        i = torch.arange(self.cfg.n_features, dtype=torch.float32)
        return self.cfg.importance_decay**i


def train_toy_model(cfg: ToyConfig, steps: int = 4000, batch: int = 1024) -> ToyModel:
    """Train the bottleneck model on importance-weighted reconstruction."""
    model = ToyModel(cfg)
    gen = torch.Generator().manual_seed(cfg.seed + 1)
    imp = model.importance()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=False)
    for _ in range(steps):
        x = model.sample_batch(batch, gen)
        x_hat = model.output(model.hidden(x))
        loss = (imp * (x - x_hat) ** 2).sum(dim=-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model.eval()


class TopKSAE(nn.Module):
    """Minimal TopK SAE satisfying the ``SparseAutoencoder`` protocol."""

    def __init__(self, d_in: int, d_sae: int, k: int, seed: int = 0) -> None:
        super().__init__()
        if not (1 <= k <= d_sae):
            raise ValueError(f"k must be in [1, d_sae]; got k={k}, d_sae={d_sae}")
        self.k = k
        gen = torch.Generator().manual_seed(seed)
        self.W_enc = nn.Parameter(torch.randn(d_in, d_sae, generator=gen) * 0.1)
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.randn(d_sae, d_in, generator=gen) * 0.1)
        self.b_dec = nn.Parameter(torch.zeros(d_in))

    def encode(self, h: torch.Tensor) -> torch.Tensor:
        pre = (h - self.b_dec) @ self.W_enc + self.b_enc
        vals, idx = torch.topk(pre, k=self.k, dim=-1)
        f = torch.zeros_like(pre)
        f.scatter_(-1, idx, torch.relu(vals))
        return f

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.W_dec + self.b_dec

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(h))


def train_topk_sae(
    model: ToyModel, d_sae: int, k: int, steps: int = 4000, batch: int = 1024, seed: int = 0
) -> TopKSAE:
    """Train a TopK SAE on the toy model's hidden activations."""
    sae = TopKSAE(model.cfg.n_hidden, d_sae, k, seed=seed)
    gen = torch.Generator().manual_seed(seed + 2)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3, foreach=False)
    for _ in range(steps):
        with torch.no_grad():
            h = model.hidden(model.sample_batch(batch, gen))
        h_hat = sae(h)
        loss = ((h - h_hat) ** 2).sum(dim=-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return sae.eval()


class ToyProbe:
    """FeatureProbe for the toy regime: single-feature-ON isolation inputs."""

    def __init__(self, model: ToyModel, seed: int = 0, magnitude: float = 1.0) -> None:
        self._model = model
        self._gen = torch.Generator().manual_seed(seed + 3)
        self._magnitude = magnitude

    def activations_with_feature(self, feature_idx: int, n_samples: int) -> torch.Tensor:
        x = torch.zeros(n_samples, self._model.cfg.n_features)
        x[:, feature_idx] = self._magnitude * torch.rand(
            n_samples, generator=self._gen
        ).clamp_min(0.5)
        with torch.no_grad():
            return self._model.hidden(x)

    def activations_without_feature(self, feature_idx: int, n_samples: int) -> torch.Tensor:
        with torch.no_grad():
            x = self._model.sample_batch(n_samples, self._gen)
            x[:, feature_idx] = 0.0
            return self._model.hidden(x)


def true_directions(model: ToyModel) -> torch.Tensor:
    """Ground-truth feature directions in activation space: columns of W."""
    return model.W.detach().T.clone()


def well_represented_mask(model: ToyModel, norm_threshold: float = 0.1) -> torch.Tensor:
    """Features the toy model actually represents (‖W_i‖² >= threshold).

    Mirrors the article's distinction: dropped features (norm ≈ 0) are
    excluded from recovery/causal claims — an SAE cannot recover what the
    model never encoded.
    """
    return (model.W.detach() ** 2).sum(dim=0) >= norm_threshold
