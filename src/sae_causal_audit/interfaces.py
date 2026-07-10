"""Structural interfaces for the causal audit pipeline.

The audit is deliberately decoupled from any specific SAE implementation.
Anything that satisfies :class:`SparseAutoencoder` — a toy SAE, an
``sae_lens.SAE``, a hand-rolled TopK module — can be audited without
inheritance or registration. The same applies to the downstream model:
the audit only needs a callable mapping reconstructed activations to a
behavioral readout (toy-model outputs, logits, a loss — anything whose
per-dimension change is meaningful to compare).

Design notes
------------
* ``Protocol`` (structural typing) rather than ABCs: zero coupling, and
  third-party objects qualify retroactively.
* Everything operates on ``torch.Tensor`` batches of shape
  ``(batch, d_in)`` for activations and ``(batch, d_sae)`` for codes.
* No implicit device movement: tensors are used on whatever device they
  arrive on. Callers own device placement (keeps the library honest on
  both CPU toy runs and GPU production runs).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class SparseAutoencoder(Protocol):
    """Minimal structural contract an SAE must satisfy to be audited.

    Required:

    * ``encode(h) -> f``: activations ``(batch, d_in)`` to sparse codes
      ``(batch, d_sae)``. Post-nonlinearity, post-TopK — i.e. the code
      the SAE would actually use to reconstruct.
    * ``decode(f) -> h_hat``: codes back to activations ``(batch, d_in)``.
    * ``W_dec``: decoder dictionary of shape ``(d_sae, d_in)`` (rows are
      atoms). Used only for correlational (cosine) matching; all causal
      metrics go through ``encode``/``decode`` so that encoder-side
      behavior — the thing decoder geometry cannot certify — is what is
      actually measured.
    """

    W_dec: torch.Tensor

    def encode(self, h: torch.Tensor) -> torch.Tensor: ...

    def decode(self, f: torch.Tensor) -> torch.Tensor: ...


class DownstreamFn(Protocol):
    """Maps reconstructed activations to a behavioral readout.

    ``(batch, d_in) -> (batch, d_out)``. For the toy setting this is the
    toy model's output stage ``ReLU(h_hat @ W + b)``; for a real
    transformer it is typically "splice ``h_hat`` into the residual
    stream and return logits (or a logit diff) at the hook point".

    The audit compares readouts before/after an intervention, so the
    only hard requirement is that the readout is deterministic given its
    input (turn off dropout / sampling) and differentiable-free (the
    audit never backpropagates through it).
    """

    def __call__(self, h_hat: torch.Tensor) -> torch.Tensor: ...


class FeatureProbe(Protocol):
    """Supplies inputs where a given ground-truth/concept feature is ON or OFF.

    This is the abstraction that lets the identical audit run in two
    regimes:

    * **Toy regime** (ground truth known): "feature i active in
      isolation" — synthetic inputs with exactly one nonzero entry.
    * **Real-model regime** (no ground truth): positive/negative probe
      datasets for a labeled concept (e.g. Neuronpedia-style examples),
      where "ON" means "the concept is present in the text".

    Both methods return **activations at the SAE's hook point**, shape
    ``(n_samples, d_in)``.
    """

    def activations_with_feature(
        self, feature_idx: int, n_samples: int
    ) -> torch.Tensor: ...

    def activations_without_feature(
        self, feature_idx: int, n_samples: int
    ) -> torch.Tensor: ...
