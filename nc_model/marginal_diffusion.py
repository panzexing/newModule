"""Residual full-catalog ranking and a drop-in diffusion PPI branch for R-NoContext."""
import numpy as np
import torch
from torch import nn


class PPIDiffusion(nn.Module):
    """Mix H0..H3 using trainable softmax weights; each update restarts at H0 with probability 0.15."""
    def __init__(self):
        super().__init__()
        self.hop_logits = nn.Parameter(torch.zeros(4))
        self._source = None
        self._normalized = None

    def forward(self, z, adjacency):
        if adjacency is not self._source:
            pairs = adjacency.coalesce().indices()
            pairs = pairs[:, pairs[0] != pairs[1]]
            undirected = torch.cat((pairs, pairs.flip(0)), dim=1)
            binary = torch.sparse_coo_tensor(undirected, z.new_ones(undirected.shape[1]),
                                             adjacency.shape).coalesce()
            receiver, sender = binary.indices()
            inv = torch.bincount(receiver, minlength=len(z)).to(z.dtype).clamp_min(1).rsqrt()
            self._normalized = torch.sparse_coo_tensor(binary.indices(), inv[receiver]*inv[sender],
                                                       adjacency.shape, is_coalesced=True)
            self._source = adjacency
        weights = self.hop_logits.softmax(0)
        h, message = z, weights[0]*z
        for weight in weights[1:]:
            h = .15*z + .85*torch.sparse.mm(self._normalized, h)
            message = message + weight*h
        return message


def build_model(config, input_dim, *, diffusion=False):
    from nc_model.model import RNoContext

    model = RNoContext(config, input_dim)
    if diffusion:
        model.encoder.layers[0].ppi = PPIDiffusion()
    return model


def ranking_loss(logits, positives, prior):
    """Full-catalog log-softmax of (f + tau*b)/T with fixed tau=T=1.

    b is log(1 + the herb's training positive count). Average negative
    log-probabilities over each symptom's known positives, then over symptoms.
    """
    if logits.ndim != 2 or logits.shape != positives.shape or prior.shape != logits.shape[:1]:
        raise ValueError("Expected herb-by-symptom logits/labels and one prior per herb")
    if positives.dtype != torch.bool or logits.device != positives.device or prior.device != logits.device:
        raise ValueError("Expected Boolean labels and all inputs on the same device")
    counts = positives.sum(0)
    if (counts == 0).any():
        raise ValueError("Each training symptom must have a positive herb")
    log_prob = torch.log_softmax(logits + prior[:, None], dim=0)
    return -(log_prob.masked_fill(~positives, 0).sum(0) / counts).mean()


def combine(scores, prior, coefficient, mode):
    """Combine finite full-catalog columns along the herb axis, including random replicates."""
    scores, prior = np.asarray(scores, dtype=np.float64), np.asarray(prior, dtype=np.float64)
    if scores.ndim not in (2, 3) or prior.shape != (scores.shape[-2],):
        raise ValueError("Expected [herb,symptom] or [repeat,herb,symptom] and one prior per herb")
    if mode == "calibrated":
        return scores + coefficient*prior[:, None]
    if mode != "original":
        raise ValueError(f"Unknown combination mode {mode}")
    std = scores.std(axis=-2, keepdims=True)
    centered = scores - scores.mean(axis=-2, keepdims=True)
    normalized = np.divide(centered, std, out=np.zeros_like(centered), where=std > 0)
    return prior[:, None] + coefficient*normalized
