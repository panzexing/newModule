"""R-NoContext (historical A4), with shared H/P/S propagation and local/final readout."""
from dataclasses import dataclass
import hashlib
import math

import torch
from torch import nn

from .ops import masked_softmax, segment_mean, segment_softmax, segment_sum


def relation_valid(graph):
    receivers = graph.ppi_plain.indices()[0]
    ppi_valid = torch.bincount(receivers, minlength=len(graph.x)) > 0
    return torch.stack((graph.h.node_degree > 0, ppi_valid, graph.s.node_degree > 0), dim=1)


class HypergraphAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.q_back = nn.Linear(dim, dim, bias=False)
        self.k_back = nn.Linear(dim, dim, bias=False)
        self.edge_value = nn.Linear(dim, dim, bias=False)

    def pool(self, z, incidence):
        n, e = incidence.node, incidence.edge
        means = segment_mean(z[n], e, incidence.num_edges)
        scores = (self.q(means)[e] * self.k(z)[n]).sum(-1) / math.sqrt(self.dim)
        alpha = segment_softmax(scores, e, incidence.num_edges)
        edges = segment_sum(alpha[:, None] * self.value(z)[n], e, incidence.num_edges)
        return edges, alpha

    def forward(self, z, incidence):
        edges, _ = self.pool(z, incidence)
        n, e = incidence.node, incidence.edge
        scores = (self.q_back(z)[n] * self.k_back(edges)[e]).sum(-1) / math.sqrt(self.dim)
        beta = segment_softmax(scores, n, incidence.num_nodes)
        message = segment_sum(beta[:, None] * self.edge_value(edges)[e], n, incidence.num_nodes)
        return message, edges


class NeighborAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)

    def forward(self, z, adjacency):
        receiver, sender = adjacency.indices()
        scores = (self.q(z)[receiver] * self.k(z)[sender]).sum(-1) / math.sqrt(self.dim)
        alpha = segment_softmax(scores, receiver, len(z))
        message = segment_sum(alpha[:, None] * self.value(z)[sender], receiver, len(z))
        return message


class RelationRound(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.dim = dim
        self.h = HypergraphAttention(dim)
        self.ppi = NeighborAttention(dim)
        self.s = HypergraphAttention(dim)
        self.local_norm = nn.ModuleList(nn.LayerNorm(dim) for _ in range(3))
        self.dropout = nn.Dropout(dropout)
        self.cross_value = nn.ModuleDict({f"{r}{s}": nn.Linear(dim, dim, bias=False)
                                         for r in range(3) for s in range(3) if r != s})
        self.gate = nn.ModuleList(nn.Linear(2 * dim, dim) for _ in range(3))
        self.cross_norm = nn.ModuleList(nn.LayerNorm(dim) for _ in range(3))
        self.q = nn.ModuleList(nn.Linear(dim, dim, bias=False) for _ in range(3))
        self.k = nn.ModuleList(nn.Linear(dim, dim, bias=False) for _ in range(3))
        self.relation_bias = nn.Parameter(torch.zeros(6))
        # Registered in the historical order for exact A4 initialization and strict
        # checkpoint loading. The R-NoContext score never reads these tensors.
        self.context_q = nn.ModuleList(nn.Linear(dim, dim, bias=False) for _ in range(3))
        self.context_k = nn.ModuleList(nn.Linear(dim, dim, bias=False) for _ in range(3))

    def cross_mask(self, valid):
        allowed = valid[:, :, None] & valid[:, None, :]
        return allowed & ~torch.eye(3, dtype=torch.bool, device=valid.device)[None]

    def ppi_step(self, z, adjacency):
        return self.ppi(z, adjacency), None

    def neighborhood_context(self, z, graph, ppi_context):
        return None

    def content_and_bias(self, z, messages):
        query = torch.stack([self.q[r](z[:, r]) for r in range(3)], 1)
        key = torch.stack([self.k[s](messages[:, s]) for s in range(3)], 1)
        content = torch.einsum("nrd,nsd->nrs", query, key) / math.sqrt(self.dim)
        offdiag = ~torch.eye(3, dtype=torch.bool, device=z.device)
        bias = z.new_zeros((3, 3)).masked_scatter(offdiag, self.relation_bias)
        return content, bias

    def cross_scores(self, z, messages, contexts):
        content, bias = self.content_and_bias(z, messages)
        return content + bias

    def cross_weights(self, z, messages, contexts, mask):
        return masked_softmax(self.cross_scores(z, messages, contexts), mask)

    def directional_message(self, r, messages, weights):
        return sum(weights[:, r, s, None] * self.cross_value[f"{r}{s}"](messages[:, s])
                   for s in range(3) if s != r)

    def receive(self, r, u, messages, weights):
        cross = self.directional_message(r, messages, weights)
        gate = torch.sigmoid(self.gate[r](torch.cat((u, cross), -1)))
        return self.cross_norm[r](u + self.dropout(gate * cross))

    def forward(self, z, graph, valid):
        mh, herbs = self.h(z[:, 0], graph.h)
        mp, ppi_context = self.ppi_step(z[:, 1], graph.ppi_plain)
        ms, symptoms = self.s(z[:, 2], graph.s)
        messages = torch.stack((mh, mp, ms), 1)
        contexts = self.neighborhood_context(z, graph, ppi_context)
        weights = self.cross_weights(z, messages, contexts, self.cross_mask(valid))
        outputs = []
        for r in range(3):
            u = self.local_norm[r](z[:, r] + self.dropout(messages[:, r]))
            out = self.receive(r, u, messages, weights)
            outputs.append(out * valid[:, r, None])
        return torch.stack(outputs, 1), herbs, symptoms


@dataclass
class EntityState:
    z: torch.Tensor
    herbs: torch.Tensor
    symptoms: torch.Tensor
    local_herbs: torch.Tensor
    local_symptoms: torch.Tensor


class Encoder(nn.Module):
    def __init__(self, input_dim, dim, rounds, dropout):
        super().__init__()
        self.num_layers = rounds
        self.input = nn.ModuleList(nn.Linear(input_dim, dim, bias=False) for _ in range(3))
        self.layers = nn.ModuleList([RelationRound(dim, dropout)])

    def forward(self, graph):
        valid = relation_valid(graph)
        z = torch.stack([projection(graph.x) for projection in self.input], 1) * valid[:, :, None]
        layer = self.layers[0]
        z, local_herbs, local_symptoms = layer(z, graph, valid)
        for _ in range(1, self.num_layers):
            z, _, _ = layer(z, graph, valid)
        herbs, _ = layer.h.pool(z[:, 0], graph.h)
        symptoms, _ = layer.s.pool(z[:, 2], graph.s)
        return EntityState(z, herbs, symptoms, local_herbs, local_symptoms)


def _pair_head(dim):
    return nn.Sequential(nn.Linear(4 * dim, 64), nn.ReLU(), nn.Linear(64, 1))


class Readout(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.head = _pair_head(dim)
        self.fusion_h = nn.Linear(2 * dim, dim)
        self.fusion_s = nn.Linear(2 * dim, dim)
        with torch.no_grad():
            eye = torch.eye(dim)
            for fusion in (self.fusion_h, self.fusion_s):
                nn.init.normal_(fusion.weight[:, :dim], std=.001 / math.sqrt(dim))
                fusion.weight[:, :dim].add_(.1 * eye)
                fusion.weight[:, dim:].copy_(eye)
                fusion.bias.zero_()

    def local_slot(self, local, final):
        return local

    def forward(self, state, pairs):
        if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype != torch.long:
            raise ValueError("pairs must be int64 [batch,2]")
        if pairs.numel() and (pairs.min() < 0 or pairs[:, 0].max() >= len(state.herbs)
                             or pairs[:, 1].max() >= len(state.symptoms)):
            raise IndexError("pair entity index outside graph")
        h_cross, s_cross = state.herbs[pairs[:, 0]], state.symptoms[pairs[:, 1]]
        h_slot = self.local_slot(state.local_herbs, state.herbs)[pairs[:, 0]]
        s_slot = self.local_slot(state.local_symptoms, state.symptoms)[pairs[:, 1]]
        h = self.fusion_h(torch.cat((h_slot, h_cross), -1))
        s = self.fusion_s(torch.cat((s_slot, s_cross), -1))
        return self.head(torch.cat((h, s, h * s, (h - s).abs()), -1)).squeeze(-1)


class RNoContext(nn.Module):
    ablation_name = "R-NoContext"

    def __init__(self, config, input_dim):
        super().__init__()
        dim = int(config.get("hidden_dim", 64))
        rounds = int(config.get("num_layers", 2))
        dropout = float(config.get("dropout", .2))
        if min(input_dim, rounds, dim) < 1 or not 0 <= dropout < 1:
            raise ValueError("positive dimensions and rounds, and dropout in [0,1) required")
        if not config.get("share_layers", True) or int(config.get("head_hidden", 64)) != 64:
            raise ValueError("R-NoContext requires shared propagation weights and pair-head width 64")
        seed = config.get("seed", 0)
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        # V12 advances the caller's CPU generator for the shared encoder/head.
        # The fusion weights come from a second, isolated constructor sequence.
        reference_encoder = Encoder(input_dim, dim, rounds, dropout)
        reference_head = _pair_head(dim)
        key = f"reference_a_entity_v1:entity_readout_v6:{seed}".encode("ascii")
        derived = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2**63 - 1)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(derived)
            self.encoder = Encoder(input_dim, dim, rounds, dropout)
            self.readout = Readout(dim)
        self.encoder.load_state_dict(reference_encoder.state_dict())
        self.readout.head.load_state_dict(reference_head.state_dict())

    def encode(self, graph):
        return self.encoder(graph)

    def decode(self, state, pairs):
        return self.readout(state, pairs)

    def forward(self, graph, pairs):
        return self.decode(self.encode(graph), pairs)


def build_model(config, input_dim):
    return RNoContext(config, input_dim)
