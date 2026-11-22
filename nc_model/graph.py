"""Unlabelled protein graph and ordered herb/symptom incidence tables."""
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch


@dataclass
class Incidence:
    node: torch.Tensor
    edge: torch.Tensor
    num_nodes: int
    num_edges: int
    node_degree: torch.Tensor
    edge_degree: torch.Tensor

    @classmethod
    def from_members(cls, members, n, device):
        lengths = []
        indices = []
        for row in members:
            row = np.asarray(row)
            if row.ndim != 1 or (row.size and row.dtype.kind not in "iu"):
                raise ValueError("membership must be a sequence of integer index lists")
            if row.size and (row.min() < 0 or row.max() >= n or len(np.unique(row)) != len(row)):
                raise ValueError("invalid or repeated protein within an entity")
            lengths.append(len(row))
            indices.append(row.astype(np.int64, copy=False))
        nodes = np.concatenate(indices) if indices else np.empty(0, dtype=np.int64)
        node = torch.as_tensor(nodes, device=device)
        edge = torch.as_tensor(np.repeat(np.arange(len(lengths), dtype=np.int64), lengths), device=device)
        return cls(node, edge, n, len(lengths), torch.bincount(node, minlength=n),
                   torch.bincount(edge, minlength=len(lengths)))


@dataclass
class Graph:
    x: torch.Tensor
    ppi_plain: torch.Tensor
    h: Incidence
    s: Incidence


def graph_from_data(data, device="cpu"):
    """Accept .x, .ppi_edges, .herb_members and .symptom_members in aligned ID order."""
    x = torch.as_tensor(data.x, dtype=torch.float32, device=device)
    if x.ndim != 2 or min(x.shape) < 1 or not torch.isfinite(x).all():
        raise ValueError("protein features must be a finite nonempty matrix")
    n = len(x)
    edges = np.asarray(data.ppi_edges)
    if edges.ndim != 2 or edges.shape[1] != 2 or (edges.size and edges.dtype.kind not in "iu"):
        raise ValueError("PPI must be an integer [E,2] endpoint array")
    if edges.size and (edges.min() < 0 or edges.max() >= n):
        raise ValueError("PPI endpoint outside protein index")
    edges = torch.as_tensor(edges, dtype=torch.long, device=device)
    edges = edges[edges[:, 0] != edges[:, 1]]
    both = torch.cat([edges, edges.flip(1)], dim=0).T
    adjacency = torch.sparse_coo_tensor(both, torch.ones(both.shape[1], device=device), (n, n)).coalesce()
    plain = torch.sparse_coo_tensor(adjacency.indices(), torch.ones_like(adjacency.values()), (n, n)).coalesce()
    return Graph(x, plain, Incidence.from_members(data.herb_members, n, device),
                 Incidence.from_members(data.symptom_members, n, device))


def _members(source, prefix):
    index, offsets = source[prefix + "_index"], source[prefix + "_offsets"]
    if index.ndim != 1 or offsets.ndim != 1 or offsets.dtype.kind not in "iu" or not len(offsets):
        raise ValueError("invalid membership arrays for " + prefix)
    if offsets[0] != 0 or offsets[-1] != len(index) or np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("invalid membership offsets for " + prefix)
    return [index[start:end] for start, end in zip(offsets[:-1], offsets[1:])]
