"""Read the real PPI graph, entity memberships and frozen outer folds."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .graph import _members, graph_from_data


def verify_files(root, dataset):
    manifest = json.loads((root / "sha256.json").read_text(encoding="utf8"))
    used = {k: v for k, v in manifest.items() if k.split("/")[0] in ("shared", dataset)}
    for rel, expected in used.items():
        if hashlib.sha256((root / rel).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Dataset checksum mismatch: {rel}")
    return hashlib.sha256(json.dumps(used, sort_keys=True).encode()).hexdigest()


def load_graph(root, dataset, device):
    shared = root / "shared"
    with np.load(root / dataset / "members.npz", allow_pickle=False) as members:
        data = SimpleNamespace(
            x=np.load(shared / "features.npy", allow_pickle=False),
            ppi_edges=np.loadtxt(shared / "RealPPI.tsv", dtype=np.int64, ndmin=2),
            herb_members=_members(members, "h"),
            symptom_members=_members(members, "s"),
        )
    return graph_from_data(data, device)


def load_fold(root, dataset, fold, device):
    with np.load(root / dataset / f"fold_{fold:02d}.npz", allow_pickle=False) as arrays:
        return {k: torch.as_tensor(arrays[k], dtype=torch.long, device=device)
                for k in ("train_pos", "train_neg", "test_pos", "test_neg")}
