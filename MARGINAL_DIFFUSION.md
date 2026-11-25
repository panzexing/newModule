# Marginal-adjusted ranking for unseen symptoms

The model factory in `nc_model/marginal_diffusion.py` builds R-NoContext with an optional replacement for its PPI attention operation. A shared bank of four softmax weights mixes the original protein state and three restart-diffusion iterates. Propagation uses the binary, symmetric, loop-free PPI operator with symmetric degree normalization and restart probability 0.15. The herb/symptom hypergraphs, directed cross-view updates and original readout are shared with the repository encoder.

For a compact training graph, `positive_mask[h, s]` is true for a training association. Training symptoms must have at least one positive. The full-catalog listwise objective gives each symptom equal weight and assigns its target mass uniformly over its positive herbs. The additive herb prior comes exclusively from the positive associations of the current training partition.

```python
from nc_model.marginal_diffusion import build_model, ranking_loss
import torch

model = build_model(config, graph.x.shape[1], diffusion=True).to(graph.x.device)
herbs = torch.arange(graph.h.num_edges, device=graph.x.device)
symptoms = torch.arange(graph.s.num_edges, device=graph.x.device)
pairs = torch.cartesian_prod(herbs, symptoms)
state = model.encode(graph)
logits = model.decode(state, pairs).reshape(len(herbs), len(symptoms))
prior = positive_mask.sum(1).to(logits.dtype).log1p()
loss = ranking_loss(logits, positive_mask, prior)
```

The caller constructs the symptom incidence table from training symptoms only and remaps training association columns into this local numbering. Evaluation adds exactly one unseen symptom hyperedge to that graph in evaluation mode. Held symptom association labels enter only the metrics. Raw scores and `combine(scores, prior, coefficient, "calibrated")` give the protein branch and the deployment ranking. Select the coefficient using a second, symptom-disjoint validation split inside the training partition. The original BCE comparator uses `combine(original_scores, prior, alpha, "original")`, which standardizes each original score column over the herb catalogue.

The full experiment runner, historic checkpoints, five-fold manifests and diagnostic reports are delivered as a separate local experiment package. The original repository training entry point continues to run its original protocol.
