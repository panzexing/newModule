"""Sparse segment primitives implemented only with stock PyTorch."""
import torch


def segment_sum(values, index, size):
    output = values.new_zeros((size,) + tuple(values.shape[1:]))
    return output.index_add(0, index, values)


def segment_mean(values, index, size):
    counts = torch.bincount(index, minlength=size).to(values.dtype).clamp_min(1)
    return segment_sum(values, index, size) / counts.reshape((-1,) + (1,) * (values.ndim - 1))


def segment_softmax(logits, index, size):
    """Softmax along each incidence segment; empty segments have no entries."""
    if logits.shape[0] == 0:
        return logits
    expanded = index.reshape((-1,) + (1,) * (logits.ndim - 1)).expand_as(logits)
    maximum = logits.new_full((size,) + tuple(logits.shape[1:]), -torch.inf)
    maximum.scatter_reduce_(0, expanded, logits.detach(), reduce='amax', include_self=True)
    exp = (logits - maximum[index]).exp()
    denom = segment_sum(exp, index, size)
    return exp / denom[index].clamp_min(torch.finfo(logits.dtype).tiny)


def masked_softmax(logits, mask, dim=-1):
    """All-masked rows return exact zeros rather than NaNs."""
    mask = mask.expand_as(logits)
    any_valid = mask.any(dim=dim, keepdim=True)
    safe = logits.masked_fill(~mask, -torch.inf)
    safe = torch.where(any_valid, safe, torch.zeros_like(safe))
    return torch.softmax(safe, dim=dim) * mask.to(logits.dtype)
