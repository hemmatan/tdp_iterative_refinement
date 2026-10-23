"""Adjacency construction, normalization and differentiable sparsification."""

import torch
import dgl


def adjency_matrix_to_dgl_graph(adj_matrix):
    src, dst = adj_matrix.nonzero(as_tuple=True)
    g = dgl.graph((src, dst), num_nodes=adj_matrix.size(0), device=adj_matrix.device)
    return g


def normalize_adjacency(
    adj: torch.Tensor,
    norm_type: str = "row",
    self_loop_alpha: float = 0.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    if norm_type != "row":
        raise ValueError("The paper model uses row normalization.")
    assert adj.ndim == 2 and adj.size(0) == adj.size(1), "adj must be [N,N]"
    N = adj.size(0)
    device, dtype = (adj.device, adj.dtype)
    A = torch.clamp(adj, min=0.0)
    deg = A.sum(dim=1, keepdim=True)
    A_hat = torch.zeros_like(A)
    nz = deg.squeeze(-1) > eps
    if nz.any():
        A_hat[nz] = A[nz] / deg[nz]
    z_idx = (~nz).nonzero(as_tuple=False).squeeze(-1)
    if z_idx.numel() > 0:
        A_hat[z_idx, z_idx] = 1.0
    if self_loop_alpha != 0.0:
        I = torch.eye(N, device=device, dtype=dtype)
        A_mp = (1.0 - self_loop_alpha) * A_hat + self_loop_alpha * I
    else:
        A_mp = A_hat
    return A_mp


def sparsify_logits_gumbel(
    logits: torch.Tensor,
    threshold,
    temperature: float = 0.5,
    binarize: bool = True,
    stochastic: bool = True,
    drop_self_loops: bool = True,
    eps: float = 1e-06,
) -> torch.Tensor:
    """Add single-Gumbel noise during training.

    Evaluation uses noise-free temperature scaling."""
    if torch.is_tensor(threshold):
        thr_p = threshold.to(dtype=logits.dtype, device=logits.device)
    else:
        thr_p = torch.tensor(float(threshold), dtype=logits.dtype, device=logits.device)
    thr_p = thr_p.clamp(eps, 1.0 - eps)
    thr_logit = torch.logit(thr_p)
    if thr_logit.dim() == 0:
        thr_logit = thr_logit.expand_as(logits)
    z = logits - thr_logit
    if stochastic:
        U = torch.rand_like(logits).clamp_(eps, 1.0 - eps)
        g = -torch.log(-torch.log(U))
        z = (z + g) / max(temperature, eps)
    else:
        z = z / max(temperature, eps)
    y_soft = torch.sigmoid(z)
    if binarize:
        y_hard = (y_soft >= 0.5).to(logits.dtype)
        out = (y_hard - y_soft).detach() + y_soft
    else:
        out = y_soft
    if drop_self_loops:
        eye = torch.eye(out.size(0), dtype=torch.bool, device=out.device)
        out = out.masked_fill(eye, 0.0)
    return out


def compute_gumbel_temperature(
    args: object, epoch: int, total_epochs: int, default_temp: float = None
) -> float:
    sched = str(getattr(args, "GUMBEL_TEMP_SCHEDULE", "none")).lower()
    if sched != "cosine":
        raise ValueError("The paper model uses a cosine temperature schedule.")
    if default_temp is None:
        default_temp = float(getattr(args, "GUMBEL_TEMP"))
    Tmin = float(getattr(args, "GUMBEL_TEMP_MIN"))
    warm = int(getattr(args, "GUMBEL_TEMP_WARMUP_EPOCHS"))
    if epoch <= warm:
        return float(default_temp)
    if total_epochs > warm:
        prog = (epoch - warm) / max(1, total_epochs - warm)
    else:
        prog = 1.0
    import math

    coeff = 0.5 * (1.0 + math.cos(math.pi * prog))
    T = Tmin + (default_temp - Tmin) * coeff
    return float(T)
