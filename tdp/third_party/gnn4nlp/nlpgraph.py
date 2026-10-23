import dgl
import dgl.nn
import torch
import torch.linalg

__all__ = ["complete_graph"]


def complete_graph(num_nodes, device=None):
    couples = [(n1, n2) for n1 in range(num_nodes) for n2 in range(num_nodes)]
    src_list, tgt_list = zip(*couples)
    return dgl.graph((torch.tensor(src_list), torch.tensor(tgt_list)), device=device)
