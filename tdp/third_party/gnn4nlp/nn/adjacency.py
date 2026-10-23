import dgl
import dgl.function as fn
import torch
import torch.nn as nn


class EdgeWeightsPredictor(nn.Module):

    def __init__(self, op, embedding_size, nheads=1):
        super(EdgeWeightsPredictor, self).__init__()
        if op != "biaffine" or nheads != 1:
            raise ValueError("The paper model uses a single biaffine scorer.")
        self.op = op
        self.nheads = 1
        self.embedding_size = embedding_size
        self.U = nn.Parameter(torch.empty(embedding_size, embedding_size))
        self.w_s = nn.Parameter(torch.empty(embedding_size))
        self.w_t = nn.Parameter(torch.empty(embedding_size))
        self.b = nn.Parameter(torch.zeros(1))
        torch.nn.init.xavier_uniform_(self.U)
        torch.nn.init.xavier_uniform_(self.w_s.unsqueeze(0))
        torch.nn.init.xavier_uniform_(self.w_t.unsqueeze(0))

    def forward(self, graph, src_embeddings, tgt_embeddings):
        if tgt_embeddings is None:
            tgt_embeddings = src_embeddings
        graph.ndata["src_embeddings"] = src_embeddings
        graph.ndata["tgt_embeddings"] = tgt_embeddings
        graph.ndata["tgt_pembeddings"] = (
            self.U @ graph.ndata["tgt_embeddings"].T
        ).T
        graph.apply_edges(fn.u_dot_v("src_embeddings", "tgt_pembeddings", "escore"))
        graph.ndata["x_aff"] = (graph.ndata["src_embeddings"] * self.w_s).sum(
            dim=1, keepdim=True
        )
        graph.ndata["y_aff"] = (graph.ndata["tgt_embeddings"] * self.w_t).sum(
            dim=1, keepdim=True
        )
        graph.apply_edges(fn.u_add_v("x_aff", "y_aff", "aff"))
        graph.edata["escore"] = (
            graph.edata["escore"] + graph.edata["aff"].squeeze(-1) + self.b
        )
        graph.edata["escore"] = graph.edata["escore"].squeeze(dim=-1)
        return (graph, graph.edata["escore"])


def dense_adj_matrix(graph, edge_weights=None, pad_value=float("-inf")):
    pad_size = max((g.num_nodes() for g in dgl.unbatch(graph)))
    if edge_weights is None:
        edge_weights = torch.ones(graph.num_edges(), device=graph.device)
    elif type(edge_weights) == str:
        edge_weights = graph.edata[edge_weights]
    graph.edata["escore"] = edge_weights
    adj_matrix_lst = []
    for g in dgl.unbatch(graph):
        g_index = torch.stack(g.edges())
        g_adj = torch.full((g.num_nodes(), pad_size), pad_value, device=graph.device)
        vals = g.edata["escore"]
        if vals.dim() > 1:
            vals = vals.squeeze(-1)
            if vals.dim() > 1:
                vals = vals.mean(dim=-1)
        g_adj[g_index[0, :], g_index[1, :]] = vals
        adj_matrix_lst.append(g_adj)
    return (torch.cat(adj_matrix_lst, dim=0), graph.batch_num_nodes())
