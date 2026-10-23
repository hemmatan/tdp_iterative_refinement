import torch
import torch.nn as nn

__all__ = ["BiaffineEdgeLabelPredictor"]


class BiaffineEdgeLabelPredictor(nn.Module):

    def __init__(self, emb_size, nlabels):
        super(BiaffineEdgeLabelPredictor, self).__init__()
        self.nlabels = nlabels
        self.U = nn.Parameter(torch.empty(nlabels, emb_size, emb_size))
        self.w_s = nn.Parameter(torch.empty(nlabels, emb_size))
        self.w_t = nn.Parameter(torch.empty(nlabels, emb_size))
        self.b = nn.Parameter(torch.zeros(nlabels))
        torch.nn.init.xavier_uniform_(self.U)
        torch.nn.init.xavier_uniform_(self.w_s)
        torch.nn.init.xavier_uniform_(self.w_t)

    def forward(self, graph, src_embeddings, tgt_embeddings=None):
        if type(src_embeddings) != str:
            graph.ndata["_e_src"] = src_embeddings
            src_embeddings = "_e_src"
        if tgt_embeddings is None:
            graph.ndata["_e_tgt"] = graph.ndata[src_embeddings]
            tgt_embeddings = "_e_tgt"
        elif type(tgt_embeddings) != str:
            graph.ndata["_e_tgt"] = tgt_embeddings
            tgt_embeddings = "_e_tgt"
        U = self.U
        w_s = self.w_s
        w_t = self.w_t
        b = self.b

        def message(edges):
            src = edges.src[src_embeddings]
            tgt = edges.dst[tgt_embeddings]
            bi = torch.einsum("be,lef,bf->bl", src, U, tgt)
            aff = (
                torch.einsum("be,le->bl", src, w_s)
                + torch.einsum("be,le->bl", tgt, w_t)
                + b
            )
            return {"_m": bi + aff}

        graph.apply_edges(message)
        return graph.edata["_m"]
