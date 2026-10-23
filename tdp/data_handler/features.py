"""Learned node type, position and sentence-index features."""

import torch
from torch import nn


class NodeFeatureAugmenter(nn.Module):

    def __init__(self, bert_dim, type_num, max_nodes, max_sents, emb_dim=8):
        super().__init__()
        self.type_emb = nn.Embedding(type_num, emb_dim)
        self.pos_emb = nn.Embedding(max_nodes, emb_dim)
        self.sent_emb = nn.Embedding(max_sents, emb_dim)
        self.out_dim = bert_dim + 3 * emb_dim

    def forward(self, bert_feats, type_idx, pos_idx, sent_idx):
        assert bert_feats.shape[0] != 0
        assert (type_idx >= 0).all() and (
            type_idx < self.type_emb.num_embeddings
        ).all(), f"type_idx out of bounds: {type_idx}"
        assert (pos_idx >= 0).all() and (
            pos_idx < self.pos_emb.num_embeddings
        ).all(), f"pos_idx out of bounds: {pos_idx}"
        assert (sent_idx >= 0).all() and (
            sent_idx < self.sent_emb.num_embeddings
        ).all(), f"sent_idx out of bounds: {sent_idx}"
        type_feats = self.type_emb(type_idx)
        pos_feats = self.pos_emb(pos_idx)
        sent_feats = self.sent_emb(sent_idx)
        return torch.cat([bert_feats, type_feats, pos_feats, sent_feats], dim=-1)
