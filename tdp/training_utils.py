"""Span encoding, loss construction and training utilities."""

import random
import os
import dgl
import csv
import torch
import torch.nn as nn
import numpy as np
from tdp import loss_functions


def make_node_embeddings(
    graph, bert_embeddings, example, use_extra_features=False, nodeFeatureAugmenter=None
):
    num_nodes = len(graph.node_attributes)
    type_idx = []
    pos_idx = []
    sent_idx = []
    valid_node_idxes = []
    for i in range(num_nodes):
        attribute = graph.node_attributes[i]
        node_idxes = attribute["subword_idxes"]
        node_id = i
        if node_idxes:
            valid_node_idxes.append(i)
            type_idx.append(
                example.TEMPORAL_TAG_ENCODER.tag2id[
                    example.node_id2type(node_id=node_id)
                ]
            )
            pos_idx.append(example.node_id2position(node_id=node_id)[0])
            sent_idx.append(example.node_id2sentence_number(node_id=node_id))
    bert_means = torch.stack(
        [
            bert_embeddings[graph.node_attributes[i]["subword_idxes"]].mean(dim=0)
            for i in valid_node_idxes
        ]
    )
    if use_extra_features:
        type_idx_tensor = torch.tensor(
            type_idx, dtype=torch.long, device=bert_embeddings.device
        )
        pos_idx_tensor = torch.tensor(
            pos_idx, dtype=torch.long, device=bert_embeddings.device
        )
        sent_idx_tensor = torch.tensor(
            sent_idx, dtype=torch.long, device=bert_embeddings.device
        )
        extra_features = nodeFeatureAugmenter(
            bert_means, type_idx_tensor, pos_idx_tensor, sent_idx_tensor
        )
        node_embeddings = extra_features
    else:
        node_embeddings = bert_means
    graph.node_features["node_feat"] = node_embeddings
    return node_embeddings


def strided_bert_embeddings(
    encoder, subwordsIDs, sliding_segmentation, output_size, device
):
    """Encode overlapping windows and average subword states in overlapping regions."""
    pad_id = getattr(getattr(encoder, "config", None), "pad_token_id", None)
    if pad_id is not None:
        attention_mask = (subwordsIDs != pad_id).to(
            dtype=torch.long, device=subwordsIDs.device
        )
    else:
        attention_mask = torch.ones_like(
            subwordsIDs, dtype=torch.long, device=subwordsIDs.device
        )
    outputs = encoder(input_ids=subwordsIDs, attention_mask=attention_mask)
    bert_embeddings = (
        outputs.last_hidden_state
        if hasattr(outputs, "last_hidden_state")
        else outputs.to_tuple()[0]
    )
    bert_embeddings = bert_embeddings[:, 1:-1, :]
    _, output_length = sliding_segmentation[-1]
    emb_size = bert_embeddings.size(-1)
    out_embeddings = torch.zeros((output_length, emb_size), device=device)
    out_counter = torch.zeros_like(out_embeddings)
    for (bidx, eidx), window in zip(sliding_segmentation, bert_embeddings):
        sl = eidx - bidx
        out_embeddings[bidx:eidx, :] += window[:sl]
        out_counter[bidx:eidx, :] += 1
    out_embeddings /= out_counter.clamp_min(1)
    return out_embeddings[:output_size, :]


class Vocabulary:

    def __init__(self, symbols=None):
        self.symb2idx = {}
        self.idx2sym = []
        if symbols:
            self.update(symbols)

    def __len__(self):
        return len(self.idx2sym)

    def update(self, symbol_list):
        for S in symbol_list:
            if S not in self.symb2idx:
                self.symb2idx[S] = len(self.idx2sym)
                self.idx2sym.append(S)

    def from_dictionary(self, vocab):
        self.idx2sym = [-1 for i in range(len(vocab))]
        for token in vocab.keys():
            self.symb2idx[token] = vocab[token]
            self.idx2sym[vocab[token]] = token

    def rev_lookup(self, idx):
        return self.idx2sym[idx]

    def __call__(self, symbol, fallback=None):
        return self.lookup(symbol, fallback)

    def lookup(self, symbol, fallback=None):
        if fallback:
            return self.symb2idx.get(symbol, self.symb2idx[fallback])
        else:
            return self.symb2idx[symbol]


def write_namespace_to_csv(namespace, directory):
    filename = os.path.join(directory, "hyperparameters.csv")
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Key", "Value"])
        for key, value in vars(namespace).items():
            writer.writerow([key, value])


def set_all_seeds(seed: int = 30):
    """Seed supported RNGs. PYTHONHASHSEED is assigned after interpreter startup."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        dgl.random.seed(seed)
    except AttributeError:
        pass


def get_edge_loss_fnc(
    loss_function=None,
    alphas=None,
    gamma=None,
    which_loss_implementation=None,
):
    if loss_function != "focal_loss" or which_loss_implementation not in ("1", 1):
        raise ValueError("The paper configurations use focal-loss implementation 1.")
    edge_loss_fnc = loss_functions.FocalLossSaturnCloud(alpha=alphas, gamma=gamma)
    return edge_loss_fnc


def create_directory_if_not_exists(directory):
    os.makedirs(directory, exist_ok=True)


def get_graph_loss_fnc(graph_loss_fnc):
    if graph_loss_fnc != "binary_ce_logits":
        raise ValueError("The paper configurations use BCEWithLogitsLoss.")
    return nn.BCEWithLogitsLoss()


def make_class_balanced_alphas(
    counts: torch.Tensor,
    beta: float = 0.999,
    clip_range=(0.25, 4.0),
    normalize: str = "mean_one",
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
):
    eps = 1e-08
    counts = counts.to(dtype=torch.float32)
    if device is not None:
        counts = counts.to(device)
    safe_counts = torch.clamp(counts, min=1.0)
    eff_num = (1.0 - torch.pow(beta, safe_counts)).clamp_min(eps)
    raw = (1.0 - beta) / eff_num
    if normalize == "mean_one":
        raw = raw / raw.mean().clamp_min(eps)
    elif normalize == "sum_one":
        raw = raw / raw.sum().clamp_min(eps)
        raw = raw * raw.numel()
    low, high = clip_range
    alphas = torch.clamp(raw, min=low, max=high).to(dtype=dtype)
    return alphas
