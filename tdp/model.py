"""BERT/biaffine parser with the paper's iterative graph refinement."""

import json
import math
import torch.nn.functional as F
import os
from typing import Any, Dict, Optional, Tuple
from tdp.data_handler import features
from tdp.helpers import _current_timestamp, _initialize_encoder
from tdp.release_support import validate_supported_model
import dgl
import pandas as pd
import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from transformers.optimization import get_cosine_schedule_with_warmup
import tdp.evaluation_tools as evaluation_tools
from tdp.third_party.gnn4nlp import complete_graph
from tdp.third_party.gnn4nlp import nn as gnn
from tdp import training_utils
from tdp import graph_utils
from tdp.data_handler.tdg_loader import BasicDataLoader
import tdp.constants as constants
from torch.utils.checkpoint import checkpoint


class DynamicProjector(nn.Module):

    def __init__(self, out_dim: int, bias: bool = False):
        super().__init__()
        self.out_dim = int(out_dim)
        self.bias = bool(bias)
        self.adapters = nn.ModuleDict()

    def _make(self, in_dim: int) -> nn.Linear:
        layer = nn.Linear(in_dim, self.out_dim, bias=self.bias)
        if in_dim == self.out_dim and (not self.bias):
            with torch.no_grad():
                nn.init.eye_(layer.weight)
                layer.weight.add_(0.01 * torch.randn_like(layer.weight))
        else:
            nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dim = int(x.size(-1))
        key = str(in_dim)
        if key not in self.adapters:
            layer = self._make(in_dim).to(x.device)
            self.adapters[key] = layer
        else:
            cur = self.adapters[key]
            try:
                p = next(cur.parameters())
                if p.device != x.device:
                    self.adapters[key] = cur.to(x.device)
            except StopIteration:
                pass
        return self.adapters[key](x)


class TemporalDependencyParser(nn.Module):

    def _get_encoder_dimensions(self) -> dict:
        config = self.encoder.config
        return {
            "hidden_size": config.hidden_size,
            "vocab_size": config.vocab_size,
            "num_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "encoder_name": self.encoder.__class__.__name__.lower(),
            "config": config,
        }

    def __init__(
        self, token_vocab: Any, dep_vocab: Any, args: Any = None, encoder_type=None
    ):
        super(TemporalDependencyParser, self).__init__()
        self.args = args
        validate_supported_model(args, encoder_type)
        try:
            if not hasattr(self.args, "EVAL_TRAIN_DURING_TRAIN"):
                setattr(self.args, "EVAL_TRAIN_DURING_TRAIN", False)
            if not hasattr(self.args, "LOG_LONG_DOC_WINDOWS"):
                setattr(self.args, "LOG_LONG_DOC_WINDOWS", 50)
        except Exception:
            pass
        try:
            self.dep_vocab = dep_vocab
            self.num_edge_classes = len(dep_vocab)
        except Exception:
            self.dep_vocab = None
            self.num_edge_classes = None
        self.encoder = _initialize_encoder(encoder_type)
        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()
        if hasattr(self.encoder, "config") and hasattr(
            self.encoder.config, "use_cache"
        ):
            self.encoder.config.use_cache = False
        self.dropout = nn.Dropout(p=args.dropout)
        self._initialize_node_features()
        self._validate_dimensions()
        self._initialize_scoring_components(dep_vocab)
        self.proj_s = DynamicProjector(self.node_embedding_dim)
        self.proj_t = DynamicProjector(self.node_embedding_dim)
        self.gumbel_temp = float(getattr(self.args, "GUMBEL_TEMP"))
        init_p = float(getattr(self.args, "SPARSIFIER_THRESH_INIT"))
        init_p = min(max(init_p, 0.0001), 1.0 - 0.0001)
        self.sparsifier_threshold_logit = nn.Parameter(
            torch.logit(torch.tensor(init_p))
        )
        self.sparsity_coef = float(getattr(self.args, "SPARSITY_COEF"))
        self.sparsity_exclude_diag = bool(getattr(self.args, "SPARSITY_EXCLUDE_DIAG"))
        if args.MAX_ITERATIONS != 0:
            self._init_gnn_layers(self.node_embedding_dim)
        self._infer_threshold_override = None
        self._warned_long_docs = set()
        self._logged_iter_loss_policy = False
        self._best_fair_epoch = None
        self._log_encoder_configuration()
        self._optimizer_param_ids = None
        self._optimizer_param_names_at_init = None
        self._unoptimized_params_reported = set()
        self._unoptimized_delta_reported = set()

    def _record_optimizer_param_ids(self, optimizer) -> None:
        try:
            ids = set()
            for g in getattr(optimizer, "param_groups", []):
                for p in g.get("params", []):
                    ids.add(id(p))
            self._optimizer_param_ids = ids
            self._optimizer_param_names_at_init = {
                name for name, p in self.named_parameters() if id(p) in ids
            }
            try:
                constants.LOGGER.info(
                    f"[OPT-CHECK] Optimizer initialized with {len(ids)} parameter tensors (model currently has {sum((1 for _ in self.parameters()))} tensors)."
                )
            except Exception:
                pass
        except Exception as e:
            try:
                constants.LOGGER.warning(
                    f"[OPT-CHECK] Could not record optimizer params: {e}"
                )
            except Exception:
                pass

    def _log_params_missing_from_optimizer_after_forward(
        self, optimizer, context: str = ""
    ) -> None:
        if optimizer is None:
            return
        if not isinstance(getattr(self, "_optimizer_param_ids", None), set):
            return
        try:
            missing = []
            for name, p in self.named_parameters():
                if not getattr(p, "requires_grad", False):
                    continue
                if id(p) not in self._optimizer_param_ids:
                    missing.append((name, p))
            if not missing:
                return
            new_missing = [
                (n, p) for n, p in missing if n not in self._unoptimized_params_reported
            ]
            if not new_missing:
                return
            for n, _ in new_missing:
                self._unoptimized_params_reported.add(n)
            header = f"[OPT-CHECK] Detected {len(new_missing)} NEW trainable parameter(s) not in optimizer"
            if context:
                header += f" ({context})"
            constants.LOGGER.warning(header)
            show_k = min(25, len(new_missing))
            for n, p in new_missing[:show_k]:
                try:
                    shape = tuple(p.shape)
                except Exception:
                    shape = "?"
                try:
                    dev = str(p.device)
                except Exception:
                    dev = "?"
                constants.LOGGER.warning(
                    f"[OPT-CHECK]  - {n} | shape={shape} | device={dev} | will_not_update=True"
                )
            if len(new_missing) > show_k:
                constants.LOGGER.warning(
                    f"[OPT-CHECK]  ... and {len(new_missing) - show_k} more"
                )
        except Exception as e:
            try:
                constants.LOGGER.warning(f"[OPT-CHECK] Missing-param check failed: {e}")
            except Exception:
                pass

    def _log_unoptimized_param_step_deltas(
        self, optimizer, pre_step_snapshots: dict, context: str = ""
    ) -> None:
        if not pre_step_snapshots:
            return
        if optimizer is None:
            return
        if not isinstance(getattr(self, "_optimizer_param_ids", None), set):
            return
        try:
            deltas = []
            for name, before_cpu in pre_step_snapshots.items():
                try:
                    if name in getattr(self, "_unoptimized_delta_reported", set()):
                        continue
                except Exception:
                    pass
                p = dict(self.named_parameters()).get(name, None)
                if p is None or id(p) in self._optimizer_param_ids:
                    continue
                after_cpu = p.detach().float().cpu()
                try:
                    diff = (after_cpu - before_cpu).norm().item()
                except Exception:
                    diff = float("nan")
                deltas.append((name, diff))
            if not deltas:
                return
            header = "[OPT-CHECK] Post-step Δ norms for params NOT in optimizer"
            if context:
                header += f" ({context})"
            constants.LOGGER.warning(header)
            for name, diff in deltas[:25]:
                constants.LOGGER.warning(
                    f"[OPT-CHECK]  - {name} | delta_L2={diff:.6e} (expected ~0.0)"
                )
                try:
                    self._unoptimized_delta_reported.add(name)
                except Exception:
                    pass
        except Exception as e:
            try:
                constants.LOGGER.warning(f"[OPT-CHECK] Delta logging failed: {e}")
            except Exception:
                pass

    def _materialize_dynamic_projectors_for_optimization(self) -> None:
        try:
            device = next(self.parameters()).device
        except Exception:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        node_dim = int(getattr(self, "node_embedding_dim", 0) or 0)
        if node_dim <= 0:
            return
        with torch.no_grad():
            x_node = torch.zeros(1, node_dim, device=device)
            try:
                _ = self.proj_s(x_node)
            except Exception:
                pass
            try:
                _ = self.proj_t(x_node)
            except Exception:
                pass
            if hasattr(self, "gnn_in_projector"):
                try:
                    hidden_dim = int(getattr(self.gnn_in_projector, "out_dim", 0) or 0)
                except Exception:
                    hidden_dim = 0
                if hidden_dim > 0:
                    x_hidden = torch.zeros(1, hidden_dim, device=device)
                    try:
                        _ = self.gnn_in_projector(x_node)
                    except Exception:
                        pass
                    try:
                        _ = self.gnn_in_projector(x_hidden)
                    except Exception:
                        pass
                    try:
                        _ = self.proj_s(x_hidden)
                    except Exception:
                        pass
                    try:
                        _ = self.proj_t(x_hidden)
                    except Exception:
                        pass
                    try:
                        for rp in getattr(self, "residual_projections", []):
                            try:
                                _ = rp(x_hidden)
                            except Exception:
                                pass
                    except Exception:
                        pass

    def _initialize_node_features(self) -> None:
        encoder_dims = self._get_encoder_dimensions()
        base_hidden_size = encoder_dims["hidden_size"]
        if self.args.USE_EXTRA_FEATURES:
            self.nodeFeatureAugmenter = features.NodeFeatureAugmenter(
                bert_dim=base_hidden_size,
                type_num=3,
                max_nodes=2056,
                max_sents=128,
                emb_dim=8,
            )
            self.node_embedding_dim = self.nodeFeatureAugmenter.out_dim
            constants.LOGGER.info(
                f"Using augmented node features: {base_hidden_size} -> {self.node_embedding_dim}"
            )
        else:
            self.nodeFeatureAugmenter = None
            self.node_embedding_dim = base_hidden_size
            constants.LOGGER.info(
                f"Using base encoder features: {self.node_embedding_dim}"
            )
        self.node_norm = nn.LayerNorm(self.node_embedding_dim)

    def _log_encoder_configuration(self) -> None:
        encoder_dims = self._get_encoder_dimensions()
        constants.LOGGER.info("=" * 50)
        constants.LOGGER.info("ENCODER CONFIGURATION")
        constants.LOGGER.info("=" * 50)
        constants.LOGGER.info(f"Encoder Type: {encoder_dims['encoder_name']}")
        constants.LOGGER.info(f"Base Hidden Size: {encoder_dims['hidden_size']}")
        constants.LOGGER.info(f"Vocabulary Size: {encoder_dims['vocab_size']}")
        constants.LOGGER.info(f"Number of Layers: {encoder_dims['num_layers']}")
        constants.LOGGER.info(f"Attention Heads: {encoder_dims['num_attention_heads']}")
        constants.LOGGER.info(f"Use Extra Features: {self.args.USE_EXTRA_FEATURES}")
        constants.LOGGER.info(f"Final Node Embedding Dim: {self.node_embedding_dim}")
        constants.LOGGER.info(f"GNN Hidden Dim: {self.args.GCN_HIDDEN_DIM}")
        constants.LOGGER.info(f"GNN Type: {self.args.GNN_TYPE}")
        constants.LOGGER.info(f"Number of GNN Layers: {self.args.NUM_GNN_LAYERS}")
        constants.LOGGER.info("=" * 50)

    def _initialize_scoring_components(self, dep_vocab: Any) -> None:
        self.graph_scorer = gnn.EdgeWeightsPredictor(
            "biaffine", embedding_size=self.node_embedding_dim
        )
        self.edge_scorer = gnn.BiaffineEdgeLabelPredictor(
            self.node_embedding_dim, len(dep_vocab)
        )

    def forward(
        self, batch, current_example, mode, epoch_number=None, collect_metrics=False
    ):
        """Score T+1 states, with T message-passing updates.

        Intermediate arc scores use the pre-update features; label scores use
        post-update features. Both use the final features at state T."""
        try:
            training = mode == "train"
            self.train(training)
            start_time = None
            node_features = self._compute_node_features(
                current_example, apply_dropout=training
            )
            if self.args.MAX_ITERATIONS == 0:
                raw_logits, sparse_probs = self._compute_adjacency_representations(
                    batch, node_features, iteration_idx=0, mode=mode
                )
                edge_logits = self.edge_scorer(
                    batch, self.proj_s(node_features), self.proj_t(node_features)
                )
                blended_sparse_probs = sparse_probs
                self._last_blended = blended_sparse_probs.detach()
                self._last_features = node_features.detach()
                if mode in ("test", "predict"):
                    return self._create_final_sparse_graph(
                        blended_sparse_probs, node_features, batch
                    )
                elif mode == "train":
                    return (
                        [raw_logits],
                        [edge_logits],
                        [blended_sparse_probs],
                        [None],
                        [None],
                    )
                else:
                    return (
                        [raw_logits],
                        [edge_logits],
                        [blended_sparse_probs],
                        [node_features],
                        [blended_sparse_probs],
                    )
            if mode == "train":
                raw_logits_history = []
                sparse_probs_history = []
                edge_logits_history = []
                normalized_adj_history = None
                node_features_history = None
            elif mode == "val":
                raw_logits_history = []
                sparse_probs_history = []
                edge_logits_history = []
                normalized_adj_history = []
                node_features_history = []
            else:
                raw_logits_history = None
                sparse_probs_history = None
                normalized_adj_history = None
                edge_logits_history = None
                node_features_history = None
            last_normalized_adj = None
            prev_blended = None
            anchor_probs = None
            for iteration in range(self.args.MAX_ITERATIONS + 1):
                raw_logits, sparse_probs = self._compute_adjacency_representations(
                    batch, node_features, iteration, mode
                )
                prev_blended_for_diagnostic = prev_blended if iteration > 0 else None
                if iteration == 0:
                    blended_sparse_probs = sparse_probs
                    anchor_probs = (
                        sparse_probs if self.args.BPTT else sparse_probs.detach()
                    )
                    prev_blended = (
                        blended_sparse_probs.detach()
                        if not self.args.BPTT
                        else blended_sparse_probs
                    )
                else:
                    blended_sparse_probs = self._blend_sparse_probabilities(
                        curr_sparse=sparse_probs,
                        prev_sparse=prev_blended,
                        anchor_sparse=anchor_probs,
                    )
                    prev_blended = (
                        blended_sparse_probs.detach()
                        if not self.args.BPTT
                        else blended_sparse_probs
                    )
                probs_for_mp = blended_sparse_probs
                normalized_adj = self._normalize_for_message_passing(
                    probs_for_mp, iteration_idx=iteration
                )
                rs = normalized_adj.sum(dim=1)
                assert torch.allclose(
                    rs, torch.ones_like(rs), atol=0.0001
                ), "Row sums must be ~1 after mixing"
                node_features_pre_gnn = node_features
                if iteration < self.args.MAX_ITERATIONS:
                    if training and self.args.BPTT:
                        features_post = checkpoint(
                            lambda X, A: self._apply_gnn_with_normalized_adj(
                                X, A, topology_probs=probs_for_mp
                            ),
                            node_features,
                            normalized_adj,
                        )
                    else:
                        features_post = self._apply_gnn_with_normalized_adj(
                            node_features, normalized_adj, topology_probs=probs_for_mp
                        )
                else:
                    features_post = node_features
                if iteration > 0 and prev_blended_for_diagnostic is not None:
                    self._log_iteration_diagnostics(
                        iteration,
                        node_features_pre_gnn,
                        features_post,
                        blended_sparse_probs,
                        prev_blended_for_diagnostic,
                        mode,
                    )
                edge_logits = self.edge_scorer(
                    batch, self.proj_s(features_post), self.proj_t(features_post)
                )
                if mode in ("train", "val"):
                    raw_logits_history.append(raw_logits)
                    sparse_probs_history.append(blended_sparse_probs)
                    edge_logits_history.append(edge_logits)
                    if mode == "val":
                        normalized_adj_history.append(normalized_adj)
                        node_features_history.append(features_post)
                last_normalized_adj = normalized_adj
                node_features = features_post
            try:
                self._last_blended = prev_blended.detach()
                self._last_features = node_features.detach()
            except Exception:
                self._last_blended = prev_blended
                self._last_features = node_features
            if mode in ("test", "predict"):
                probs_for_labeling = getattr(self, "_last_blended", None)
                if probs_for_labeling is None:
                    probs_for_labeling = prev_blended
                predicted_sparse_graph = self._create_final_sparse_graph(
                    probs_for_labeling, node_features, batch
                )
                return predicted_sparse_graph
            elif mode == "train":
                steps = len(sparse_probs_history)
                return (
                    raw_logits_history,
                    edge_logits_history,
                    sparse_probs_history,
                    [None] * steps,
                    [None] * steps,
                )
            else:
                return (
                    raw_logits_history,
                    edge_logits_history,
                    sparse_probs_history,
                    node_features_history,
                    normalized_adj_history,
                )
        except Exception as e:
            constants.LOGGER.error(f"Error in forward method: {e}")
            constants.LOGGER.error(f"Mode: {mode}")
            constants.LOGGER.error(f"Epoch: {epoch_number}")
            constants.LOGGER.error(f"Args: {self.args}")
            raise e

    def _compute_adjacency_representations(self, batch, nodeX, iteration_idx, mode):
        """Return dense arc logits and sparsified message-passing probabilities."""
        complete_batch = self._build_complete_batch(batch, batch.device)
        raw_logits, _ = gnn.dense_adj_matrix(
            *self.graph_scorer(complete_batch, self.proj_s(nodeX), self.proj_t(nodeX)),
            pad_value=self.args.PAD_VALUE,
        )
        thr = torch.sigmoid(self.sparsifier_threshold_logit)
        sparse_probs = graph_utils.sparsify_logits_gumbel(
            logits=raw_logits,
            threshold=thr,
            temperature=float(self.gumbel_temp),
            binarize=bool(self.args.BINARIZE),
            stochastic=self.training,
            drop_self_loops=True,
        )
        return (raw_logits, sparse_probs)

    def _blend_sparse_probabilities(self, curr_sparse, prev_sparse, anchor_sparse):
        LMB = self.args.WEIGHT_PREV
        blended = LMB * prev_sparse + (1 - LMB) * curr_sparse
        return blended


    def _log_iteration_diagnostics(
        self,
        iteration,
        node_features_pre,
        node_features_post,
        sparse_probs,
        prev_sparse_probs,
        mode,
    ):
        if mode != "val" or iteration == 0:
            return
        with torch.no_grad():
            cos_sim = F.cosine_similarity(
                node_features_pre, node_features_post, dim=-1
            ).mean()
            delta_norm = torch.norm(
                node_features_post - node_features_pre, dim=-1
            ).mean()
            h_norm = torch.norm(node_features_pre, dim=-1).mean()
            relative_change = delta_norm / (h_norm + 1e-08)
            if hasattr(self, "sparsifier_threshold_logit"):
                learned_threshold = torch.sigmoid(
                    self.sparsifier_threshold_logit
                ).item()
                threshold_logit = self.sparsifier_threshold_logit.item()
            else:
                constants.LOGGER.warning(
                    "No sparsifier threshold logit found, using default 0.5"
                )
                learned_threshold = 0.5
                threshold_logit = 0.0
            curr_edges = (sparse_probs > learned_threshold).float()
            prev_edges = (prev_sparse_probs > learned_threshold).float()
            intersection = (curr_edges * prev_edges).sum()
            union = (curr_edges + prev_edges > 0).float().sum()
            jaccard = intersection / (union + 1e-08)
            edges_added = (curr_edges - prev_edges > 0).sum().item()
            edges_removed = (prev_edges - curr_edges > 0).sum().item()
            total_edges = curr_edges.sum().item()
            possible_edges = sparse_probs.numel()
            sparsity = curr_edges.mean().item()
            weight_change = torch.abs(sparse_probs - prev_sparse_probs).mean()
            ema_delta = torch.abs(sparse_probs - prev_sparse_probs).mean().item()
            constants.LOGGER.debug(
                f"🔬 [Iter {iteration - 1}→{iteration}] Feature: cos={cos_sim:.3f}, Δh/h={relative_change:.3f} | Adjacency: Jaccard={jaccard:.3f}, edges={total_edges:.0f} (+{edges_added:.0f}/-{edges_removed:.0f}), Δw={weight_change:.4f}"
            )
            constants.LOGGER.debug(
                f"  🎚️  Sparsifier: threshold={learned_threshold:.4f} (logit={threshold_logit:.4f}) | Sparsity: {sparsity:.4f} ({total_edges}/{possible_edges} edges) | EMA Δ: {ema_delta:.4f}"
            )
            if not hasattr(self, "_diagnostics_history"):
                self._diagnostics_history = []
            self._diagnostics_history.append(
                {
                    "iteration": iteration,
                    "cosine_sim": cos_sim.item(),
                    "relative_change": relative_change.item(),
                    "jaccard": jaccard.item(),
                    "edges_added": edges_added,
                    "edges_removed": edges_removed,
                    "total_edges": total_edges,
                    "weight_change": weight_change.item(),
                    "learned_threshold": learned_threshold,
                    "threshold_logit": threshold_logit,
                    "sparsity": sparsity,
                    "ema_delta": ema_delta,
                }
            )

    def _log_diagnostics_summary(self, mode):
        if (
            mode != "val"
            or not hasattr(self, "_diagnostics_history")
            or (not self._diagnostics_history)
        ):
            return
        avg_cos_sim = sum((d["cosine_sim"] for d in self._diagnostics_history)) / len(
            self._diagnostics_history
        )
        avg_rel_change = sum(
            (d["relative_change"] for d in self._diagnostics_history)
        ) / len(self._diagnostics_history)
        avg_jaccard = sum((d["jaccard"] for d in self._diagnostics_history)) / len(
            self._diagnostics_history
        )
        total_edges_added = sum((d["edges_added"] for d in self._diagnostics_history))
        total_edges_removed = sum(
            (d["edges_removed"] for d in self._diagnostics_history)
        )
        avg_sparsity = sum((d["sparsity"] for d in self._diagnostics_history)) / len(
            self._diagnostics_history
        )
        avg_ema_delta = sum((d["ema_delta"] for d in self._diagnostics_history)) / len(
            self._diagnostics_history
        )
        learned_threshold = self._diagnostics_history[0].get("learned_threshold", 0.5)
        threshold_logit = self._diagnostics_history[0].get("threshold_logit", 0.0)
        constants.LOGGER.info(
            f"📊 [Validation Summary] Avg Feature: cos={avg_cos_sim:.3f}, Δh/h={avg_rel_change:.3f} | Avg Adjacency: Jaccard={avg_jaccard:.3f} | Total Δ edges: +{total_edges_added:.0f}/-{total_edges_removed:.0f}"
        )
        constants.LOGGER.info(
            f"  🎚️  Threshold: {learned_threshold:.4f} (logit={threshold_logit:.4f}) | Avg Sparsity: {avg_sparsity:.4f} | Avg EMA Δ: {avg_ema_delta:.4f}"
        )
        if avg_cos_sim > 0.95:
            constants.LOGGER.warning(
                "⚠️  Feature cosine similarity > 0.95 → Features barely changing (residual dominance?)"
            )
        if avg_rel_change < 0.1:
            constants.LOGGER.warning(
                "⚠️  Relative change < 0.10 → GNN updates too weak (increase capacity?)"
            )
        if avg_jaccard > 0.97:
            constants.LOGGER.warning(
                "⚠️  Adjacency Jaccard > 0.97 → Structure not evolving (check loss aggregation?)"
            )
        if learned_threshold > 0.75:
            constants.LOGGER.warning(
                f"⚠️  Learned threshold very high ({learned_threshold:.4f}) → May be filtering out most edges!"
            )
        if learned_threshold < 0.25:
            constants.LOGGER.warning(
                f"⚠️  Learned threshold very low ({learned_threshold:.4f}) → May be keeping too many edges!"
            )
        if avg_sparsity < 0.01:
            constants.LOGGER.warning(
                f"⚠️  Very low sparsity ({avg_sparsity:.4f}) → Almost no edges predicted!"
            )
        if avg_sparsity > 0.5:
            constants.LOGGER.warning(
                f"⚠️  Very high sparsity ({avg_sparsity:.4f}) → Too many edges, graph too dense!"
            )
        if avg_ema_delta < 0.001:
            constants.LOGGER.warning(
                f"⚠️  EMA delta very small ({avg_ema_delta:.4f}) → EMA may be freezing adjacency!"
            )
        self._diagnostics_history = []

    def _normalize_for_message_passing(self, sparse_probs, iteration_idx):
        alpha_max = float(getattr(self.args, "SELF_LOOP_ALPHA"))
        alpha = alpha_max
        norm_type = 'row'
        self_loop_w = alpha
        return graph_utils.normalize_adjacency(
            sparse_probs, norm_type=norm_type, self_loop_alpha=self_loop_w
        )

    def _apply_gnn_with_normalized_adj(self, nodeX, normalized_adj, topology_probs):
        eps = 1e-08
        A = normalized_adj
        edge_mask = A.abs() > eps
        g = graph_utils.adjency_matrix_to_dgl_graph(edge_mask.float())
        g = g.to(nodeX.device)
        src, dst = g.edges(order="eid")
        edge_weights = A[src, dst].to(dtype=nodeX.dtype, device=nodeX.device)
        h = self.gnn_in_projector(nodeX)
        for i, gnn in enumerate(self.gnn_layers):
            h_in = h
            h_norm = self.gcn_layer_norms[i](h_in)
            out = gnn(g, h_norm, edge_weight=edge_weights)
            out = torch.relu(out)
            out = self.gcn_dropout(out)
            resid = self.residual_projections[i](h_in)
            h = resid + out
        return h

    def _create_final_sparse_graph(
        self, adj_matrix: torch.Tensor, nodeX: torch.Tensor, batch
    ):
        device = adj_matrix.device
        N = adj_matrix.size(0)
        probs = (
            adj_matrix
            if isinstance(adj_matrix, torch.Tensor)
            else getattr(self, "_last_blended", None)
        )
        if probs is None:
            try:
                N_fallback = int(nodeX.size(0))
                probs = torch.eye(N_fallback, dtype=nodeX.dtype, device=nodeX.device)
                constants.LOGGER.warning(
                    "No adjacency provided; defaulting to identity (will drop diagonal later)"
                )
            except Exception:
                probs = getattr(self, "_last_blended", None)
        infer_topk = int(getattr(self.args, "INFER_TOPK", 0))
        if infer_topk > 0:
            topk_vals, topk_idx = torch.topk(
                probs, k=min(infer_topk, probs.size(1)), dim=-1
            )
            mask = torch.zeros_like(probs, dtype=torch.bool)
            row_idx = (
                torch.arange(probs.size(0), device=probs.device)
                .unsqueeze(-1)
                .expand_as(topk_idx)
            )
            mask[row_idx, topk_idx] = True
        else:
            override = getattr(self, "_infer_threshold_override", None)
            if override is not None:
                thr = float(override)
            else:
                try:
                    thr = float(
                        torch.sigmoid(self.sparsifier_threshold_logit).detach().item()
                    )
                except Exception:
                    thr = float(
                        getattr(
                            self.args,
                            'SPARSIFIER_THRESH_PREDICTION',
                        )
                    )
                    constants.LOGGER.warning(
                        "No sparsifier threshold logit found; using default value"
                    )
            mask = probs > thr
        eye = torch.eye(N, dtype=torch.bool, device=device)
        mask = mask & ~eye
        src, dst = mask.nonzero(as_tuple=True)
        g = dgl.graph((src, dst), num_nodes=N, device=device)
        edge_logits = self.edge_scorer(g, self.proj_s(nodeX), self.proj_t(nodeX))
        g.edata[BasicDataLoader.DEPID_FEATURE] = edge_logits.argmax(1)
        g.ndata[BasicDataLoader.TOKID_FEATURE] = batch.ndata[
            BasicDataLoader.TOKID_FEATURE
        ]
        return g

    def set_inference_threshold(self, tau, sync_learned: bool = False):
        try:
            self._infer_threshold_override = None if tau is None else float(tau)
        except Exception:
            self._infer_threshold_override = None
        if sync_learned and tau is not None:
            eps = 1e-06
            val = float(max(min(float(tau), 1.0 - eps), eps))
            with torch.no_grad():
                try:
                    tgt = torch.tensor(
                        val,
                        device=self.sparsifier_threshold_logit.device,
                        dtype=self.sparsifier_threshold_logit.dtype,
                    )
                    self.sparsifier_threshold_logit.copy_(torch.logit(tgt))
                except Exception:
                    pass


    def create_graph_from_logits_with_threshold(
        self, logits: torch.Tensor, nodeX: torch.Tensor, batch, tau: float
    ):
        device = logits.device
        N = int(logits.size(0))
        probs = torch.sigmoid(logits)
        mask = probs > float(tau)
        eye = torch.eye(N, dtype=torch.bool, device=device)
        mask = mask & ~eye
        src, dst = mask.nonzero(as_tuple=True)
        g = dgl.graph((src, dst), num_nodes=N, device=device)
        edge_logits = self.edge_scorer(g, self.proj_s(nodeX), self.proj_t(nodeX))
        g.edata[BasicDataLoader.DEPID_FEATURE] = edge_logits.argmax(1)
        g.ndata[BasicDataLoader.TOKID_FEATURE] = batch.ndata[
            BasicDataLoader.TOKID_FEATURE
        ]
        return g

    def create_graph_from_probs_with_threshold(
        self, probs: torch.Tensor, nodeX: torch.Tensor, batch, tau: float
    ):
        device = probs.device
        N = int(probs.size(0))
        mask = probs > float(tau)
        eye = torch.eye(N, dtype=torch.bool, device=device)
        mask = mask & ~eye
        src, dst = mask.nonzero(as_tuple=True)
        g = dgl.graph((src, dst), num_nodes=N, device=device)
        edge_logits = self.edge_scorer(g, self.proj_s(nodeX), self.proj_t(nodeX))
        g.edata[BasicDataLoader.DEPID_FEATURE] = edge_logits.argmax(1)
        g.ndata[BasicDataLoader.TOKID_FEATURE] = batch.ndata[
            BasicDataLoader.TOKID_FEATURE
        ]
        return g


    def _init_gnn_layers(self, in_dim: int):
        num_layers = self.args.NUM_GNN_LAYERS
        hidden = int(self.args.GCN_HIDDEN_DIM)
        self.gnn_in_projector = DynamicProjector(hidden, bias=False)
        self.gnn_layers = nn.ModuleList()
        layer_norms = []
        residual_projections = []
        res_gate_params = []
        for i in range(num_layers):
            dim_in = hidden
            self.gnn_layers.append(
                GraphConv(dim_in, hidden, norm="none", allow_zero_in_degree=True)
            )
            out_dim = hidden
            layer_norms.append(nn.LayerNorm(out_dim))
            residual_projections.append(DynamicProjector(out_dim, bias=False))
            res_gate_params.append(nn.Parameter(torch.tensor(-1.0986123)))
        self.gcn_layer_norms = nn.ModuleList(layer_norms)
        self.gcn_dropout = nn.Dropout(p=self.args.GCN_DROPOUT)
        self.residual_projections = nn.ModuleList(residual_projections)
        self.res_gates_raw = nn.ParameterList(res_gate_params)


    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def run_epoch(
        self,
        dataloader,
        batch_size,
        graph_loss_coef,
        graph_loss_function,
        loss_function,
        training,
        optimizer=None,
        scheduler=None,
        epoch_number=None,
    ):
        if training:
            self.train()
            loop_ctx = torch.enable_grad()
        else:
            self.eval()
            loop_ctx = torch.no_grad()
        total_loss = total_graph = total_edge = 0.0
        with loop_ctx:
            for batch, idxs in dataloader.generate_batch(
                batch_size, randomize=bool(training)
            ):
                if training:
                    optimizer.zero_grad()
                batch = batch.to(self.device)
                current_example = dataloader.example_loader.dataset[idxs[0]]
                ref_matrix, _ = gnn.dense_adj_matrix(
                    batch, edge_weights=None, pad_value=0.0
                )
                forward_result = self(
                    batch,
                    current_example,
                    mode="train" if training else "val",
                    epoch_number=epoch_number,
                )
                if training and optimizer is not None:
                    try:
                        self._log_params_missing_from_optimizer_after_forward(
                            optimizer,
                            context=f"epoch={epoch_number}, doc={int(idxs[0])}",
                        )
                    except Exception:
                        pass
                if training and hasattr(self, "_train_msg_self_ratio_buffer"):
                    if not hasattr(self, "_msg_self_ratio_per_iter"):
                        self._msg_self_ratio_per_iter = {}
                    for it_idx, val in enumerate(self._train_msg_self_ratio_buffer):
                        if val is not None and val == val:
                            self._msg_self_ratio_per_iter.setdefault(it_idx, []).append(
                                float(val)
                            )
                    del self._train_msg_self_ratio_buffer
                logit_list, edge_preds, prob_list, _, norm_adj_list = forward_result
                assert (
                    len(logit_list)
                    == len(edge_preds)
                    == len(prob_list)
                    == self.args.MAX_ITERATIONS + 1
                )
                graph_loss, per_iter_graph_losses = (
                    evaluation_tools._compute_iterative_graph_loss(
                        adj_logits_list=logit_list,
                        adj_probs_list=prob_list,
                        ref_matrix=ref_matrix,
                        graph_loss_function=graph_loss_function,
                        agg=getattr(self.args, 'GRAPH_LOSS_AGG'),
                        graph_loss_weights=getattr(
                            self.args, "graph_loss_weights", None
                        ),
                    )
                )
                labels = batch.edata[BasicDataLoader.DEPID_FEATURE]
                edge_loss, per_iter_edge_losses = (
                    evaluation_tools._compute_iterative_edge_loss(
                        edge_preds,
                        labels,
                        loss_function,
                        agg=getattr(self.args, 'EDGE_LOSS_AGG'),
                        edge_loss_weights=None,
                    )
                )
                if not getattr(self, "_logged_iter_loss_policy", False):
                    try:
                        edge_agg = getattr(
                            self.args, 'EDGE_LOSS_AGG', None
                        )
                        graph_agg = getattr(
                            self.args, 'GRAPH_LOSS_AGG', None
                        )
                        gl_name = getattr(self.args, "graph_loss_fnc", None)
                        graph_space = (
                            "probs"
                            if isinstance(graph_loss_function, nn.BCELoss)
                            else "logits"
                        )
                        constants.LOGGER.info(
                            "[LOSS][ITER] EDGE_LOSS_AGG=%s | GRAPH_LOSS_AGG=%s | graph_loss_fnc=%s supervises=%s (MP uses blended probs)",
                            edge_agg,
                            graph_agg,
                            gl_name,
                            graph_space,
                        )
                        if (
                            edge_agg
                            == 'weighted_mean'
                        ):
                            constants.LOGGER.info(
                                "[LOSS][ITER] EDGE_LOSS_AGG=weighted_mean default weights=1..T (later iterations upweighted)"
                            )
                        if (
                            graph_agg
                            == 'weighted_mean'
                        ):
                            constants.LOGGER.info(
                                "[LOSS][ITER] GRAPH_LOSS_AGG=weighted_mean default weights=1..T (later iterations upweighted)"
                            )
                    except Exception:
                        pass
                    self._logged_iter_loss_policy = True
                sparsity_loss = torch.tensor(0.0, device=self.device)
                if self.sparsity_coef > 0.0:
                    offdiag_mask = None
                    acc = []
                    for probs_i in prob_list:
                        P = probs_i
                        if self.sparsity_exclude_diag:
                            N = P.size(0)
                            if offdiag_mask is None or offdiag_mask.size(0) != N:
                                offdiag_mask = ~torch.eye(
                                    N, dtype=torch.bool, device=P.device
                                )
                            val = P.masked_select(offdiag_mask).mean()
                        else:
                            val = P.mean()
                        acc.append(val)
                    sparsity_loss = torch.stack(acc).mean()
                base_loss = (
                    graph_loss_coef * graph_loss + (1 - graph_loss_coef) * edge_loss
                )
                loss_value = base_loss + self.sparsity_coef * sparsity_loss
                if training:
                    loss_value.backward()
                    try:

                        def _group_grad_norm(params_iter):
                            sq_sum = 0.0
                            count = 0
                            for p in params_iter:
                                if p.grad is None:
                                    continue
                                g = p.grad.detach()
                                sq_sum += float(g.pow(2).sum().item())
                                count += g.numel()
                            return sq_sum**0.5 if count > 0 else 0.0

                        enc_norm = (
                            _group_grad_norm(self.encoder.parameters())
                            if hasattr(self, "encoder")
                            else 0.0
                        )
                        nonenc_params = (
                            p
                            for n, p in self.named_parameters()
                            if not n.startswith("encoder.")
                        )
                        nonenc_norm = _group_grad_norm(nonenc_params)
                        if not hasattr(self, "_grad_norm_module"):
                            self._grad_norm_module = {}
                        self._grad_norm_module.setdefault("encoder", []).append(
                            float(enc_norm)
                        )
                        self._grad_norm_module.setdefault("non_encoder", []).append(
                            float(nonenc_norm)
                        )
                    except Exception:
                        pass
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                    pre_step = {}
                    try:
                        if isinstance(getattr(self, "_optimizer_param_ids", None), set):
                            already_reported = getattr(
                                self, "_unoptimized_delta_reported", set()
                            )
                            for name, p in self.named_parameters():
                                if (
                                    getattr(p, "requires_grad", False)
                                    and id(p) not in self._optimizer_param_ids
                                    and (name not in already_reported)
                                ):
                                    pre_step[name] = p.detach().float().cpu().clone()
                    except Exception:
                        pre_step = {}
                    optimizer.step()
                    try:
                        if scheduler is not None:
                            scheduler.step()
                    except Exception:
                        pass
                    try:
                        self._log_unoptimized_param_step_deltas(
                            optimizer,
                            pre_step,
                            context=f"epoch={epoch_number}, doc={int(idxs[0])}",
                        )
                    except Exception:
                        pass
                total_loss += loss_value.item()
                total_graph += graph_loss.item()
                total_edge += edge_loss.item()
                try:
                    if training:
                        if per_iter_graph_losses:
                            if not hasattr(self, "_per_iter_graph_loss"):
                                self._per_iter_graph_loss = {}
                            for it_idx, gl in enumerate(per_iter_graph_losses):
                                self._per_iter_graph_loss.setdefault(it_idx, []).append(
                                    float(gl.detach().item())
                                )
                        if per_iter_edge_losses:
                            if not hasattr(self, "_per_iter_edge_loss"):
                                self._per_iter_edge_loss = {}
                            for it_idx, el in enumerate(per_iter_edge_losses):
                                self._per_iter_edge_loss.setdefault(it_idx, []).append(
                                    float(el.detach().item())
                                )
                        try:

                            def _per_sample_edge_loss(logits, labels, lf):
                                try:
                                    old_red = getattr(lf, "reduction", None)
                                    if old_red is not None:
                                        lf.reduction = "none"
                                        out = lf(logits, labels)
                                        lf.reduction = old_red
                                        return out
                                except Exception:
                                    try:
                                        if old_red is not None:
                                            lf.reduction = old_red
                                    except Exception:
                                        pass
                                return torch.nn.functional.cross_entropy(
                                    logits, labels, reduction="none"
                                )

                            per_class_map = getattr(self, "_per_class_edge_loss", None)
                            if per_class_map is None:
                                self._per_class_edge_loss = {}
                                per_class_map = self._per_class_edge_loss
                            for it_idx in range(len(edge_preds)):
                                try:
                                    per_sample_losses = _per_sample_edge_loss(
                                        edge_preds[it_idx], labels, loss_function
                                    )
                                    unique_labels = torch.unique(labels).tolist()
                                    for cls in unique_labels:
                                        cls = int(cls)
                                        mask = labels == cls
                                        if mask.any():
                                            mean_loss = float(
                                                per_sample_losses[mask]
                                                .mean()
                                                .detach()
                                                .item()
                                            )
                                            per_class_map.setdefault(
                                                it_idx, {}
                                            ).setdefault(cls, []).append(mean_loss)
                                except Exception:
                                    pass
                        except Exception:
                            pass
                except Exception:
                    pass
        n_graphs = len(dataloader.graphs)
        return (total_loss / n_graphs, total_graph / n_graphs, total_edge / n_graphs)

    def predict(
        self, predict_loader: Any, batch_size: int, collect_metrics: bool = False
    ):
        self.eval()
        with torch.no_grad():
            for batch, idxs in predict_loader.generate_batch(
                batch_size, randomize=False
            ):
                batch = batch.to(self.device)
                current_example = predict_loader.example_loader.dataset[idxs[0]]
                sparse_batch = self(
                    batch,
                    current_example,
                    mode="predict",
                    collect_metrics=collect_metrics,
                )
                try:
                    sparse_batch = sparse_batch.to("cpu")
                except Exception:
                    pass
                yield (sparse_batch, idxs)

    def _setup_optimizer(self, args):
        """Build AdamW groups with decay excluded for biases and 1-D parameters."""
        base_lr = float(getattr(args, "learning_rate", 5e-05))
        weight_decay = float(getattr(args, "weight_decay", 0.01))
        encoder_lr = float(getattr(args, "encoder_learning_rate", base_lr))
        head_lr_mult = float(getattr(args, "head_lr_mult", 1.0))
        head_lr = float(getattr(args, "head_learning_rate", encoder_lr * head_lr_mult))
        no_decay_ids = set()
        try:
            for module in self.modules():
                if isinstance(module, nn.LayerNorm):
                    for p in module.parameters(recurse=False):
                        no_decay_ids.add(id(p))
        except Exception:
            pass
        encoder_decay = []
        encoder_no_decay = []
        head_decay = []
        head_no_decay = []
        for name, p in self.named_parameters():
            if not getattr(p, "requires_grad", False):
                continue
            is_encoder = name.startswith("encoder.")
            is_no_decay = (
                id(p) in no_decay_ids
                or name.endswith(".bias")
                or getattr(p, "ndim", 0) == 1
            )
            if is_encoder:
                (encoder_no_decay if is_no_decay else encoder_decay).append(p)
            else:
                (head_no_decay if is_no_decay else head_decay).append(p)
        optimizer_grouped_parameters = []
        if encoder_decay:
            optimizer_grouped_parameters.append(
                {
                    "params": encoder_decay,
                    "lr": encoder_lr,
                    "weight_decay": weight_decay,
                }
            )
        if encoder_no_decay:
            optimizer_grouped_parameters.append(
                {"params": encoder_no_decay, "lr": encoder_lr, "weight_decay": 0.0}
            )
        if head_decay:
            optimizer_grouped_parameters.append(
                {"params": head_decay, "lr": head_lr, "weight_decay": weight_decay}
            )
        if head_no_decay:
            optimizer_grouped_parameters.append(
                {"params": head_no_decay, "lr": head_lr, "weight_decay": 0.0}
            )
        try:
            constants.LOGGER.info(
                "[OPT] AdamW groups: enc_decay=%d, enc_no_decay=%d, head_decay=%d, head_no_decay=%d | encoder_lr=%.2e, head_lr=%.2e, weight_decay=%.2e",
                len(encoder_decay),
                len(encoder_no_decay),
                len(head_decay),
                len(head_no_decay),
                encoder_lr,
                head_lr,
                weight_decay,
            )
        except Exception:
            pass
        optimizer = optim.AdamW(optimizer_grouped_parameters)
        return optimizer


    def train_all(
        self,
        train_dataloader: Any,
        val_dataloader: Any,
        test_dataloader: Any,
        args,
        now: str,
        args_string: str,
        writer: Optional[SummaryWriter] = None,
        graph_loss_function=None,
        loss_function=None,
        param_combination: Dict = None,
        kfoldsetting: Any = None,
    ) -> str:
        own_writer = False
        if writer is None:
            writer = SummaryWriter(comment=args_string)
            own_writer = True
        num_epochs = args.num_epochs
        batch_size = args.batch_size
        graph_loss_coef = args.graph_loss_coef
        file_to_write = None
        timestamp = _current_timestamp()
        constants.LOGGER.info(f"{timestamp} Starting train_all()")
        try:
            self._materialize_dynamic_projectors_for_optimization()
        except Exception as e:
            try:
                constants.LOGGER.warning(
                    f"[OPT-CHECK] Could not materialize dynamic projectors pre-optimizer: {e}"
                )
            except Exception:
                pass
        optimizer = self._setup_optimizer(args)
        scheduler = None
        try:
            n_graphs = int(len(getattr(train_dataloader, "graphs", [])))
            steps_per_epoch = int(math.ceil(max(1, n_graphs) / max(1, int(batch_size))))
            num_training_steps = int(max(1, num_epochs) * steps_per_epoch)
            warmup_steps = int(getattr(args, "warmup_steps", 0) or 0)
            if warmup_steps <= 0:
                warmup_ratio = float(getattr(args, "warmup_ratio", 0.1))
                warmup_steps = int(round(warmup_ratio * num_training_steps))
            warmup_steps = int(
                max(0, min(warmup_steps, max(0, num_training_steps - 1)))
            )
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps,
            )
            constants.LOGGER.info(
                f"[LR-SCHED] warmup+cosine | steps/epoch={steps_per_epoch} | total_steps={num_training_steps} | warmup_steps={warmup_steps}"
            )
        except Exception as e:
            constants.LOGGER.warning(
                f"[LR-SCHED] Could not initialize warmup+cosine scheduler; falling back to constant LR. Error: {e}"
            )
            scheduler = None
        try:
            self._record_optimizer_param_ids(optimizer)
        except Exception:
            pass
        min_loss = float("inf")
        max_val_f1_macro_labeled = -1
        max_val_f1_macro_unlabeled = -1
        metrics = constants.METRICS.copy()
        tau_star_last_epoch = None
        eval_protocol_last_epoch = None
        dir_current_model = os.path.join(
            constants.PROJECT_PATH,
            "logs",
            constants.APP_START_TIMESTAMP,
            "statistics",
            args_string,
        )
        training_utils.create_directory_if_not_exists(dir_current_model)
        training_utils.write_namespace_to_csv(args, dir_current_model)
        constants.LOGGER.info(
            f"Model initialized with {self.get_trainable_parameter_count()[0]:,} trainable parameters out of {self.get_trainable_parameter_count()[1]:,} total"
        )
        for epoch in range(1, num_epochs + 1):
            try:
                if (
                    getattr(self.args, "GUMBEL_TEMP_SCHEDULE", "none") != "none"
                    and getattr(self.args, "SPARSIFIER_TYPE", None)
                    == 'GUMBEL'
                ):
                    self.gumbel_temp = graph_utils.compute_gumbel_temperature(
                        self.args,
                        epoch=epoch,
                        total_epochs=num_epochs,
                        default_temp=float(getattr(self.args, "GUMBEL_TEMP")),
                    )
                    try:
                        constants.LOGGER.info(
                            f"Updated Gumbel temperature to {self.gumbel_temp:.4f} (schedule={getattr(self.args, 'GUMBEL_TEMP_SCHEDULE', 'none')}, epoch={epoch})"
                        )
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                self._per_iter_graph_loss = {}
                self._per_iter_edge_loss = {}
                self._grad_norm_adj_per_iter = {}
                self._msg_self_ratio_per_iter = {}
                self._per_class_edge_loss = {}
                self._grad_norm_module = {}
                self._diagnostics_history = []
            except Exception:
                pass
            try:
                if torch.cuda.is_available():
                    dev = next(self.parameters()).device
                    if getattr(dev, "type", None) == "cuda":
                        idx = (
                            dev.index
                            if getattr(dev, "index", None) is not None
                            else torch.cuda.current_device()
                        )
                        try:
                            torch.cuda.reset_peak_memory_stats(idx)
                        except TypeError:
                            torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
            constants.LOGGER.info(f"{'=' * 60}")
            constants.LOGGER.info(f"[{args_string}] Epoch {epoch}/{num_epochs}")
            constants.LOGGER.info(f"{'=' * 60}")
            train_loss, train_graph_loss, train_edge_loss = self.run_epoch(
                dataloader=train_dataloader,
                batch_size=batch_size,
                graph_loss_coef=graph_loss_coef,
                graph_loss_function=graph_loss_function,
                loss_function=loss_function,
                training=True,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch_number=epoch,
            )
            try:
                diag_dir = os.path.join(
                    constants.PROJECT_PATH,
                    "logs",
                    constants.APP_START_TIMESTAMP,
                    "statistics",
                    args_string,
                    "diagnostics",
                )
                evaluation_tools.generate_gradient_flow_report(
                    self,
                    out_dir=diag_dir,
                    filename=f"gradient_flow_epoch_{epoch}.txt",
                    logger=constants.LOGGER,
                )
            except Exception as e:
                constants.LOGGER.warning(
                    f"Could not generate gradient flow report: {e}"
                )
            val_loss, val_graph_loss, val_edge_loss = self.run_epoch(
                dataloader=val_dataloader,
                batch_size=batch_size,
                graph_loss_coef=graph_loss_coef,
                graph_loss_function=graph_loss_function,
                loss_function=loss_function,
                training=False,
                epoch_number=epoch,
            )
            eval_protocol = str(
                getattr(self.args, "EVAL_PROTOCOL", "global_tau")
            ).lower()
            sync_learned = bool(
                getattr(self.args, "EVAL_SYNC_LEARNED_THRESHOLD", False)
            )
            try:
                device = next(self.parameters()).device
            except Exception:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            fair_dev_metrics = None
            misclassified_val = []
            tau_star_dev = None
            try:
                if eval_protocol == "global_tau":
                    tau_star_dev, _ = (
                        evaluation_tools.calibrate_global_threshold_on_dev(
                            self,
                            val_dataloader,
                            batch_size=1,
                            device=device,
                            metric="labeled_f1_doc_macro",
                            grid=None,
                            use_logits=True,
                            max_docs=None,
                        )
                    )
                    try:
                        self.set_inference_threshold(
                            tau_star_dev, sync_learned=sync_learned
                        )
                    except Exception:
                        pass
                    dev_graphs = evaluation_tools.predict_with_global_threshold(
                        self,
                        val_dataloader,
                        tau_star_dev,
                        batch_size=batch_size,
                        device=device,
                        use_logits=True,
                    )
                    fair_dev_metrics, misclassified_val = (
                        evaluation_tools.evaluate_predicted_graphs(
                            dev_graphs, val_dataloader, save_statistics=True
                        )
                    )
                    try:
                        constants.LOGGER.info(
                            f"[EVAL=global_tau][DEV][epoch {epoch}] tau*={tau_star_dev:.3f} | labeled F1 (doc-macro/micro/label-macro)={fair_dev_metrics[constants.METRIC_NAMES.labeled_f1_doc_macro]:.3f}/{fair_dev_metrics[constants.METRIC_NAMES.labeled_f1_micro]:.3f}/{fair_dev_metrics[constants.METRIC_NAMES.labeled_f1_macro]:.3f}; unlabeled F1 (doc-macro/micro)={fair_dev_metrics[constants.METRIC_NAMES.unlabeled_f1_doc_macro]:.3f}/{fair_dev_metrics[constants.METRIC_NAMES.unlabeled_f1_micro]:.3f}"
                        )
                        constants.LOGGER.info(
                            f"[EVAL=global_tau][DEV][epoch {epoch}][temporal-only] labeled F1 (doc-macro/micro/label-macro)={fair_dev_metrics[constants.METRIC_NAMES.labeled_f1_doc_macro_temporal]:.3f}/{fair_dev_metrics[constants.METRIC_NAMES.labeled_f1_micro_temporal]:.3f}/{fair_dev_metrics[constants.METRIC_NAMES.labeled_f1_macro_temporal]:.3f}; unlabeled F1 (doc-macro/micro)={fair_dev_metrics[constants.METRIC_NAMES.unlabeled_f1_doc_macro_temporal]:.3f}/{fair_dev_metrics[constants.METRIC_NAMES.unlabeled_f1_micro_temporal]:.3f}"
                        )
                    except Exception:
                        pass
                else:
                    metrics_on_val, misclassified_val = (
                        evaluation_tools.predict_and_evaluate(
                            self, val_dataloader, batch_size, save_statistics=True
                        )
                    )
                    fair_dev_metrics = metrics_on_val
                    try:
                        constants.LOGGER.info(
                            f"[EVAL=learned][DEV][epoch {epoch}] labeled F1 (doc-macro/micro/label-macro)={metrics_on_val[constants.METRIC_NAMES.labeled_f1_doc_macro]:.3f}/{metrics_on_val[constants.METRIC_NAMES.labeled_f1_micro]:.3f}/{metrics_on_val[constants.METRIC_NAMES.labeled_f1_macro]:.3f}; unlabeled F1 (doc-macro/micro)={metrics_on_val[constants.METRIC_NAMES.unlabeled_f1_doc_macro]:.3f}/{metrics_on_val[constants.METRIC_NAMES.unlabeled_f1_micro]:.3f}"
                        )
                        constants.LOGGER.info(
                            f"[EVAL=learned][DEV][epoch {epoch}][temporal-only] labeled F1 (doc-macro/micro/label-macro)={metrics_on_val[constants.METRIC_NAMES.labeled_f1_doc_macro_temporal]:.3f}/{metrics_on_val[constants.METRIC_NAMES.labeled_f1_micro_temporal]:.3f}/{metrics_on_val[constants.METRIC_NAMES.labeled_f1_macro_temporal]:.3f}; unlabeled F1 (doc-macro/micro)={metrics_on_val[constants.METRIC_NAMES.unlabeled_f1_doc_macro_temporal]:.3f}/{metrics_on_val[constants.METRIC_NAMES.unlabeled_f1_micro_temporal]:.3f}"
                        )
                    except Exception:
                        pass
            except Exception as e:
                constants.LOGGER.warning(
                    f"[EVAL] Dev evaluation failed during training: {e}"
                )
                fair_dev_metrics = None
                misclassified_val = []
            metrics_on_train = None
            try:
                eval_train = bool(getattr(self.args, "EVAL_TRAIN_DURING_TRAIN", False))
            except Exception:
                eval_train = False
            if eval_train:
                try:
                    if eval_protocol == "global_tau" and tau_star_dev is not None:
                        train_graphs = evaluation_tools.predict_with_global_threshold(
                            self,
                            train_dataloader,
                            tau_star_dev,
                            batch_size=batch_size,
                            device=device,
                            use_logits=True,
                        )
                        metrics_on_train, _ = (
                            evaluation_tools.evaluate_predicted_graphs(
                                train_graphs, train_dataloader, save_statistics=False
                            )
                        )
                        try:
                            del train_graphs
                        except Exception:
                            pass
                    else:
                        metrics_on_train, _ = evaluation_tools.predict_and_evaluate(
                            self, train_dataloader, batch_size, save_statistics=False
                        )
                except Exception as e:
                    constants.LOGGER.warning(f"[EVAL] Train evaluation failed: {e}")
                    metrics_on_train = None
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
            elif not getattr(self, "_logged_train_eval_disabled", False):
                constants.LOGGER.info(
                    "[EVAL] Skipping train evaluation during training (EVAL_TRAIN_DURING_TRAIN=False)"
                )
                setattr(self, "_logged_train_eval_disabled", True)
            metrics_on_val = (
                fair_dev_metrics
                if fair_dev_metrics is not None
                else constants.METRICS.copy()
            )
            try:
                tau_star_last_epoch = tau_star_dev
                eval_protocol_last_epoch = eval_protocol
            except Exception:
                pass
            try:
                self._log_diagnostics_summary("val")
            except Exception as e:
                constants.LOGGER.warning(f"Could not log diagnostics summary: {e}")
            try:
                lrs = [
                    float(pg.get("lr", 0.0))
                    for pg in getattr(optimizer, "param_groups", [])
                ]
                lr_main = (
                    max(lrs) if lrs else float(getattr(args, "learning_rate", 0.0))
                )
                lr_min = min(lrs) if lrs else lr_main
                lr_max = lr_main
                if lr_min != lr_max:
                    constants.LOGGER.info(
                        f"  LR groups: min={lr_min:.2e} max={lr_max:.2e} (n_groups={len(lrs)})"
                    )
            except Exception:
                lr_main = float(getattr(args, "learning_rate", 0.0))
            self._log_epoch_summary(
                epoch,
                train_loss,
                val_loss,
                train_graph_loss,
                val_graph_loss,
                train_edge_loss,
                val_edge_loss,
                metrics_on_train,
                metrics_on_val,
                lr_main,
            )
            max_val_f1_macro_unlabeled, max_val_f1_macro_labeled, min_loss, metrics = (
                self._save_best_model(
                    epoch,
                    metrics_on_train,
                    metrics_on_val,
                    now,
                    args_string,
                    max_val_f1_macro_labeled,
                    max_val_f1_macro_unlabeled,
                    val_loss,
                    min_loss,
                    metrics,
                    fair_dev_metrics=fair_dev_metrics,
                    prefer_fair=True,
                    model_basename=f"{now}{args_string}",
                )
            )
            if writer:
                self._write_tensorboard_summary(
                    writer,
                    epoch,
                    train_loss,
                    val_loss,
                    train_graph_loss,
                    val_graph_loss,
                    train_edge_loss,
                    val_edge_loss,
                    metrics_on_train,
                    metrics_on_val,
                )
                if epoch % 5 == 0:
                    writer.flush()
        misclassified_edges_properties_val = misclassified_val
        try:
            if torch.cuda.is_available():
                dev = next(self.parameters()).device
                if getattr(dev, "type", None) == "cuda":
                    idx = (
                        dev.index
                        if getattr(dev, "index", None) is not None
                        else torch.cuda.current_device()
                    )
                    alloc = torch.cuda.memory_allocated(idx) / (1024.0 * 1024.0)
                    reserved = torch.cuda.memory_reserved(idx) / (1024.0 * 1024.0)
                    peak_alloc = torch.cuda.max_memory_allocated(idx) / (
                        1024.0 * 1024.0
                    )
                    try:
                        peak_reserved = torch.cuda.max_memory_reserved(idx) / (
                            1024.0 * 1024.0
                        )
                    except Exception:
                        peak_reserved = float("nan")
                    total = torch.cuda.get_device_properties(idx).total_memory / (
                        1024.0 * 1024.0
                    )
                    constants.LOGGER.info(
                        f"[MEM][epoch {epoch}] CUDA alloc={alloc:.1f} MB, reserved={reserved:.1f} MB | peak alloc={peak_alloc:.1f} MB, peak reserved={peak_reserved:.1f} MB of {total:.0f} MB"
                    )
        except Exception:
            pass
        if epoch % 5 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        try:
            per_iter_dir = os.path.join(
                constants.PROJECT_PATH,
                "logs",
                constants.APP_START_TIMESTAMP,
                "per_iter",
                f"{now}_{args_string}",
            )
            training_utils.create_directory_if_not_exists(per_iter_dir)
            extra_cols = {"timestamp": now, "model_name": args_string}
            try:
                if param_combination is not None:
                    extra_cols["param_combination"] = dict(param_combination)
            except Exception:
                pass
            try:
                if hasattr(args, "seed") and args.seed is not None:
                    extra_cols["seed"] = int(args.seed)
            except Exception:
                pass
            evaluation_tools.save_train_per_iter_metrics(
                self,
                per_iter_dir,
                logger=constants.LOGGER,
                extra_columns=extra_cols,
                epoch=epoch,
                append=True,
            )
        except Exception as e:
            constants.LOGGER.warning(
                f"Could not append per-epoch train per-iteration metrics: {e}"
            )
        constants.LOGGER.info("=" * 60)
        constants.LOGGER.info(
            "Final dev evaluation using best checkpoint (protocol-controlled)"
        )
        constants.LOGGER.info("=" * 60)
        try:
            best_fair_path = getattr(self, "_best_fair_model_path", None)
            if best_fair_path is not None and os.path.exists(best_fair_path):
                self.load_state_dict(torch.load(best_fair_path))
        except Exception as e:
            constants.LOGGER.warning(f"Could not reload best fair checkpoint: {e}")
        try:
            device = next(self.parameters()).device
        except Exception:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tau_star_final = None
        misclassified_best_val = []
        try:
            eval_protocol = str(
                getattr(self.args, "EVAL_PROTOCOL", "global_tau")
            ).lower()
            sync_learned = bool(
                getattr(self.args, "EVAL_SYNC_LEARNED_THRESHOLD", False)
            )
            if eval_protocol == "global_tau":
                tau_star_final, _ = evaluation_tools.calibrate_global_threshold_on_dev(
                    self,
                    val_dataloader,
                    batch_size=1,
                    device=device,
                    metric="labeled_f1_doc_macro",
                    grid=None,
                    use_logits=True,
                    max_docs=None,
                )
                try:
                    self.set_inference_threshold(
                        tau_star_final, sync_learned=sync_learned
                    )
                except Exception:
                    pass
                dev_graphs_final = evaluation_tools.predict_with_global_threshold(
                    self,
                    val_dataloader,
                    tau_star_final,
                    batch_size=1,
                    device=device,
                    use_logits=True,
                )
                fair_dev_final_metrics, misclassified_best_val = (
                    evaluation_tools.evaluate_predicted_graphs(
                        dev_graphs_final, val_dataloader, save_statistics=True
                    )
                )
                constants.LOGGER.info(
                    f"[EVAL=global_tau][FINAL DEV] tau*={tau_star_final:.3f} | labeled F1 (doc-macro/micro/label-macro)={fair_dev_final_metrics[constants.METRIC_NAMES.labeled_f1_doc_macro]:.3f}/{fair_dev_final_metrics[constants.METRIC_NAMES.labeled_f1_micro]:.3f}/{fair_dev_final_metrics[constants.METRIC_NAMES.labeled_f1_macro]:.3f}; unlabeled F1 (doc-macro/micro)={fair_dev_final_metrics[constants.METRIC_NAMES.unlabeled_f1_doc_macro]:.3f}/{fair_dev_final_metrics[constants.METRIC_NAMES.unlabeled_f1_micro]:.3f}"
                )
                try:
                    constants.LOGGER.info(
                        f"[EVAL=global_tau][FINAL DEV][temporal-only] labeled F1 (doc-macro/micro/label-macro)={fair_dev_final_metrics[constants.METRIC_NAMES.labeled_f1_doc_macro_temporal]:.3f}/{fair_dev_final_metrics[constants.METRIC_NAMES.labeled_f1_micro_temporal]:.3f}/{fair_dev_final_metrics[constants.METRIC_NAMES.labeled_f1_macro_temporal]:.3f}; unlabeled F1 (doc-macro/micro)={fair_dev_final_metrics[constants.METRIC_NAMES.unlabeled_f1_doc_macro_temporal]:.3f}/{fair_dev_final_metrics[constants.METRIC_NAMES.unlabeled_f1_micro_temporal]:.3f}"
                    )
                except Exception:
                    pass
            else:
                fair_dev_final_metrics, misclassified_best_val = (
                    evaluation_tools.predict_and_evaluate(
                        self, val_dataloader, batch_size=1, save_statistics=True
                    )
                )
                constants.LOGGER.info(
                    f"[EVAL=learned][FINAL DEV] labeled F1 (doc-macro/micro/label-macro)={fair_dev_final_metrics[constants.METRIC_NAMES.labeled_f1_doc_macro]:.3f}/{fair_dev_final_metrics[constants.METRIC_NAMES.labeled_f1_micro]:.3f}/{fair_dev_final_metrics[constants.METRIC_NAMES.labeled_f1_macro]:.3f}; unlabeled F1 (doc-macro/micro)={fair_dev_final_metrics[constants.METRIC_NAMES.unlabeled_f1_doc_macro]:.3f}/{fair_dev_final_metrics[constants.METRIC_NAMES.unlabeled_f1_micro]:.3f}"
                )
                try:
                    constants.LOGGER.info(
                        f"[EVAL=learned][FINAL DEV][temporal-only] labeled F1 (doc-macro/micro/label-macro)={fair_dev_final_metrics[constants.METRIC_NAMES.labeled_f1_doc_macro_temporal]:.3f}/{fair_dev_final_metrics[constants.METRIC_NAMES.labeled_f1_micro_temporal]:.3f}/{fair_dev_final_metrics[constants.METRIC_NAMES.labeled_f1_macro_temporal]:.3f}; unlabeled F1 (doc-macro/micro)={fair_dev_final_metrics[constants.METRIC_NAMES.unlabeled_f1_doc_macro_temporal]:.3f}/{fair_dev_final_metrics[constants.METRIC_NAMES.unlabeled_f1_micro_temporal]:.3f}"
                    )
                except Exception:
                    pass
        except Exception as e:
            constants.LOGGER.warning(f"[FINAL] evaluation failed: {e}")
            fair_dev_final_metrics = None
            misclassified_best_val = []
        cpu_param_combination = (
            {
                k: v.cpu().detach().numpy() if isinstance(v, torch.Tensor) else v
                for k, v in param_combination.items()
            }
            if param_combination is not None
            else {}
        )
        results = cpu_param_combination.copy()
        results.update({"kfoldsetting": kfoldsetting, "timestamp": now, **metrics})
        try:
            if fair_dev_final_metrics is not None:
                results.update(
                    {
                        "fair_dev_f1_labeled_micro": fair_dev_final_metrics[
                            constants.METRIC_NAMES.labeled_f1_micro
                        ],
                        "fair_dev_f1_labeled_macro": fair_dev_final_metrics[
                            constants.METRIC_NAMES.labeled_f1_macro
                        ],
                        "fair_dev_f1_labeled_doc_macro": fair_dev_final_metrics[
                            constants.METRIC_NAMES.labeled_f1_doc_macro
                        ],
                        "fair_dev_f1_unlabeled_micro": fair_dev_final_metrics[
                            constants.METRIC_NAMES.unlabeled_f1_micro
                        ],
                        "fair_dev_f1_unlabeled_doc_macro": fair_dev_final_metrics[
                            constants.METRIC_NAMES.unlabeled_f1_doc_macro
                        ],
                    }
                )
        except Exception:
            pass
        try:
            provenance_last = {
                "kind": "last_epoch",
                "epoch": int(num_epochs),
                "eval_protocol": str(
                    eval_protocol_last_epoch
                    or getattr(self.args, "EVAL_PROTOCOL", "global_tau")
                ).lower(),
                "tau_star": (
                    float(tau_star_last_epoch)
                    if tau_star_last_epoch is not None
                    else None
                ),
                "best_fair_epoch": (
                    int(getattr(self, "_best_fair_epoch"))
                    if getattr(self, "_best_fair_epoch", None) is not None
                    else None
                ),
                "note": "These files reflect the dev evaluation performed during training at the final epoch.",
            }
        except Exception:
            provenance_last = None
        self._save_final_statistics(
            dir_current_model,
            misclassified_edges_properties_val,
            [],
            metrics_on_val,
            results,
            file_to_write,
            statistics_subdir="final_statistics",
            write_results=True,
            provenance=provenance_last,
        )
        try:
            provenance_best = {
                "kind": "best_fair_checkpoint",
                "best_fair_epoch": (
                    int(getattr(self, "_best_fair_epoch"))
                    if getattr(self, "_best_fair_epoch", None) is not None
                    else None
                ),
                "checkpoint_path": getattr(self, "_best_fair_model_path", None),
                "eval_protocol": str(
                    getattr(self.args, "EVAL_PROTOCOL", "global_tau")
                ).lower(),
                "tau_star": (
                    float(tau_star_final) if tau_star_final is not None else None
                ),
                "note": "These files reflect the reloaded best fair checkpoint and the final dev calibration (tau*).",
            }
        except Exception:
            provenance_best = None
        try:
            self._save_final_statistics(
                dir_current_model,
                misclassified_best_val,
                [],
                fair_dev_final_metrics if fair_dev_final_metrics is not None else {},
                results,
                file_to_write,
                statistics_subdir="final_statistics_best_fair",
                write_results=False,
                provenance=provenance_best,
            )
        except Exception as e:
            constants.LOGGER.warning(
                f"Could not save best-checkpoint final statistics: {e}"
            )
        try:
            if own_writer and writer is not None:
                writer.flush()
                writer.close()
        except Exception:
            pass
        try:
            best_fair_path = getattr(self, "_best_fair_model_path", None)
            if best_fair_path is not None and os.path.exists(best_fair_path):
                return (
                    os.path.basename(best_fair_path),
                    fair_dev_final_metrics,
                    tau_star_final,
                )
        except Exception:
            pass
        return (
            f"{now}{args_string}_best_fair.pt",
            fair_dev_final_metrics,
            tau_star_final,
        )

    def _log_epoch_summary(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_graph_loss: float,
        val_graph_loss: float,
        train_edge_loss: float,
        val_edge_loss: float,
        metrics_on_train: dict,
        metrics_on_val: dict,
        learning_rate: float,
    ):

        def _fmt(metrics: Any, key: str, default: str = "N/A") -> str:
            try:
                if not isinstance(metrics, dict):
                    return str(default)
                v = metrics.get(key, None)
                if v is None:
                    return str(default)
                return f"{float(v):.4f}"
            except Exception:
                return str(default)

        constants.LOGGER.info(f"\nEpoch {epoch} Summary:")
        constants.LOGGER.info(f"  Learning rate: {learning_rate:.2e}")
        constants.LOGGER.info(f"  Losses:")
        constants.LOGGER.info(
            f"    Train: {train_loss:.4f} (graph: {train_graph_loss:.4f}, edge: {train_edge_loss:.4f})"
        )
        constants.LOGGER.info(
            f"    Val:   {val_loss:.4f} (graph: {val_graph_loss:.4f}, edge: {val_edge_loss:.4f})"
        )
        constants.LOGGER.info(f"  Performance:")
        constants.LOGGER.info(
            f"    Train F1: {_fmt(metrics_on_train, constants.METRIC_NAMES.labeled_f1_doc_macro)} (labeled doc-macro), {_fmt(metrics_on_train, constants.METRIC_NAMES.unlabeled_f1_doc_macro)} (unlabeled doc-macro)"
        )
        constants.LOGGER.info(
            f"    Val F1:   {_fmt(metrics_on_val, constants.METRIC_NAMES.labeled_f1_doc_macro)} (labeled doc-macro), {_fmt(metrics_on_val, constants.METRIC_NAMES.unlabeled_f1_doc_macro)} (unlabeled doc-macro)"
        )
        constants.LOGGER.info(
            f"    Train P/R: {_fmt(metrics_on_train, constants.METRIC_NAMES.labeled_precision_macro)}/{_fmt(metrics_on_train, constants.METRIC_NAMES.labeled_recall_macro)} (labeled), {_fmt(metrics_on_train, constants.METRIC_NAMES.unlabeled_precision_doc_macro)}/{_fmt(metrics_on_train, constants.METRIC_NAMES.unlabeled_recall_doc_macro)} (unlabeled doc-macro)"
        )
        constants.LOGGER.info(
            f"    Val P/R:   {_fmt(metrics_on_val, constants.METRIC_NAMES.labeled_precision_macro)}/{_fmt(metrics_on_val, constants.METRIC_NAMES.labeled_recall_macro)} (labeled), {_fmt(metrics_on_val, constants.METRIC_NAMES.unlabeled_precision_doc_macro)}/{_fmt(metrics_on_val, constants.METRIC_NAMES.unlabeled_recall_doc_macro)} (unlabeled doc-macro)"
        )

    def _write_tensorboard_summary(
        self,
        writer,
        epoch,
        train_loss,
        val_loss,
        train_graph_loss,
        val_graph_loss,
        train_edge_loss,
        val_edge_loss,
        metrics_on_train,
        metrics_on_val,
    ):

        def _m(metrics: Any, key: str, default: float = 0.0) -> float:
            try:
                if not isinstance(metrics, dict):
                    return float(default)
                v = metrics.get(key, default)
                if v is None:
                    return float(default)
                return float(v)
            except Exception:
                return float(default)

        writer.add_scalars("Loss/Total", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars(
            "Loss/Graph", {"train": train_graph_loss, "val": val_graph_loss}, epoch
        )
        writer.add_scalars(
            "Loss/Edge", {"train": train_edge_loss, "val": val_edge_loss}, epoch
        )
        f1_labeled = {
            "val": _m(metrics_on_val, constants.METRIC_NAMES.labeled_f1_doc_macro)
        }
        if isinstance(metrics_on_train, dict):
            f1_labeled["train"] = _m(
                metrics_on_train, constants.METRIC_NAMES.labeled_f1_doc_macro
            )
        writer.add_scalars("F1/LabeledDocMacro", f1_labeled, epoch)
        f1_unl_doc = {
            "val": _m(metrics_on_val, constants.METRIC_NAMES.unlabeled_f1_doc_macro)
        }
        if isinstance(metrics_on_train, dict):
            f1_unl_doc["train"] = _m(
                metrics_on_train, constants.METRIC_NAMES.unlabeled_f1_doc_macro
            )
        writer.add_scalars("F1/UnlabeledDocMacro", f1_unl_doc, epoch)
        prec_labeled = {
            "val": _m(metrics_on_val, constants.METRIC_NAMES.labeled_precision_macro)
        }
        if isinstance(metrics_on_train, dict):
            prec_labeled["train"] = _m(
                metrics_on_train, constants.METRIC_NAMES.labeled_precision_macro
            )
        writer.add_scalars("Precision/Labeled", prec_labeled, epoch)
        prec_unl_doc = {
            "val": _m(
                metrics_on_val, constants.METRIC_NAMES.unlabeled_precision_doc_macro
            )
        }
        if isinstance(metrics_on_train, dict):
            prec_unl_doc["train"] = _m(
                metrics_on_train, constants.METRIC_NAMES.unlabeled_precision_doc_macro
            )
        writer.add_scalars("Precision/UnlabeledDocMacro", prec_unl_doc, epoch)
        rec_labeled = {
            "val": _m(metrics_on_val, constants.METRIC_NAMES.labeled_recall_macro)
        }
        if isinstance(metrics_on_train, dict):
            rec_labeled["train"] = _m(
                metrics_on_train, constants.METRIC_NAMES.labeled_recall_macro
            )
        writer.add_scalars("Recall/Labeled", rec_labeled, epoch)
        rec_unl_doc = {
            "val": _m(metrics_on_val, constants.METRIC_NAMES.unlabeled_recall_doc_macro)
        }
        if isinstance(metrics_on_train, dict):
            rec_unl_doc["train"] = _m(
                metrics_on_train, constants.METRIC_NAMES.unlabeled_recall_doc_macro
            )
        writer.add_scalars("Recall/UnlabeledDocMacro", rec_unl_doc, epoch)

    def _save_final_statistics(
        self,
        dir_current_model,
        misclassified_val,
        misclassified_test,
        metrics_on_val,
        results,
        file_to_write,
        statistics_subdir: str = "final_statistics",
        write_results: bool = True,
        provenance: Any = None,
    ):
        statistics_dir = os.path.join(
            dir_current_model, str(statistics_subdir or "final_statistics")
        )
        training_utils.create_directory_if_not_exists(statistics_dir)
        try:
            pd.DataFrame(misclassified_val or []).to_csv(
                os.path.join(statistics_dir, "misclassified_val.csv"), index=False
            )
            pd.DataFrame(misclassified_test or []).to_csv(
                os.path.join(statistics_dir, "misclassified_test.csv"), index=False
            )
            cm_val = None
            try:
                cm_val = metrics_on_val.get(constants.METRIC_NAMES.confusion, None)
            except Exception:
                try:
                    cm_val = metrics_on_val[constants.METRIC_NAMES.confusion]
                except Exception:
                    cm_val = None
            if cm_val is not None:
                pd.DataFrame(cm_val).to_csv(
                    os.path.join(statistics_dir, "confusion_val.csv"), index=False
                )

            def _m(key: str, default: float = 0.0) -> float:
                try:
                    if isinstance(metrics_on_val, dict):
                        v = metrics_on_val.get(key, default)
                    else:
                        v = default
                    if v is None:
                        return float(default)
                    return float(v)
                except Exception:
                    return float(default)

            precision_recall_metrics = {
                "labeled_precision_micro": _m(
                    constants.METRIC_NAMES.labeled_precision_micro
                ),
                "labeled_precision_macro": _m(
                    constants.METRIC_NAMES.labeled_precision_macro
                ),
                "labeled_recall_micro": _m(constants.METRIC_NAMES.labeled_recall_micro),
                "labeled_recall_macro": _m(constants.METRIC_NAMES.labeled_recall_macro),
                "unlabeled_precision_micro": _m(
                    constants.METRIC_NAMES.unlabeled_precision_micro
                ),
                "unlabeled_precision_doc_macro": _m(
                    constants.METRIC_NAMES.unlabeled_precision_doc_macro
                ),
                "unlabeled_recall_micro": _m(
                    constants.METRIC_NAMES.unlabeled_recall_micro
                ),
                "unlabeled_recall_doc_macro": _m(
                    constants.METRIC_NAMES.unlabeled_recall_doc_macro
                ),
            }
            pd.DataFrame([precision_recall_metrics]).to_csv(
                os.path.join(statistics_dir, "precision_recall_metrics.csv"),
                index=False,
            )
            if provenance is not None:
                try:
                    with open(
                        os.path.join(statistics_dir, "eval_provenance.json"),
                        "w",
                        encoding="utf-8",
                    ) as f:
                        json.dump(provenance, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    constants.LOGGER.warning(
                        f"Could not write eval_provenance.json: {e}"
                    )
        except Exception as e:
            constants.LOGGER.error(f"Error saving statistics: {e}")
        if not write_results:
            return
        if results is None:
            return
        if file_to_write is None:
            file_to_write = os.path.join(
                constants.PROJECT_PATH,
                "logs",
                constants.APP_START_TIMESTAMP,
                "results.csv",
            )
        else:
            file_to_write = os.path.join(
                constants.PROJECT_PATH,
                "logs",
                constants.APP_START_TIMESTAMP,
                file_to_write,
            )
        try:
            if os.path.exists(file_to_write):
                df = pd.read_csv(file_to_write)
                df = pd.concat([df, pd.DataFrame([results])], ignore_index=True)
                df.to_csv(file_to_write, index=False)
            else:
                pd.DataFrame([results]).to_csv(file_to_write, index=False)
            constants.LOGGER.info(f"Results saved to {file_to_write}")
        except Exception as e:
            constants.LOGGER.error(f"Error saving results CSV: {e}")


    def _compute_node_features(
        self, example: Any, apply_dropout: bool = True
    ) -> torch.Tensor:
        try:
            n_windows = int(len(getattr(example, "TOKENS_IDS", []) or []))
            n_tokens = int(getattr(example, "NTOKENS", -1))
            doc_id = int(getattr(example, "DOC_ID", -1))
            try:
                n_nodes = int(
                    len(
                        getattr(
                            getattr(example, "REF_GRAPH", None), "node_attributes", []
                        )
                        or []
                    )
                )
            except Exception:
                n_nodes = -1
            try:
                win_len = int(len(example.TOKENS_IDS[0])) if n_windows > 0 else -1
            except Exception:
                win_len = -1
            thr = int(getattr(self.args, "LOG_LONG_DOC_WINDOWS", 50) or 50)
            if n_windows >= thr and doc_id not in self._warned_long_docs:
                constants.LOGGER.warning(
                    f"[DATA][LONG DOC] doc_id={doc_id} | windows={n_windows} | window_len={win_len} | NTOKENS={n_tokens} | nodes={n_nodes} | encoder={self.encoder.__class__.__name__}"
                )
                self._warned_long_docs.add(doc_id)
        except Exception:
            pass
        subwords_ids = torch.LongTensor(example.TOKENS_IDS).to(self.device)
        bert_embeddings = training_utils.strided_bert_embeddings(
            self.encoder,
            subwords_ids,
            example.TOKENS_SEGMENTATION,
            example.NTOKENS,
            self.device,
        )
        node_features = training_utils.make_node_embeddings(
            example.REF_GRAPH,
            bert_embeddings,
            example,
            self.args.USE_EXTRA_FEATURES,
            self.nodeFeatureAugmenter,
        )
        node_features = self.node_norm(node_features)
        if apply_dropout:
            node_features = self.dropout(node_features)
        return node_features

    def _build_complete_batch(self, batch: Any, device: torch.device) -> dgl.DGLGraph:
        graphs = [
            complete_graph(n, device=device) for n in batch.batch_num_nodes().tolist()
        ]
        return dgl.batch(graphs)


    def get_trainable_parameter_count(self) -> Tuple[int, int]:
        trainable = sum((p.numel() for p in self.parameters() if p.requires_grad))
        total = sum((p.numel() for p in self.parameters()))
        return (trainable, total)


    def _save_best_model(
        self,
        epoch,
        metrics_on_train,
        metrics_on_val,
        now,
        args_string,
        max_val_f1_macro_labeled,
        max_val_f1_macro_unlabeled,
        val_loss,
        min_loss,
        metrics,
        fair_dev_metrics=None,
        prefer_fair=False,
        model_basename=None,
    ):
        if prefer_fair and isinstance(fair_dev_metrics, dict):
            try:
                new_fair = float(
                    fair_dev_metrics.get(
                        constants.METRIC_NAMES.labeled_f1_doc_macro, -1.0
                    )
                )
            except Exception:
                new_fair = -1.0
            best_fair = getattr(self, "_best_fair_dev_doc_macro", None)
            if best_fair is None or new_fair > best_fair:
                try:
                    os.makedirs("models", exist_ok=True)
                    fair_name = (
                        model_basename + "_best_fair.pt"
                        if model_basename
                        else f"{now}{args_string}_best_fair.pt"
                    )
                    fair_path = os.path.join("models", fair_name)
                    torch.save(self.state_dict(), fair_path)
                    setattr(self, "_best_fair_dev_doc_macro", new_fair)
                    setattr(self, "_best_fair_model_path", fair_path)
                    setattr(self, "_best_fair_epoch", int(epoch))
                    constants.LOGGER.info(
                        f"[FAIR] Improved dev labeled F1_doc_macro={new_fair:.4f} at epoch={epoch}; saved checkpoint to {fair_path}"
                    )
                except Exception as e:
                    constants.LOGGER.warning(
                        f"[FAIR] Could not save fair-selected checkpoint: {e}"
                    )
        return (max_val_f1_macro_unlabeled, max_val_f1_macro_labeled, min_loss, metrics)


    def _validate_dimensions(self):
        encoder_dim = self._get_encoder_dimensions()["hidden_size"]
        if self.args.USE_EXTRA_FEATURES:
            expected_dim = encoder_dim + 3 * 8
        else:
            expected_dim = encoder_dim
        assert (
            self.node_embedding_dim == expected_dim
        ), f"Dimension mismatch: expected {expected_dim}, got {self.node_embedding_dim}"
