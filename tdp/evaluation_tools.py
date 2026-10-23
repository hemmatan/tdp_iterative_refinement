"""Document-level parsing metrics, DEV calibration and refinement diagnostics."""

from tdp.constants import METRIC_NAMES
import numpy as np
import pandas as pd
import logging
from statistics import mean, pstdev
from collections import defaultdict
import json
import re
import os
import torch
import torch.nn as nn
import tdp.constants as constants


def save_train_per_iter_metrics(
    parser, output_dir, logger=None, extra_columns=None, epoch=None, append=False
):
    os.makedirs(output_dir, exist_ok=True)
    rows_by_it = {}
    try:
        if hasattr(parser, "_grad_norm_adj_per_iter"):
            for it, vals in sorted(parser._grad_norm_adj_per_iter.items()):
                if not vals:
                    continue
                rows_by_it.setdefault(it, {"iteration": it})
                rows_by_it[it].update(
                    {
                        "grad_norm_adj_pred_mean": float(mean(vals)),
                        "grad_norm_adj_pred_std": (
                            float(pstdev(vals)) if len(vals) > 1 else 0.0
                        ),
                        "grad_norm_adj_pred_count": int(len(vals)),
                    }
                )
    except Exception as e:
        if logger is not None:
            logger.warning(f"Could not add grad norms: {e}")
    try:
        if hasattr(parser, "_msg_self_ratio_per_iter"):
            for it, vals in sorted(parser._msg_self_ratio_per_iter.items()):
                if not vals:
                    continue
                rows_by_it.setdefault(it, {"iteration": it})
                rows_by_it[it].update(
                    {
                        "msg_to_self_ratio_mean": float(mean(vals)),
                        "msg_to_self_ratio_std": (
                            float(pstdev(vals)) if len(vals) > 1 else 0.0
                        ),
                        "msg_to_self_ratio_count": int(len(vals)),
                    }
                )
    except Exception as e:
        if logger is not None:
            logger.warning(f"Could not add msg/self ratios: {e}")
    try:
        if hasattr(parser, "_per_iter_graph_loss"):
            for it, vals in sorted(parser._per_iter_graph_loss.items()):
                if not vals:
                    continue
                rows_by_it.setdefault(it, {"iteration": it})
                rows_by_it[it].update(
                    {
                        "graph_loss_mean": float(mean(vals)),
                        "graph_loss_std": float(pstdev(vals)) if len(vals) > 1 else 0.0,
                        "graph_loss_count": int(len(vals)),
                    }
                )
        if hasattr(parser, "_per_iter_edge_loss"):
            for it, vals in sorted(parser._per_iter_edge_loss.items()):
                if not vals:
                    continue
                rows_by_it.setdefault(it, {"iteration": it})
                rows_by_it[it].update(
                    {
                        "edge_loss_mean": float(mean(vals)),
                        "edge_loss_std": float(pstdev(vals)) if len(vals) > 1 else 0.0,
                        "edge_loss_count": int(len(vals)),
                    }
                )
    except Exception as e:
        if logger is not None:
            logger.warning(f"Could not add per-iteration losses: {e}")
    try:
        if hasattr(parser, "_per_class_edge_loss"):

            def _sanitize(name: str) -> str:
                s = re.sub("[^0-9a-zA-Z]+", "_", str(name).strip()).strip("_").lower()
                if s == "" or s[0].isdigit():
                    s = f"label_{s}"
                return s

            id2name = None
            try:
                if hasattr(parser, "dep_vocab") and hasattr(
                    parser.dep_vocab, "idx2sym"
                ):
                    id2name = list(parser.dep_vocab.idx2sym)
            except Exception:
                id2name = None
            for it, cls_map in sorted(parser._per_class_edge_loss.items()):
                if not cls_map:
                    continue
                rows_by_it.setdefault(it, {"iteration": it})
                for cls_id, vals in sorted(cls_map.items(), key=lambda kv: int(kv[0])):
                    if not vals:
                        continue
                    mean_val = float(mean(vals))
                    cnt_val = int(len(vals))
                    wrote_name = False
                    try:
                        if id2name is not None and int(cls_id) < len(id2name):
                            cname = _sanitize(id2name[int(cls_id)])
                            rows_by_it[it][f"class_{cname}_edge_loss_mean"] = mean_val
                            rows_by_it[it][f"class_{cname}_edge_loss_count"] = cnt_val
                            wrote_name = True
                    except Exception:
                        wrote_name = False
                    if not wrote_name:
                        rows_by_it[it][f"class_{int(cls_id)}_edge_loss_mean"] = mean_val
                        rows_by_it[it][f"class_{int(cls_id)}_edge_loss_count"] = cnt_val
    except Exception as e:
        if logger is not None:
            logger.warning(f"Could not add per-class edge losses: {e}")
    try:
        if hasattr(parser, "_grad_norm_module"):
            enc_vals = parser._grad_norm_module.get("encoder", [])
            nonenc_vals = parser._grad_norm_module.get("non_encoder", [])
            if enc_vals:
                for it in sorted(rows_by_it):
                    rows_by_it[it].update(
                        {
                            "grad_norm_encoder_mean": float(mean(enc_vals)),
                            "grad_norm_encoder_std": (
                                float(pstdev(enc_vals)) if len(enc_vals) > 1 else 0.0
                            ),
                            "grad_norm_encoder_count": int(len(enc_vals)),
                        }
                    )
            if nonenc_vals:
                for it in sorted(rows_by_it):
                    rows_by_it[it].update(
                        {
                            "grad_norm_nonencoder_mean": float(mean(nonenc_vals)),
                            "grad_norm_nonencoder_std": (
                                float(pstdev(nonenc_vals))
                                if len(nonenc_vals) > 1
                                else 0.0
                            ),
                            "grad_norm_nonencoder_count": int(len(nonenc_vals)),
                        }
                    )
    except Exception as e:
        if logger is not None:
            logger.warning(f"Could not add module grad norms: {e}")
    df = pd.DataFrame([rows_by_it[it] for it in sorted(rows_by_it)])
    if epoch is not None:
        df["epoch"] = int(epoch)
        cols = ["epoch"] + [c for c in df.columns if c != "epoch"]
        df = df[cols]
    if extra_columns:
        try:

            def _to_jsonable(obj):
                if obj is None or isinstance(obj, (str, int, float, bool)):
                    return obj
                try:
                    import numpy as _np

                    if isinstance(obj, _np.integer):
                        return int(obj)
                    if isinstance(obj, _np.floating):
                        return float(obj)
                    if isinstance(obj, _np.ndarray):
                        return obj.tolist()
                except Exception:
                    pass
                try:
                    import torch

                    if isinstance(obj, torch.Tensor):
                        t = obj.detach().cpu()
                        return t.item() if t.numel() == 1 else t.tolist()
                except Exception:
                    pass
                if isinstance(obj, dict):
                    return {str(k): _to_jsonable(v) for k, v in obj.items()}
                if isinstance(obj, (list, tuple, set)):
                    return [_to_jsonable(v) for v in obj]
                try:
                    return float(obj)
                except Exception:
                    return str(obj)

            payload = _to_jsonable(extra_columns)
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, "hyperparams.json")
            tmp_path = out_path + ".tmp"
            try:
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2, ensure_ascii=False)
                os.replace(tmp_path, out_path)
            finally:
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except Exception:
                    pass
        except Exception as e:
            if logger is not None:
                logger.warning(f"Could not write hyperparams.json: {e}")
    csv_path = os.path.join(output_dir, "train_per_iter_slice_metrics.csv")
    if append and os.path.exists(csv_path):
        try:
            old = pd.read_csv(csv_path)
            df = pd.concat([old, df], ignore_index=True)
        except Exception:
            pass
    df.to_csv(csv_path, index=False)
    if logger is not None:
        logger.info(
            f"Saved train per-iteration metrics to {os.path.join(output_dir, 'train_per_iter_slice_metrics.csv')}"
        )
    return df


def generate_gradient_flow_report(
    model, out_dir=None, filename="gradient_flow.txt", logger=None
):
    total_params = 0
    params_with_grad = 0
    zero_grad_params = 0
    total_grad_sq_sum = 0.0
    any_nan = False

    def _accumulate_params(named_params):
        nonlocal total_params, params_with_grad, zero_grad_params, total_grad_sq_sum, any_nan
        for _, p in named_params:
            total_params += 1
            if p.grad is None:
                continue
            params_with_grad += 1
            g = p.grad.detach()
            if g.numel() == 0 or float(g.abs().sum().item()) == 0.0:
                zero_grad_params += 1
            total_grad_sq_sum += float(g.pow(2).sum().item())
            if not torch.isfinite(g).all():
                any_nan = True

    try:
        _accumulate_params(model.named_parameters())
    except Exception:
        pass
    total_grad_norm = total_grad_sq_sum**0.5 if total_grad_sq_sum > 0.0 else 0.0
    enc_norm = 0.0
    nonenc_norm = 0.0
    try:
        if hasattr(model, "encoder"):
            enc_sq = 0.0
            for p in model.encoder.parameters():
                if p.grad is None:
                    continue
                enc_sq += float(p.grad.detach().pow(2).sum().item())
            enc_norm = enc_sq**0.5 if enc_sq > 0.0 else 0.0
        nonenc_sq = 0.0
        for n, p in model.named_parameters():
            if n.startswith("encoder."):
                continue
            if p.grad is None:
                continue
            nonenc_sq += float(p.grad.detach().pow(2).sum().item())
        nonenc_norm = nonenc_sq**0.5 if nonenc_sq > 0.0 else 0.0
    except Exception:
        pass
    per_iter_adj = None
    try:
        if hasattr(model, "_grad_norm_adj_per_iter") and model._grad_norm_adj_per_iter:
            from statistics import mean

            per_iter_adj = {
                it: float(mean(vals))
                for it, vals in sorted(model._grad_norm_adj_per_iter.items())
                if vals
            }
    except Exception:
        per_iter_adj = None

    def _fmt(val, fmt="{:.4f}"):
        try:
            return fmt.format(val)
        except Exception:
            return str(val)

    report_lines = []
    report_lines.append("Gradient Flow Report")
    report_lines.append("---------------------")
    report_lines.append(f"Total parameters: {total_params}")
    report_lines.append(f"With grad: {params_with_grad}")
    report_lines.append(
        f"Zero-grad (with grad tensor but all zeros): {zero_grad_params}"
    )
    report_lines.append(f"Total grad L2 norm: {_fmt(total_grad_norm)}")
    report_lines.append("")
    report_lines.append("Module grad norms:")
    report_lines.append(f"  encoder:     {_fmt(enc_norm)}")
    report_lines.append(f"  non-encoder: {_fmt(nonenc_norm)}")
    if per_iter_adj is not None:
        report_lines.append("")
        report_lines.append(
            "Adjacency predictor grad norms (per iteration, mean over batches):"
        )
        for it, v in per_iter_adj.items():
            report_lines.append(f"  it={it}: {_fmt(v)}")
    if any_nan:
        report_lines.append("")
        report_lines.append(
            "Warning: Non-finite gradients detected (NaN/Inf). Consider anomaly detection."
        )
    flowing = total_grad_norm > 0.0 and params_with_grad > 0
    report_lines.append("")
    report_lines.append(
        f"Verdict: {('GRADIENTS FLOWING' if flowing else 'NO GRADIENT FLOW DETECTED')}"
    )
    report = "\n".join(report_lines)
    if out_dir is not None:
        try:
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, filename), "w") as f:
                f.write(report)
        except Exception:
            if logger is not None:
                logger.warning("Could not write gradient flow report to disk")
    if logger is not None:
        logger.info("Generated gradient flow report")
    return report


def turn_predictions_into_labeled_tuples(predicted_graph, ref_graph, edge_vocab):
    src, tgt = ref_graph.edges()
    ref_src, ref_tgt, ref_depids = (
        src.tolist(),
        tgt.tolist(),
        ref_graph.edata["_depID"].tolist(),
    )
    src, tgt = predicted_graph.edges()
    pred_src, pred_tgt, pred_depids = (
        src.tolist(),
        tgt.tolist(),
        predicted_graph.edata["_depID"].tolist(),
    )
    ref_tuples = zip(ref_src, ref_tgt, ref_depids)
    pred_tuples = zip(pred_src, pred_tgt, pred_depids)
    return (set(ref_tuples), set(pred_tuples))


def turn_predictions_into_unlabeled_tuples(predicted_graph, ref_graph, edge_vocab):
    src, tgt = ref_graph.edges()
    ref_src, ref_tgt = (src.tolist(), tgt.tolist())
    src, tgt = predicted_graph.edges()
    pred_src, pred_tgt = (src.tolist(), tgt.tolist())
    ref_tuples = zip(ref_src, ref_tgt)
    pred_tuples = zip(pred_src, pred_tgt)
    return (set(ref_tuples), set(pred_tuples))


_DEFAULT_STRUCTURAL_LABELS_LOWER = {"depend-on"}


def _get_structural_label_ids(edge_vocab, structural_labels_lower=None):
    names = getattr(edge_vocab, "idx2sym", []) or []
    want = set(structural_labels_lower or _DEFAULT_STRUCTURAL_LABELS_LOWER)
    out = set()
    for idx, name in enumerate(names):
        try:
            if str(name).lower() in want:
                out.add(int(idx))
        except Exception:
            continue
    return out


def _edge_dict_from_graph(graph, exclude_label_ids=None):
    exclude = set(exclude_label_ids or [])
    src, tgt = graph.edges()
    labels = graph.edata["_depID"].tolist()
    d = {}
    for s, t, lbl in zip(src.tolist(), tgt.tolist(), labels):
        lbl = int(lbl)
        if exclude and lbl in exclude:
            continue
        d[int(s), int(t)] = lbl
    return d


def _compute_labeled_per_class_counts(
    gold_edges: dict, pred_edges: dict, num_labels: int
):
    tp = [0] * num_labels
    fp = [0] * num_labels
    fn = [0] * num_labels
    for e, pl in pred_edges.items():
        gl = gold_edges.get(e)
        if gl is None:
            fp[pl] += 1
        elif pl == gl:
            tp[pl] += 1
        else:
            fp[pl] += 1
            fn[gl] += 1
    for e, gl in gold_edges.items():
        if e not in pred_edges:
            fn[gl] += 1
    return (tp, fp, fn)


def _compute_labeled_micro_counts(gold_edges: dict, pred_edges: dict):
    TP = FP = FN = 0
    for e, pl in pred_edges.items():
        gl = gold_edges.get(e)
        if gl is None:
            FP += 1
        elif pl == gl:
            TP += 1
        else:
            FP += 1
            FN += 1
    for e in gold_edges.keys():
        if e not in pred_edges:
            FN += 1
    return (TP, FP, FN)


def _compute_unlabeled_counts(gold_edge_pairs: set, pred_edge_pairs: set):
    TP = len(gold_edge_pairs & pred_edge_pairs)
    FP = len(pred_edge_pairs - gold_edge_pairs)
    FN = len(gold_edge_pairs - pred_edge_pairs)
    return (TP, FP, FN)


def _macro_prf_over_labels(tp, fp, fn, label_ids, only_gold_support: bool = True):
    rows = []
    for lid in label_ids:
        lid = int(lid)
        gold_support = tp[lid] + fn[lid]
        if only_gold_support and gold_support == 0:
            continue
        P = calculate_precision(tp[lid], fp[lid])
        R = calculate_recall(tp[lid], fn[lid])
        F1 = calculate_f1_from_tp_fp_fn(tp[lid], fp[lid], fn[lid])
        rows.append((P, R, F1))
    if not rows:
        return (0.0, 0.0, 0.0)
    p_macro = sum((r[0] for r in rows)) / len(rows)
    r_macro = sum((r[1] for r in rows)) / len(rows)
    f_macro = sum((r[2] for r in rows)) / len(rows)
    return (float(p_macro), float(r_macro), float(f_macro))


def evaluate_predicted_graphs(
    predicted_graphs,
    data_loader,
    save_statistics=True,
    compute_confusion: bool = False,
    include_diagonal: bool = True,
):
    """Compute micro, document-macro and label-macro metrics separately.

    Document-macro F1 averages the F1 scores of individual documents."""
    overall_metrics = {
        METRIC_NAMES.unlabeled_tp: 0,
        METRIC_NAMES.unlabeled_fp: 0,
        METRIC_NAMES.unlabeled_fn: 0,
    }
    cm = None
    cmu = None
    if compute_confusion:
        cm = display_confusion_matrix_labeled(
            predicted_graphs, data_loader, include_diagonal=include_diagonal
        )
        cmu = display_confusion_matrix_unlabeled(
            predicted_graphs, data_loader, include_diagonal=include_diagonal
        )
    if save_statistics:
        misclassified_edges_properties = extract_misclassified_edges_properties(
            predicted_graphs, data_loader
        )
    else:
        misclassified_edges_properties = None
    edge_vocab = data_loader.edge_vocabulary
    num_labels = len(getattr(edge_vocab, "idx2sym", []))
    structural_ids = _get_structural_label_ids(edge_vocab)
    tp_all = [0] * num_labels
    fp_all = [0] * num_labels
    fn_all = [0] * num_labels
    tp_tmp = [0] * num_labels
    fp_tmp = [0] * num_labels
    fn_tmp = [0] * num_labels
    micro_lab_all = {"TP": 0, "FP": 0, "FN": 0}
    micro_lab_tmp = {"TP": 0, "FP": 0, "FN": 0}
    micro_un_all = {"TP": 0, "FP": 0, "FN": 0}
    micro_un_tmp = {"TP": 0, "FP": 0, "FN": 0}
    doc_scores_labeled_all = []
    doc_scores_labeled_tmp = []
    doc_scores_unlabeled_all = []
    doc_scores_unlabeled_tmp = []
    for idx, predicted_graph in enumerate(predicted_graphs):
        ref_graph = data_loader.graphs[idx]
        gold_all = _edge_dict_from_graph(ref_graph, exclude_label_ids=None)
        pred_all = _edge_dict_from_graph(predicted_graph, exclude_label_ids=None)
        gold_tmp = _edge_dict_from_graph(ref_graph, exclude_label_ids=structural_ids)
        pred_tmp = _edge_dict_from_graph(
            predicted_graph, exclude_label_ids=structural_ids
        )
        TP, FP, FN = _compute_labeled_micro_counts(gold_all, pred_all)
        micro_lab_all["TP"] += TP
        micro_lab_all["FP"] += FP
        micro_lab_all["FN"] += FN
        doc_scores_labeled_all.append(
            (
                calculate_precision(TP, FP),
                calculate_recall(TP, FN),
                calculate_f1_from_tp_fp_fn(TP, FP, FN),
            )
        )
        TP, FP, FN = _compute_labeled_micro_counts(gold_tmp, pred_tmp)
        micro_lab_tmp["TP"] += TP
        micro_lab_tmp["FP"] += FP
        micro_lab_tmp["FN"] += FN
        doc_scores_labeled_tmp.append(
            (
                calculate_precision(TP, FP),
                calculate_recall(TP, FN),
                calculate_f1_from_tp_fp_fn(TP, FP, FN),
            )
        )
        tpi, fpi, fni = _compute_labeled_per_class_counts(
            gold_all, pred_all, num_labels
        )
        for j in range(num_labels):
            tp_all[j] += tpi[j]
            fp_all[j] += fpi[j]
            fn_all[j] += fni[j]
        tpi, fpi, fni = _compute_labeled_per_class_counts(
            gold_tmp, pred_tmp, num_labels
        )
        for j in range(num_labels):
            tp_tmp[j] += tpi[j]
            fp_tmp[j] += fpi[j]
            fn_tmp[j] += fni[j]
        gold_edges_all = set(gold_all.keys())
        pred_edges_all = set(pred_all.keys())
        TP, FP, FN = _compute_unlabeled_counts(gold_edges_all, pred_edges_all)
        micro_un_all["TP"] += TP
        micro_un_all["FP"] += FP
        micro_un_all["FN"] += FN
        doc_scores_unlabeled_all.append(
            (
                calculate_precision(TP, FP),
                calculate_recall(TP, FN),
                calculate_f1_from_tp_fp_fn(TP, FP, FN),
            )
        )
        gold_edges_tmp = set(gold_tmp.keys())
        pred_edges_tmp = set(pred_tmp.keys())
        TP, FP, FN = _compute_unlabeled_counts(gold_edges_tmp, pred_edges_tmp)
        micro_un_tmp["TP"] += TP
        micro_un_tmp["FP"] += FP
        micro_un_tmp["FN"] += FN
        doc_scores_unlabeled_tmp.append(
            (
                calculate_precision(TP, FP),
                calculate_recall(TP, FN),
                calculate_f1_from_tp_fp_fn(TP, FP, FN),
            )
        )
    micro_labeled_precision_all = calculate_precision(
        micro_lab_all["TP"], micro_lab_all["FP"]
    )
    micro_labeled_recall_all = calculate_recall(
        micro_lab_all["TP"], micro_lab_all["FN"]
    )
    micro_labeled_f1_all = calculate_f1_from_tp_fp_fn(
        micro_lab_all["TP"], micro_lab_all["FP"], micro_lab_all["FN"]
    )
    micro_labeled_precision_tmp = calculate_precision(
        micro_lab_tmp["TP"], micro_lab_tmp["FP"]
    )
    micro_labeled_recall_tmp = calculate_recall(
        micro_lab_tmp["TP"], micro_lab_tmp["FN"]
    )
    micro_labeled_f1_tmp = calculate_f1_from_tp_fp_fn(
        micro_lab_tmp["TP"], micro_lab_tmp["FP"], micro_lab_tmp["FN"]
    )
    micro_unlabeled_precision_all = calculate_precision(
        micro_un_all["TP"], micro_un_all["FP"]
    )
    micro_unlabeled_recall_all = calculate_recall(
        micro_un_all["TP"], micro_un_all["FN"]
    )
    micro_unlabeled_f1_all = calculate_f1_from_tp_fp_fn(
        micro_un_all["TP"], micro_un_all["FP"], micro_un_all["FN"]
    )
    micro_unlabeled_precision_tmp = calculate_precision(
        micro_un_tmp["TP"], micro_un_tmp["FP"]
    )
    micro_unlabeled_recall_tmp = calculate_recall(
        micro_un_tmp["TP"], micro_un_tmp["FN"]
    )
    micro_unlabeled_f1_tmp = calculate_f1_from_tp_fp_fn(
        micro_un_tmp["TP"], micro_un_tmp["FP"], micro_un_tmp["FN"]
    )

    def _avg(scores, idx):
        return float(sum((s[idx] for s in scores)) / max(1, len(scores)))

    p_doc_macro_labeled_all = _avg(doc_scores_labeled_all, 0)
    r_doc_macro_labeled_all = _avg(doc_scores_labeled_all, 1)
    f_doc_macro_labeled_all = _avg(doc_scores_labeled_all, 2)
    p_doc_macro_labeled_tmp = _avg(doc_scores_labeled_tmp, 0)
    r_doc_macro_labeled_tmp = _avg(doc_scores_labeled_tmp, 1)
    f_doc_macro_labeled_tmp = _avg(doc_scores_labeled_tmp, 2)
    p_doc_macro_unlabeled_all = _avg(doc_scores_unlabeled_all, 0)
    r_doc_macro_unlabeled_all = _avg(doc_scores_unlabeled_all, 1)
    f_doc_macro_unlabeled_all = _avg(doc_scores_unlabeled_all, 2)
    p_doc_macro_unlabeled_tmp = _avg(doc_scores_unlabeled_tmp, 0)
    r_doc_macro_unlabeled_tmp = _avg(doc_scores_unlabeled_tmp, 1)
    f_doc_macro_unlabeled_tmp = _avg(doc_scores_unlabeled_tmp, 2)
    label_ids_all = list(range(num_labels))
    label_ids_tmp = [i for i in range(num_labels) if i not in structural_ids]
    p_macro_labeled_all, r_macro_labeled_all, f_macro_labeled_all = (
        _macro_prf_over_labels(
            tp_all, fp_all, fn_all, label_ids_all, only_gold_support=True
        )
    )
    p_macro_labeled_tmp, r_macro_labeled_tmp, f_macro_labeled_tmp = (
        _macro_prf_over_labels(
            tp_tmp, fp_tmp, fn_tmp, label_ids_tmp, only_gold_support=True
        )
    )
    overall_metrics[METRIC_NAMES.labeled_f1_micro_all] = micro_labeled_f1_all
    overall_metrics[METRIC_NAMES.labeled_f1_macro_all] = f_macro_labeled_all
    overall_metrics[METRIC_NAMES.labeled_f1_doc_macro_all] = f_doc_macro_labeled_all
    overall_metrics[METRIC_NAMES.labeled_precision_micro_all] = (
        micro_labeled_precision_all
    )
    overall_metrics[METRIC_NAMES.labeled_precision_macro_all] = p_macro_labeled_all
    overall_metrics[METRIC_NAMES.labeled_recall_micro_all] = micro_labeled_recall_all
    overall_metrics[METRIC_NAMES.labeled_recall_macro_all] = r_macro_labeled_all
    overall_metrics[METRIC_NAMES.unlabeled_f1_micro_all] = micro_unlabeled_f1_all
    overall_metrics[METRIC_NAMES.unlabeled_f1_macro_all] = f_doc_macro_unlabeled_all
    overall_metrics[METRIC_NAMES.unlabeled_f1_doc_macro_all] = f_doc_macro_unlabeled_all
    overall_metrics[METRIC_NAMES.unlabeled_precision_micro_all] = (
        micro_unlabeled_precision_all
    )
    overall_metrics[METRIC_NAMES.unlabeled_precision_macro_all] = (
        p_doc_macro_unlabeled_all
    )
    overall_metrics[
        getattr(
            METRIC_NAMES,
            "unlabeled_precision_doc_macro_all",
            "unlabeled_precision_doc_macro_all",
        )
    ] = p_doc_macro_unlabeled_all
    overall_metrics[METRIC_NAMES.unlabeled_recall_micro_all] = (
        micro_unlabeled_recall_all
    )
    overall_metrics[METRIC_NAMES.unlabeled_recall_macro_all] = r_doc_macro_unlabeled_all
    overall_metrics[
        getattr(
            METRIC_NAMES,
            "unlabeled_recall_doc_macro_all",
            "unlabeled_recall_doc_macro_all",
        )
    ] = r_doc_macro_unlabeled_all
    overall_metrics[METRIC_NAMES.labeled_f1_micro_temporal] = micro_labeled_f1_tmp
    overall_metrics[METRIC_NAMES.labeled_f1_macro_temporal] = f_macro_labeled_tmp
    overall_metrics[METRIC_NAMES.labeled_f1_doc_macro_temporal] = (
        f_doc_macro_labeled_tmp
    )
    overall_metrics[METRIC_NAMES.labeled_precision_micro_temporal] = (
        micro_labeled_precision_tmp
    )
    overall_metrics[METRIC_NAMES.labeled_precision_macro_temporal] = p_macro_labeled_tmp
    overall_metrics[METRIC_NAMES.labeled_recall_micro_temporal] = (
        micro_labeled_recall_tmp
    )
    overall_metrics[METRIC_NAMES.labeled_recall_macro_temporal] = r_macro_labeled_tmp
    overall_metrics[METRIC_NAMES.unlabeled_f1_micro_temporal] = micro_unlabeled_f1_tmp
    overall_metrics[METRIC_NAMES.unlabeled_f1_macro_temporal] = (
        f_doc_macro_unlabeled_tmp
    )
    overall_metrics[METRIC_NAMES.unlabeled_f1_doc_macro_temporal] = (
        f_doc_macro_unlabeled_tmp
    )
    overall_metrics[METRIC_NAMES.unlabeled_precision_micro_temporal] = (
        micro_unlabeled_precision_tmp
    )
    overall_metrics[METRIC_NAMES.unlabeled_precision_macro_temporal] = (
        p_doc_macro_unlabeled_tmp
    )
    overall_metrics[
        getattr(
            METRIC_NAMES,
            "unlabeled_precision_doc_macro_temporal",
            "unlabeled_precision_doc_macro_temporal",
        )
    ] = p_doc_macro_unlabeled_tmp
    overall_metrics[METRIC_NAMES.unlabeled_recall_micro_temporal] = (
        micro_unlabeled_recall_tmp
    )
    overall_metrics[METRIC_NAMES.unlabeled_recall_macro_temporal] = (
        r_doc_macro_unlabeled_tmp
    )
    overall_metrics[
        getattr(
            METRIC_NAMES,
            "unlabeled_recall_doc_macro_temporal",
            "unlabeled_recall_doc_macro_temporal",
        )
    ] = r_doc_macro_unlabeled_tmp
    overall_metrics[METRIC_NAMES.labeled_f1_micro] = micro_labeled_f1_all
    overall_metrics[METRIC_NAMES.labeled_f1_macro] = f_macro_labeled_all
    overall_metrics[METRIC_NAMES.labeled_f1_doc_macro] = f_doc_macro_labeled_all
    overall_metrics[METRIC_NAMES.unlabeled_f1_micro] = micro_unlabeled_f1_all
    overall_metrics[METRIC_NAMES.unlabeled_f1_macro] = f_doc_macro_unlabeled_all
    overall_metrics[METRIC_NAMES.unlabeled_f1_doc_macro] = f_doc_macro_unlabeled_all
    overall_metrics[METRIC_NAMES.confusion] = cm
    overall_metrics[METRIC_NAMES.confusion_unlabeled] = cmu
    overall_metrics[METRIC_NAMES.labeled_precision_micro] = micro_labeled_precision_all
    overall_metrics[METRIC_NAMES.labeled_precision_macro] = p_macro_labeled_all
    overall_metrics[METRIC_NAMES.labeled_recall_micro] = micro_labeled_recall_all
    overall_metrics[METRIC_NAMES.labeled_recall_macro] = r_macro_labeled_all
    overall_metrics[METRIC_NAMES.unlabeled_precision_micro] = (
        micro_unlabeled_precision_all
    )
    overall_metrics[METRIC_NAMES.unlabeled_precision_macro] = p_doc_macro_unlabeled_all
    overall_metrics[
        getattr(
            METRIC_NAMES,
            "unlabeled_precision_doc_macro",
            "unlabeled_precision_doc_macro",
        )
    ] = p_doc_macro_unlabeled_all
    overall_metrics[METRIC_NAMES.unlabeled_recall_micro] = micro_unlabeled_recall_all
    overall_metrics[METRIC_NAMES.unlabeled_recall_macro] = r_doc_macro_unlabeled_all
    overall_metrics[
        getattr(
            METRIC_NAMES, "unlabeled_recall_doc_macro", "unlabeled_recall_doc_macro"
        )
    ] = r_doc_macro_unlabeled_all
    overall_metrics[METRIC_NAMES.unlabeled_tp] = micro_un_all["TP"]
    overall_metrics[METRIC_NAMES.unlabeled_fp] = micro_un_all["FP"]
    overall_metrics[METRIC_NAMES.unlabeled_fn] = micro_un_all["FN"]
    try:
        logging.info(
            "[EVAL][all] labeled micro F1=%.3f | labeled label-macro F1=%.3f | labeled doc-macro F1=%.3f",
            micro_labeled_f1_all,
            f_macro_labeled_all,
            f_doc_macro_labeled_all,
        )
        logging.info(
            "[EVAL][temporal] labeled micro F1=%.3f | labeled label-macro F1=%.3f | labeled doc-macro F1=%.3f",
            micro_labeled_f1_tmp,
            f_macro_labeled_tmp,
            f_doc_macro_labeled_tmp,
        )
    except Exception:
        pass
    return (overall_metrics, misclassified_edges_properties)


def predict_and_evaluate(
    parser,
    data_loader,
    batch_size,
    save_statistics=True,
    collect_metrics=False,
    compute_confusion: bool = False,
):
    predications = parser.predict(
        data_loader, batch_size=batch_size, collect_metrics=collect_metrics
    )
    predicted_graphs = [graph for graph, _ in predications]
    overall_metrics, misclassified_edges_properties = evaluate_predicted_graphs(
        predicted_graphs,
        data_loader,
        save_statistics=save_statistics,
        compute_confusion=compute_confusion,
    )
    return (overall_metrics, misclassified_edges_properties)


def display_confusion_matrix_unlabeled(
    predicted_graphs, data_loader, include_diagonal: bool = True
):
    labels = ["EDGE", "NO_EDGE"]
    TP_tot = FP_tot = FN_tot = TN_tot = 0
    for predicted_graph, ref_graph in zip(predicted_graphs, data_loader.graphs):
        n = int(ref_graph.num_nodes())
        src_ref, tgt_ref = ref_graph.edges()
        edges_ref = set(zip(src_ref.tolist(), tgt_ref.tolist()))
        src_pred, tgt_pred = predicted_graph.edges()
        edges_pred = set(zip(src_pred.tolist(), tgt_pred.tolist()))
        if not include_diagonal:
            edges_ref = {e for e in edges_ref if e[0] != e[1]}
            edges_pred = {e for e in edges_pred if e[0] != e[1]}
            total_pairs = n * (n - 1)
        else:
            total_pairs = n * n
        TP = len(edges_ref & edges_pred)
        FP = len(edges_pred - edges_ref)
        FN = len(edges_ref - edges_pred)
        TN = int(total_pairs - TP - FP - FN)
        TP_tot += TP
        FP_tot += FP
        FN_tot += FN
        TN_tot += TN
    cm = np.array([[TP_tot, FN_tot], [FP_tot, TN_tot]], dtype=int)
    return pd.DataFrame(cm, index=labels, columns=labels)


def display_confusion_matrix_labeled(
    predicted_graphs, data_loader, include_diagonal: bool = True
):
    label_names = list(getattr(data_loader.edge_vocabulary, "idx2sym", []))
    num_labels = len(label_names)
    no_edge_id = num_labels
    labels = label_names + ["NO_EDGE"]
    cm = np.zeros((num_labels + 1, num_labels + 1), dtype=int)
    for predicted_graph, ref_graph in zip(predicted_graphs, data_loader.graphs):
        n = int(ref_graph.num_nodes())
        total_pairs = n * n if include_diagonal else n * (n - 1)
        gold = _edge_dict_from_graph(ref_graph, exclude_label_ids=None)
        pred = _edge_dict_from_graph(predicted_graph, exclude_label_ids=None)
        if not include_diagonal:
            gold = {e: l for e, l in gold.items() if e[0] != e[1]}
            pred = {e: l for e, l in pred.items() if e[0] != e[1]}
        union_edges = set(gold.keys()) | set(pred.keys())
        for e in union_edges:
            tl = gold.get(e, no_edge_id)
            pl = pred.get(e, no_edge_id)
            try:
                tl = int(tl)
            except Exception:
                tl = no_edge_id
            try:
                pl = int(pl)
            except Exception:
                pl = no_edge_id
            if tl < 0 or tl > no_edge_id:
                tl = no_edge_id
            if pl < 0 or pl > no_edge_id:
                pl = no_edge_id
            cm[tl, pl] += 1
        rest = int(total_pairs - len(union_edges))
        if rest > 0:
            cm[no_edge_id, no_edge_id] += rest
    return pd.DataFrame(cm, index=labels, columns=labels)


def extract_misclassified_edges_properties(predicted_graphs, data_loader):
    misclassified_edges_properties = []
    label_names = list(getattr(data_loader.edge_vocabulary, "idx2sym", []))
    no_edge_id = len(label_names)
    label_names = label_names + ["NO_EDGE"]
    reference_graphs = data_loader.graphs
    for idx, (predicted_graph, ref_graph) in enumerate(
        zip(predicted_graphs, reference_graphs)
    ):
        edge_labels_ref = _edge_dict_from_graph(ref_graph, exclude_label_ids=None)
        edge_labels_pred = _edge_dict_from_graph(
            predicted_graph, exclude_label_ids=None
        )
        union_edges = set(edge_labels_ref.keys()) | set(edge_labels_pred.keys())
        for edge in union_edges:
            true_label = int(edge_labels_ref.get(edge, no_edge_id))
            pred_label = int(edge_labels_pred.get(edge, no_edge_id))
            if true_label != pred_label:
                node_h_str = data_loader.example_loader.dataset[idx].node_id2string(
                    edge[0]
                )
                node_d_str = data_loader.example_loader.dataset[idx].node_id2string(
                    edge[1]
                )
                misclassified_edges_properties.append(
                    {
                        "edge": edge,
                        "true_label": (
                            label_names[true_label]
                            if 0 <= true_label < len(label_names)
                            else "NO_EDGE"
                        ),
                        "pred_label": (
                            label_names[pred_label]
                            if 0 <= pred_label < len(label_names)
                            else "NO_EDGE"
                        ),
                        "head": node_h_str,
                        "dependent": node_d_str,
                        "head_type": data_loader.example_loader.dataset[
                            idx
                        ].node_id2type(edge[0]),
                        "dependent_type": data_loader.example_loader.dataset[
                            idx
                        ].node_id2type(edge[1]),
                        "head_sentence_no": data_loader.example_loader.dataset[
                            idx
                        ].node_id2sentence_number(edge[0]),
                        "dependent_sentence_no": data_loader.example_loader.dataset[
                            idx
                        ].node_id2sentence_number(edge[1]),
                        "sentence_distance": data_loader.example_loader.dataset[
                            idx
                        ].node_id2sentence_number(edge[1])
                        - data_loader.example_loader.dataset[
                            idx
                        ].node_id2sentence_number(edge[0]),
                        "head_no_of_tokens": data_loader.example_loader.dataset[
                            idx
                        ].node_id2no_of_tokens(edge[0]),
                        "dependent_no_of_tokens": data_loader.example_loader.dataset[
                            idx
                        ].node_id2no_of_tokens(edge[1]),
                        "context": data_loader.example_loader.dataset[
                            idx
                        ].get_context_between(
                            data_loader.example_loader.dataset[
                                idx
                            ].node_id2sentence_number(edge[0]),
                            data_loader.example_loader.dataset[
                                idx
                            ].node_id2sentence_number(edge[1]),
                        ),
                        "doc_id": idx,
                    }
                )
    return misclassified_edges_properties


def calculate_precision(TP, FP):
    if TP + FP == 0:
        return 0
    precision = TP / (TP + FP)
    return precision


def calculate_recall(TP, FN):
    if TP + FN == 0:
        return 0
    recall = TP / (TP + FN)
    return recall


def calculate_f1_from_tp_fp_fn(TP, FP, FN):
    precision = calculate_precision(TP, FP)
    recall = calculate_recall(TP, FN)
    if precision + recall == 0:
        return 0
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def _compute_iterative_edge_loss(
    edge_preds, labels, loss_function, agg, edge_loss_weights
):
    per_iter_edge_losses = []
    try:
        for i in range(len(edge_preds)):
            per_iter_edge_losses.append(loss_function(edge_preds[i], labels))
    except Exception:
        per_iter_edge_losses = []
    if per_iter_edge_losses:
        if agg == 'mean':
            edge_loss = sum(per_iter_edge_losses) / len(per_iter_edge_losses)
        elif agg == 'weighted_mean':
            T = len(per_iter_edge_losses)
            if not edge_loss_weights or len(edge_loss_weights) != T:
                edge_loss_weights = [i + 1 for i in range(T)]
            ws = torch.tensor(
                edge_loss_weights,
                dtype=per_iter_edge_losses[0].dtype,
                device=per_iter_edge_losses[0].device,
            )
            ws = ws / ws.sum()
            edge_loss = sum((li * wi for li, wi in zip(per_iter_edge_losses, ws)))
        else:
            edge_loss = per_iter_edge_losses[-1]
    else:
        edge_loss = loss_function(edge_preds[-1], labels)
    return (edge_loss, per_iter_edge_losses)


def _compute_iterative_graph_loss(
    adj_logits_list,
    adj_probs_list,
    ref_matrix,
    graph_loss_function,
    agg="mean",
    graph_loss_weights=None,
):
    per_iter_graph_losses = []
    try:
        if isinstance(graph_loss_function, nn.BCELoss):
            for probs_i in adj_probs_list:
                per_iter_graph_losses.append(graph_loss_function(probs_i, ref_matrix))
        else:
            for logits_i in adj_logits_list:
                per_iter_graph_losses.append(graph_loss_function(logits_i, ref_matrix))
    except Exception:
        per_iter_graph_losses = []
    if per_iter_graph_losses:
        if agg == 'mean':
            graph_loss = sum(per_iter_graph_losses) / len(per_iter_graph_losses)
        elif agg == 'weighted_mean':
            T = len(per_iter_graph_losses)
            if not graph_loss_weights or len(graph_loss_weights) != T:
                graph_loss_weights = [i + 1 for i in range(T)]
            ws = torch.tensor(
                graph_loss_weights,
                dtype=per_iter_graph_losses[0].dtype,
                device=per_iter_graph_losses[0].device,
            )
            ws = ws / ws.sum()
            graph_loss = sum((li * wi for li, wi in zip(per_iter_graph_losses, ws)))
        else:
            graph_loss = per_iter_graph_losses[-1]
    elif isinstance(graph_loss_function, nn.BCELoss):
        graph_loss = graph_loss_function(adj_probs_list[-1], ref_matrix)
    else:
        graph_loss = graph_loss_function(adj_logits_list[-1], ref_matrix)
    return (graph_loss, per_iter_graph_losses)


def calibrate_global_threshold_on_dev(
    parser,
    dev_loader,
    batch_size=1,
    device=None,
    metric="labeled_f1_doc_macro",
    grid=None,
    use_logits=True,
    max_docs=None,
):
    """Select a global cutoff on DEV using dense final-state arc logits."""
    import numpy as np

    if grid is None:
        grid = [i / 100.0 for i in range(5, 96, 5)]
    if device is None:
        device = next(parser.parameters()).device
    parser.eval()
    samples = []
    with torch.no_grad():
        count = 0
        for batch, idxs in dev_loader.generate_batch(batch_size, randomize=False):
            if max_docs is not None and count >= max_docs:
                break
            batch = batch.to(device)
            example = dev_loader.example_loader.dataset[idxs[0]]
            forward_result = parser(batch, example, mode="val")
            adj_logits_list, _, adj_probs_list, nodeX_list = forward_result[:4]
            mat = adj_logits_list[-1] if use_logits else adj_probs_list[-1]
            nodeX = nodeX_list[-1]
            ref_graph = dev_loader.graphs[idxs[0]]
            samples.append((mat, nodeX, batch, ref_graph))
            count += 1

    def build_pred_graph(mat, nodeX, batch, tau):
        return (
            parser.create_graph_from_logits_with_threshold(mat, nodeX, batch, tau)
            if use_logits
            else parser.create_graph_from_probs_with_threshold(mat, nodeX, batch, tau)
        )

    best_tau = None
    best_score = -1.0
    metric_raw = str(metric).lower().strip()
    scope = "temporal" if metric_raw.endswith("_temporal") else "all"
    metric_name = metric_raw[:-9] if scope == "temporal" else metric_raw
    want_labeled = metric_name.startswith("labeled_")
    want_unlabeled = metric_name.startswith("unlabeled_")
    if not (want_labeled or want_unlabeled):
        raise ValueError(
            f"Unknown metric '{metric}'. Expected 'labeled_*' or 'unlabeled_*'."
        )
    structural_ids = _get_structural_label_ids(dev_loader.edge_vocabulary)
    exclude_ids = structural_ids if scope == "temporal" else None
    num_labels = len(getattr(dev_loader.edge_vocabulary, "idx2sym", []))
    label_ids_for_macro = (
        list(range(num_labels))
        if scope == "all"
        else [i for i in range(num_labels) if i not in structural_ids]
    )
    if metric_name.endswith("_f1_micro"):
        avg = "micro"
    elif metric_name.endswith("_f1_doc_macro"):
        avg = "doc_macro"
    elif metric_name.endswith("_f1_macro"):
        avg = "label_macro" if want_labeled else "doc_macro"
    else:
        raise ValueError(f"Unsupported metric '{metric}'.")
    for tau in grid:
        if want_labeled:
            if avg == "label_macro":
                tp = [0] * num_labels
                fp = [0] * num_labels
                fn = [0] * num_labels
                for mat, nodeX, batch, ref_graph in samples:
                    g = build_pred_graph(mat, nodeX, batch, tau)
                    gold = _edge_dict_from_graph(
                        ref_graph, exclude_label_ids=exclude_ids
                    )
                    pred = _edge_dict_from_graph(g, exclude_label_ids=exclude_ids)
                    tpi, fpi, fni = _compute_labeled_per_class_counts(
                        gold, pred, num_labels
                    )
                    for j in range(num_labels):
                        tp[j] += tpi[j]
                        fp[j] += fpi[j]
                        fn[j] += fni[j]
                _, _, score = _macro_prf_over_labels(
                    tp, fp, fn, label_ids_for_macro, only_gold_support=True
                )
            elif avg == "doc_macro":
                f1_list = []
                for mat, nodeX, batch, ref_graph in samples:
                    g = build_pred_graph(mat, nodeX, batch, tau)
                    gold = _edge_dict_from_graph(
                        ref_graph, exclude_label_ids=exclude_ids
                    )
                    pred = _edge_dict_from_graph(g, exclude_label_ids=exclude_ids)
                    TP, FP, FN = _compute_labeled_micro_counts(gold, pred)
                    f1_list.append(calculate_f1_from_tp_fp_fn(TP, FP, FN))
                score = float(np.mean(f1_list)) if f1_list else 0.0
            else:
                TP = FP = FN = 0
                for mat, nodeX, batch, ref_graph in samples:
                    g = build_pred_graph(mat, nodeX, batch, tau)
                    gold = _edge_dict_from_graph(
                        ref_graph, exclude_label_ids=exclude_ids
                    )
                    pred = _edge_dict_from_graph(g, exclude_label_ids=exclude_ids)
                    tp_i, fp_i, fn_i = _compute_labeled_micro_counts(gold, pred)
                    TP += tp_i
                    FP += fp_i
                    FN += fn_i
                score = calculate_f1_from_tp_fp_fn(TP, FP, FN)
        elif avg == "doc_macro":
            f1_list = []
            for mat, nodeX, batch, ref_graph in samples:
                g = build_pred_graph(mat, nodeX, batch, tau)
                gold = _edge_dict_from_graph(ref_graph, exclude_label_ids=exclude_ids)
                pred = _edge_dict_from_graph(g, exclude_label_ids=exclude_ids)
                TP, FP, FN = _compute_unlabeled_counts(
                    set(gold.keys()), set(pred.keys())
                )
                f1_list.append(calculate_f1_from_tp_fp_fn(TP, FP, FN))
            score = float(np.mean(f1_list)) if f1_list else 0.0
        else:
            TP = FP = FN = 0
            for mat, nodeX, batch, ref_graph in samples:
                g = build_pred_graph(mat, nodeX, batch, tau)
                gold = _edge_dict_from_graph(ref_graph, exclude_label_ids=exclude_ids)
                pred = _edge_dict_from_graph(g, exclude_label_ids=exclude_ids)
                tp_i, fp_i, fn_i = _compute_unlabeled_counts(
                    set(gold.keys()), set(pred.keys())
                )
                TP += tp_i
                FP += fp_i
                FN += fn_i
            score = calculate_f1_from_tp_fp_fn(TP, FP, FN)
        if score > best_score:
            best_score = float(score)
            best_tau = float(tau)
    return (float(best_tau), float(best_score))


def predict_with_global_threshold(
    parser, data_loader, tau, batch_size=1, device=None, use_logits=True
):
    if device is None:
        device = next(parser.parameters()).device
    parser.eval()
    graphs = []
    with torch.no_grad():
        for batch, idxs in data_loader.generate_batch(batch_size, randomize=False):
            batch = batch.to(device)
            example = data_loader.example_loader.dataset[idxs[0]]
            forward_result = parser(batch, example, mode="val")
            adj_logits_list, _, adj_probs_list, nodeX_list = forward_result[:4]
            mat = adj_logits_list[-1] if use_logits else adj_probs_list[-1]
            nodeX = nodeX_list[-1]
            if use_logits:
                g = parser.create_graph_from_logits_with_threshold(
                    mat, nodeX, batch, tau
                )
            else:
                g = parser.create_graph_from_probs_with_threshold(
                    mat, nodeX, batch, tau
                )
            try:
                g = g.to("cpu")
            except Exception as e:
                constants.LOGGER.warning(f"Could not move predicted graph to CPU: {e}")
            graphs.append(g)
    return graphs


def compute_log_and_save_per_iter_metrics_fair(
    parser,
    data_loader,
    tau_star,
    output_dir,
    max_docs=None,
    logger=None,
    use_logits=True,
):
    os.makedirs(output_dir, exist_ok=True)
    from collections import defaultdict

    try:
        doc_limit = (
            int(max_docs)
            if max_docs is not None
            else int(len(getattr(data_loader, "graphs", [])))
        )
        if doc_limit <= 0:
            doc_limit = int(len(getattr(data_loader, "graphs", [])))
    except Exception:
        doc_limit = int(len(getattr(data_loader, "graphs", [])))
    results_fair = []
    parser.eval()
    num_seen = 0
    device = next(parser.parameters()).device
    with torch.no_grad():
        for batch, idxs in data_loader.generate_batch(1, randomize=False):
            if num_seen >= doc_limit:
                break
            batch = batch.to(device)
            example = data_loader.example_loader.dataset[idxs[0]]
            forward_result = parser(batch, example, mode="val")
            adj_logits_list, _, adj_probs_list, nodeX_list = forward_result[:4]
            steps = []
            for i in range(len(nodeX_list)):
                mat = adj_logits_list[i] if use_logits else adj_probs_list[i]
                nodeX = nodeX_list[i]
                if use_logits:
                    g = parser.create_graph_from_logits_with_threshold(
                        mat, nodeX, batch, tau_star
                    )
                else:
                    g = parser.create_graph_from_probs_with_threshold(
                        mat, nodeX, batch, tau_star
                    )
                try:
                    g = g.to("cpu")
                except Exception:
                    pass
                steps.append({"iteration": i, "sparse_batch": g})
            results_fair.append({"idx": idxs[0], "per_iteration": steps})
            num_seen += 1
    structural_ids = _get_structural_label_ids(data_loader.edge_vocabulary)
    num_labels = len(getattr(data_loader.edge_vocabulary, "idx2sym", []))
    label_ids_all = list(range(num_labels))
    label_ids_tmp = [i for i in label_ids_all if i not in structural_ids]
    scopes = [("all", None, label_ids_all), ("temporal", structural_ids, label_ids_tmp)]
    lab_micro = {
        s: defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0}) for s, _, _ in scopes
    }
    un_micro = {
        s: defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0}) for s, _, _ in scopes
    }
    lab_doc = {
        s: defaultdict(lambda: {"P": [], "R": [], "F1": []}) for s, _, _ in scopes
    }
    un_doc = {
        s: defaultdict(lambda: {"P": [], "R": [], "F1": []}) for s, _, _ in scopes
    }
    lab_tp = {s: defaultdict(lambda: [0] * num_labels) for s, _, _ in scopes}
    lab_fp = {s: defaultdict(lambda: [0] * num_labels) for s, _, _ in scopes}
    lab_fn = {s: defaultdict(lambda: [0] * num_labels) for s, _, _ in scopes}
    for sample in results_fair:
        doc_id = sample["idx"]
        gt_graph = data_loader.graphs[doc_id]
        gold_all = _edge_dict_from_graph(gt_graph, exclude_label_ids=None)
        gold_tmp = _edge_dict_from_graph(gt_graph, exclude_label_ids=structural_ids)
        for step in sample["per_iteration"]:
            it = int(step["iteration"])
            pred_graph = step["sparse_batch"]
            pred_all = _edge_dict_from_graph(pred_graph, exclude_label_ids=None)
            pred_tmp = _edge_dict_from_graph(
                pred_graph, exclude_label_ids=structural_ids
            )
            for scope_name, exclude_ids, label_ids_for_macro in scopes:
                gold = gold_all if scope_name == "all" else gold_tmp
                pred = pred_all if scope_name == "all" else pred_tmp
                TP, FP, FN = _compute_labeled_micro_counts(gold, pred)
                lab_micro[scope_name][it]["TP"] += TP
                lab_micro[scope_name][it]["FP"] += FP
                lab_micro[scope_name][it]["FN"] += FN
                lab_doc[scope_name][it]["P"].append(calculate_precision(TP, FP))
                lab_doc[scope_name][it]["R"].append(calculate_recall(TP, FN))
                lab_doc[scope_name][it]["F1"].append(
                    calculate_f1_from_tp_fp_fn(TP, FP, FN)
                )
                tpi, fpi, fni = _compute_labeled_per_class_counts(
                    gold, pred, num_labels
                )
                tp_arr = lab_tp[scope_name][it]
                fp_arr = lab_fp[scope_name][it]
                fn_arr = lab_fn[scope_name][it]
                for j in range(num_labels):
                    tp_arr[j] += tpi[j]
                    fp_arr[j] += fpi[j]
                    fn_arr[j] += fni[j]
                TP, FP, FN = _compute_unlabeled_counts(
                    set(gold.keys()), set(pred.keys())
                )
                un_micro[scope_name][it]["TP"] += TP
                un_micro[scope_name][it]["FP"] += FP
                un_micro[scope_name][it]["FN"] += FN
                un_doc[scope_name][it]["P"].append(calculate_precision(TP, FP))
                un_doc[scope_name][it]["R"].append(calculate_recall(TP, FN))
                un_doc[scope_name][it]["F1"].append(
                    calculate_f1_from_tp_fp_fn(TP, FP, FN)
                )
    rows = []
    for scope_name, _, label_ids_for_macro in scopes:
        iters = set(lab_micro[scope_name].keys()) | set(un_micro[scope_name].keys())
        if not iters:
            continue
        for it in sorted(iters):
            TP = lab_micro[scope_name][it]["TP"]
            FP = lab_micro[scope_name][it]["FP"]
            FN = lab_micro[scope_name][it]["FN"]
            p_micro = calculate_precision(TP, FP)
            r_micro = calculate_recall(TP, FN)
            f1_micro = calculate_f1_from_tp_fp_fn(TP, FP, FN)
            denom = TP + FP + FN
            jacc = float(TP / denom) if denom > 0 else 0.0
            P_list = lab_doc[scope_name][it]["P"]
            R_list = lab_doc[scope_name][it]["R"]
            F1_list = lab_doc[scope_name][it]["F1"]
            p_doc_macro = float(np.mean(P_list)) if P_list else 0.0
            r_doc_macro = float(np.mean(R_list)) if R_list else 0.0
            f1_doc_macro = float(np.mean(F1_list)) if F1_list else 0.0
            tp_arr = lab_tp[scope_name][it]
            fp_arr = lab_fp[scope_name][it]
            fn_arr = lab_fn[scope_name][it]
            p_label_macro, r_label_macro, f1_label_macro = _macro_prf_over_labels(
                tp_arr, fp_arr, fn_arr, label_ids_for_macro, only_gold_support=True
            )
            rows.append(
                {
                    "iteration": int(it),
                    "scope": scope_name,
                    "mode": "labeled",
                    "p_micro": p_micro,
                    "r_micro": r_micro,
                    "f1_micro": f1_micro,
                    "p_doc_macro": p_doc_macro,
                    "r_doc_macro": r_doc_macro,
                    "f1_doc_macro": f1_doc_macro,
                    "p_label_macro": p_label_macro,
                    "r_label_macro": r_label_macro,
                    "f1_label_macro": f1_label_macro,
                    "jaccard_micro": jacc,
                }
            )
            TP = un_micro[scope_name][it]["TP"]
            FP = un_micro[scope_name][it]["FP"]
            FN = un_micro[scope_name][it]["FN"]
            p_micro = calculate_precision(TP, FP)
            r_micro = calculate_recall(TP, FN)
            f1_micro = calculate_f1_from_tp_fp_fn(TP, FP, FN)
            denom = TP + FP + FN
            jacc = float(TP / denom) if denom > 0 else 0.0
            P_list = un_doc[scope_name][it]["P"]
            R_list = un_doc[scope_name][it]["R"]
            F1_list = un_doc[scope_name][it]["F1"]
            p_doc_macro = float(np.mean(P_list)) if P_list else 0.0
            r_doc_macro = float(np.mean(R_list)) if R_list else 0.0
            f1_doc_macro = float(np.mean(F1_list)) if F1_list else 0.0
            rows.append(
                {
                    "iteration": int(it),
                    "scope": scope_name,
                    "mode": "unlabeled",
                    "p_micro": p_micro,
                    "r_micro": r_micro,
                    "f1_micro": f1_micro,
                    "p_doc_macro": p_doc_macro,
                    "r_doc_macro": r_doc_macro,
                    "f1_doc_macro": f1_doc_macro,
                    "p_label_macro": np.nan,
                    "r_label_macro": np.nan,
                    "f1_label_macro": np.nan,
                    "jaccard_micro": jacc,
                }
            )
    df_out = pd.DataFrame(rows)
    df_out = df_out.reindex(
        columns=[
            "iteration",
            "scope",
            "mode",
            "p_micro",
            "r_micro",
            "f1_micro",
            "p_doc_macro",
            "r_doc_macro",
            "f1_doc_macro",
            "p_label_macro",
            "r_label_macro",
            "f1_label_macro",
            "jaccard_micro",
        ]
    )
    try:
        bin_defs = [("same", 0, 0), ("adjacent", 1, 1), ("far2plus", 2, None)]
        from collections import defaultdict

        agg_bins = defaultdict(
            lambda: {
                "labeled": {b[0]: {"TP": 0, "FP": 0, "FN": 0} for b in bin_defs},
                "unlabeled": {b[0]: {"TP": 0, "FP": 0, "FN": 0} for b in bin_defs},
            }
        )
        num_seen2 = 0
        for sample in results_fair:
            if num_seen2 >= doc_limit:
                break
            gt_graph = data_loader.graphs[sample["idx"]]
            try:
                example = data_loader.example_loader.dataset[sample["idx"]]
            except Exception:
                example = None
            for step in sample["per_iteration"]:
                pred_graph = step["sparse_batch"]
                ref_tuples, pred_tuples = turn_predictions_into_labeled_tuples(
                    pred_graph, gt_graph, data_loader.edge_vocabulary
                )
                for name, _, _ in bin_defs:
                    try:
                        if example is None:
                            ref_b = set(ref_tuples)
                            pred_b = set(pred_tuples)
                        else:
                            ref_b = set(
                                (
                                    t
                                    for t in ref_tuples
                                    if _distance_bin_name(example, t[0], t[1], bin_defs)
                                    == name
                                )
                            )
                            pred_b = set(
                                (
                                    t
                                    for t in pred_tuples
                                    if _distance_bin_name(example, t[0], t[1], bin_defs)
                                    == name
                                )
                            )
                        TP = len(ref_b & pred_b)
                        FP = len(pred_b - ref_b)
                        FN = len(ref_b - pred_b)
                        agg_bins[step["iteration"]]["labeled"][name]["TP"] += TP
                        agg_bins[step["iteration"]]["labeled"][name]["FP"] += FP
                        agg_bins[step["iteration"]]["labeled"][name]["FN"] += FN
                    except Exception:
                        pass
                ref_edges, pred_edges = turn_predictions_into_unlabeled_tuples(
                    pred_graph, gt_graph, data_loader.edge_vocabulary
                )
                for name, _, _ in bin_defs:
                    try:
                        if example is None:
                            ref_b = set(ref_edges)
                            pred_b = set(pred_edges)
                        else:
                            ref_b = set(
                                (
                                    e
                                    for e in ref_edges
                                    if _distance_bin_name(example, e[0], e[1], bin_defs)
                                    == name
                                )
                            )
                            pred_b = set(
                                (
                                    e
                                    for e in pred_edges
                                    if _distance_bin_name(example, e[0], e[1], bin_defs)
                                    == name
                                )
                            )
                        TP = len(ref_b & pred_b)
                        FP = len(pred_b - ref_b)
                        FN = len(ref_b - pred_b)
                        agg_bins[step["iteration"]]["unlabeled"][name]["TP"] += TP
                        agg_bins[step["iteration"]]["unlabeled"][name]["FP"] += FP
                        agg_bins[step["iteration"]]["unlabeled"][name]["FN"] += FN
                    except Exception:
                        pass
            num_seen2 += 1
        rows_bin = []
        for it, modes in agg_bins.items():
            for mode_name, bin_map in modes.items():
                for name, counts in bin_map.items():
                    TP = counts["TP"]
                    FP = counts["FP"]
                    FN = counts["FN"]
                    p = calculate_precision(TP, FP)
                    r = calculate_recall(TP, FN)
                    f1 = calculate_f1_from_tp_fp_fn(TP, FP, FN)
                    rows_bin.append(
                        {
                            "iteration": it,
                            "mode": mode_name,
                            "bin": name,
                            "p_micro": p,
                            "r_micro": r,
                            "f1_micro": f1,
                        }
                    )
        df_bins = pd.DataFrame(rows_bin)
    except Exception:
        df_bins = pd.DataFrame(
            columns=["iteration", "mode", "bin", "p_micro", "r_micro", "f1_micro"]
        )
    try:
        from collections import defaultdict

        counts = defaultdict(lambda: defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0}))
        num_seen3 = 0
        for sample in results_fair:
            if num_seen3 >= doc_limit:
                break
            gt_graph = data_loader.graphs[sample["idx"]]
            for step in sample["per_iteration"]:
                pred_graph = step["sparse_batch"]
                ref_tuples, pred_tuples = turn_predictions_into_labeled_tuples(
                    pred_graph, gt_graph, data_loader.edge_vocabulary
                )
                ref_map = {(i, j): lbl for i, j, lbl in ref_tuples}
                pred_map = {(i, j): lbl for i, j, lbl in pred_tuples}
                pos = set(list(ref_map.keys()) + list(pred_map.keys()))
                for i, j in pos:
                    ref_lbl = ref_map.get((i, j), None)
                    pred_lbl = pred_map.get((i, j), None)
                    if ref_lbl is not None and pred_lbl is not None:
                        if ref_lbl == pred_lbl:
                            counts[step["iteration"]][ref_lbl]["TP"] += 1
                        else:
                            counts[step["iteration"]][pred_lbl]["FP"] += 1
                            counts[step["iteration"]][ref_lbl]["FN"] += 1
                    elif ref_lbl is not None:
                        counts[step["iteration"]][ref_lbl]["FN"] += 1
                    elif pred_lbl is not None:
                        counts[step["iteration"]][pred_lbl]["FP"] += 1
            num_seen3 += 1
        rows_lbl = []
        id2name = getattr(data_loader.edge_vocabulary, "idx2sym", None)
        for it, label_map in counts.items():
            for lbl_id, c in label_map.items():
                TP = c["TP"]
                FP = c["FP"]
                FN = c["FN"]
                p = calculate_precision(TP, FP)
                r = calculate_recall(TP, FN)
                f1 = calculate_f1_from_tp_fp_fn(TP, FP, FN)
                lbl_name = (
                    id2name[lbl_id]
                    if isinstance(id2name, list) and 0 <= lbl_id < len(id2name)
                    else str(lbl_id)
                )
                rows_lbl.append(
                    {
                        "iteration": it,
                        "label_id": int(lbl_id),
                        "label": lbl_name,
                        "p_micro": p,
                        "r_micro": r,
                        "f1_micro": f1,
                        "TP": int(TP),
                        "FP": int(FP),
                        "FN": int(FN),
                    }
                )
        df_labels = pd.DataFrame(rows_lbl)
    except Exception:
        df_labels = pd.DataFrame(
            columns=[
                "iteration",
                "label_id",
                "label",
                "p_micro",
                "r_micro",
                "f1_micro",
                "TP",
                "FP",
                "FN",
            ]
        )
    df_out.to_csv(
        os.path.join(output_dir, "per_iter_slice_metrics_fair.csv"), index=False
    )
    df_bins.to_csv(
        os.path.join(output_dir, "per_iter_slice_metrics_fair_distance_binned.csv"),
        index=False,
    )
    df_labels.to_csv(
        os.path.join(output_dir, "per_iter_slice_metrics_fair_per_label.csv"),
        index=False,
    )
    try:
        from statistics import mean as _mean

        inter_unl = defaultdict(int)
        union_unl = defaultdict(int)
        inter_lab = defaultdict(int)
        union_lab = defaultdict(int)
        macro_unl = defaultdict(list)
        macro_lab = defaultdict(list)
        num_seen4 = 0
        for sample in results_fair:
            if num_seen4 >= doc_limit:
                break
            steps = sample["per_iteration"]
            if len(steps) < 2:
                num_seen4 += 1
                continue
            for i in range(1, len(steps)):
                g_prev = steps[i - 1]["sparse_batch"]
                g_cur = steps[i]["sparse_batch"]
                sp, tp = g_prev.edges()
                su, tu = g_cur.edges()
                E_prev = set(zip(sp.tolist(), tp.tolist()))
                E_cur = set(zip(su.tolist(), tu.tolist()))
                inter = len(E_prev & E_cur)
                uni = len(E_prev | E_cur)
                inter_unl[i] += inter
                union_unl[i] += uni
                macro_unl[i].append(inter / uni if uni > 0 else 0.0)
                lp = g_prev.edata.get("_depID", None)
                lc = g_cur.edata.get("_depID", None)
                if lp is not None and lc is not None:
                    lp = lp.tolist()
                    lc = lc.tolist()
                    E_prev_l = set(
                        (
                            (s, t, l)
                            for (s, t), l in zip(zip(sp.tolist(), tp.tolist()), lp)
                        )
                    )
                    E_cur_l = set(
                        (
                            (s, t, l)
                            for (s, t), l in zip(zip(su.tolist(), tu.tolist()), lc)
                        )
                    )
                    inter_l = len(E_prev_l & E_cur_l)
                    uni_l = len(E_prev_l | E_cur_l)
                    inter_lab[i] += inter_l
                    union_lab[i] += uni_l
                    macro_lab[i].append(inter_l / uni_l if uni_l > 0 else 0.0)
            num_seen4 += 1
        rows_stab = []
        for it in sorted(set(list(inter_unl.keys()) + list(inter_lab.keys()))):
            IU = inter_unl.get(it, 0)
            UU = union_unl.get(it, 0)
            j_micro_u = float(IU / UU) if UU > 0 else 0.0
            j_macro_u = (
                float(_mean(macro_unl.get(it, [0.0]))) if macro_unl.get(it) else 0.0
            )
            rows_stab.append(
                {
                    "iteration": it,
                    "mode": "unlabeled",
                    "jaccard_micro": j_micro_u,
                    "jaccard_macro": j_macro_u,
                }
            )
            IL = inter_lab.get(it, 0)
            UL = union_lab.get(it, 0)
            j_micro_l = float(IL / UL) if UL > 0 else 0.0
            j_macro_l = (
                float(_mean(macro_lab.get(it, [0.0]))) if macro_lab.get(it) else 0.0
            )
            rows_stab.append(
                {
                    "iteration": it,
                    "mode": "labeled",
                    "jaccard_micro": j_micro_l,
                    "jaccard_macro": j_macro_l,
                }
            )
        df_stability = pd.DataFrame(rows_stab)
    except Exception:
        df_stability = pd.DataFrame(
            columns=["iteration", "mode", "jaccard_micro", "jaccard_macro"]
        )
    df_stability.to_csv(
        os.path.join(output_dir, "per_iter_slice_metrics_fair_stability.csv"),
        index=False,
    )
    if logger is not None:
        logger.info(
            f"Saved fair per-iteration metrics to {os.path.join(output_dir, 'per_iter_slice_metrics_fair.csv')}"
        )
        logger.info(
            f"Saved fair per-iteration distance-binned metrics to {os.path.join(output_dir, 'per_iter_slice_metrics_fair_distance_binned.csv')}"
        )
        logger.info(
            f"Saved fair per-iteration per-label metrics to {os.path.join(output_dir, 'per_iter_slice_metrics_fair_per_label.csv')}"
        )
        logger.info(
            f"Saved fair per-iteration stability metrics to {os.path.join(output_dir, 'per_iter_slice_metrics_fair_stability.csv')}"
        )
    return df_out


def _distance_bin_name(example, i, j, bins):
    try:
        di = int(example.node_id2sentence_number(i))
        dj = int(example.node_id2sentence_number(j))
        d = abs(dj - di)
    except Exception:
        d = 999999
    for name, lo, hi in bins:
        if d >= lo and (hi is None or d <= hi):
            return name
    return None
