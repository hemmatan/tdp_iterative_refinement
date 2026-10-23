"""Train and evaluate one fixed paper configuration. Launched by train.py."""

import argparse
import copy
import datetime
import gc
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path

import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

from tdp import configuration, constants, evaluation_tools, training_utils
from tdp.data_handler.tdg_loader import BasicDataLoader
from tdp.model import TemporalDependencyParser
from tdp.release_support import validate_paper_config


def parse_arguments(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--config", type=Path, required=True, help="Runtime JSON with corpus paths")
    cli.add_argument("--phase", choices=("baseline", "iterative"), required=True)
    cli.add_argument("--seed", type=int, choices=(30, 31, 32), required=True)
    cli.add_argument("--gpu_id", type=int, default=0)
    cli.add_argument("--run_timestamp", required=True, help="Unique run identifier")
    cli.add_argument("--message", default="run")
    cli.add_argument("--details", default="")
    selected = cli.parse_args(argv)
    runtime = json.loads(selected.config.read_text())
    required = {"train_file", "dev_file", "test_file"}
    allowed = required | {"log_level", "log_rotate_mb", "log_rotate_backups", "log_console"}
    if not required <= runtime.keys() or runtime.keys() - allowed:
        cli.error("Runtime configuration must contain corpus paths and supported logging options only.")
    if Path(selected.run_timestamp).name != selected.run_timestamp or selected.run_timestamp in (".", ".."):
        cli.error("Run identifier must be a single directory name.")
    runtime.update(vars(selected))
    return argparse.Namespace(**runtime)


def setup_logging(directory, args):
    directory.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("tdp")
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    logger.setLevel(getattr(logging, str(getattr(args, "log_level", "INFO")).upper()))
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    handler = RotatingFileHandler(
        directory / "run.log", maxBytes=int(getattr(args, "log_rotate_mb", 20)) * 1024 * 1024,
        backupCount=int(getattr(args, "log_rotate_backups", 10)), encoding="utf-8",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    if getattr(args, "log_console", False):
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)
    return logger


def jsonable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def load_for_final_evaluation(parser, checkpoint, device):
    """Materialize dynamic projectors and load the selected checkpoint."""
    try:
        with torch.no_grad():
            dummy_node = torch.zeros(1, parser.node_embedding_dim, device=device)
            parser.proj_s(dummy_node)
            parser.proj_t(dummy_node)
            parser.gnn_in_projector(dummy_node)
            hidden_dim = int(getattr(parser.args, "GCN_HIDDEN_DIM", parser.node_embedding_dim))
            dummy_hidden = torch.zeros(1, hidden_dim, device=device)
            parser.gnn_in_projector(dummy_hidden)
            parser.proj_s(dummy_hidden)
            parser.proj_t(dummy_hidden)
            for projection in getattr(parser, "residual_projections", []):
                projection(dummy_hidden)
        parser.load_state_dict(torch.load(checkpoint, map_location=device), strict=True)
    except (AttributeError, RuntimeError, ValueError) as error:
        constants.LOGGER.warning("Strict checkpoint preparation failed: %s", error)
        missing, unexpected = parser.load_state_dict(
            torch.load(checkpoint, map_location=device), strict=False
        )
        if missing or unexpected:
            raise RuntimeError(f"Checkpoint parameters differ: missing={missing}, unexpected={unexpected}")


def main(argv=None):
    args = parse_arguments(argv)
    output = Path(os.environ["TDP_OUTPUT_DIR"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(output)
    constants.PROJECT_PATH = str(output)
    constants.APP_START_TIMESTAMP = args.run_timestamp
    log_dir = output / "logs" / args.run_timestamp
    constants.CURRENT_RUN_LOG_DIR = str(log_dir)
    constants.LOGGER = setup_logging(log_dir, args)
    if args.details:
        (log_dir / "details.txt").write_text(args.details + "\n")
    training_utils.set_all_seeds(args.seed)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    train_loader = BasicDataLoader(args.train_file, apply_algebra=False, encoder_type="BERT")
    loaders = [train_loader]
    for split in ("dev", "test"):
        loaders.append(BasicDataLoader(
            getattr(args, f"{split}_file"), token_vocabulary=train_loader.tok_vocab,
            edge_vocabulary=train_loader.edge_vocabulary, apply_algebra=False, encoder_type="BERT",
        ))
    _, dev_loader, test_loader = loaders
    counts = [0 for _ in train_loader.example_loader.edge_vocab.idx2sym]
    for graph in train_loader.graphs:
        labels = graph.edata["_depID"]
        for i in range(len(counts)):
            counts[i] += (labels == i).sum().item()
    alphas = training_utils.make_class_balanced_alphas(
        torch.tensor(counts), beta=0.999, clip_range=(0.25, 4.0),
        normalize="mean_one", device=None, dtype=torch.float32,
    )
    parameters = configuration.load_parameters(args.phase)
    parameters["alphas"] = alphas.cpu()
    configuration.advance_setup_rng()
    validate_paper_config(parameters)
    for name, value in parameters.items():
        setattr(args, name, value)
    edge_loss = training_utils.get_edge_loss_fnc(
        args.loss_function, args.alphas, args.gamma, args.which_loss_implementation,
    )
    graph_loss = training_utils.get_graph_loss_fnc(args.graph_loss_fnc)
    parser = TemporalDependencyParser(
        train_loader.tok_vocab, train_loader.edge_vocabulary, args=args, encoder_type="BERT",
    ).to(device)
    model_name = f"{args.phase}_s{args.seed}"
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    config_path = output / "model_config.json"
    config_path.write_text(json.dumps(jsonable(vars(args)), indent=2) + "\n")
    constants.LOGGER.info("Training %s on %s", model_name, device)
    with SummaryWriter(log_dir=str(output / "tensorboard")) as writer:
        model_file, dev_metrics, dev_cutoff = parser.train_all(
            train_loader, dev_loader, test_loader, args, timestamp, model_name,
            writer=writer, graph_loss_function=graph_loss, loss_function=edge_loss,
            param_combination=parameters,
        )
    if dev_metrics is None or dev_cutoff is None:
        raise RuntimeError("Training did not return selected-checkpoint DEV metrics.")
    checkpoint = output / "models" / model_file
    missing, unexpected = parser.load_state_dict(torch.load(checkpoint, map_location=device), strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint parameters differ: missing={missing}, unexpected={unexpected}")
    diagnostic_dir = log_dir / "per_iter" / f"{timestamp}_{model_name}"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    evaluation_tools.compute_log_and_save_per_iter_metrics_fair(
        parser, dev_loader, dev_cutoff, str(diagnostic_dir), max_docs=None,
        logger=constants.LOGGER, use_logits=True,
    )
    metadata = {
        "hp_timestamp": timestamp, "model_name": model_name,
        "parser_class": type(parser).__name__, "config_path": str(config_path.relative_to(output)),
        "best_fair_epoch": parser._best_fair_epoch, "tau_star_dev": float(dev_cutoff),
        "dev_metrics": jsonable(dev_metrics), "param_combination": jsonable(parameters),
    }
    (diagnostic_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if hasattr(dev_loader, "parser_for_eval"):
        delattr(dev_loader, "parser_for_eval")
    parser.to("cpu")
    del parser
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    final_parser = TemporalDependencyParser(
        train_loader.tok_vocab, train_loader.edge_vocabulary,
        args=copy.deepcopy(args), encoder_type="BERT",
    ).to(device)
    load_for_final_evaluation(final_parser, checkpoint, device)
    cutoff, _ = evaluation_tools.calibrate_global_threshold_on_dev(
        final_parser, dev_loader, batch_size=1, device=device,
        metric="labeled_f1_doc_macro", grid=None, use_logits=True,
    )
    final_parser.set_inference_threshold(cutoff, sync_learned=args.EVAL_SYNC_LEARNED_THRESHOLD)
    predictions = evaluation_tools.predict_with_global_threshold(
        final_parser, test_loader, cutoff, batch_size=1, device=device, use_logits=True,
    )
    metrics, _ = evaluation_tools.evaluate_predicted_graphs(predictions, test_loader, save_statistics=True)
    metric_names = (
        "labeled_f1_micro", "labeled_f1_macro", "labeled_f1_doc_macro",
        "unlabeled_f1_micro", "unlabeled_f1_doc_macro", "labeled_f1_micro_temporal",
        "labeled_f1_macro_temporal", "labeled_f1_doc_macro_temporal",
        "unlabeled_f1_micro_temporal", "unlabeled_f1_doc_macro_temporal",
    )
    row = {"timestamp": args.run_timestamp, "model_name": model_name, "tau_star": cutoff}
    row.update({f"test_{name}": metrics[getattr(constants.METRIC_NAMES, name)] for name in metric_names})
    pd.DataFrame([row]).to_csv(log_dir / "final_test_metrics.csv", index=False)
    constants.LOGGER.info("Completed %s; final metrics saved.", model_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
