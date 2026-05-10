"""
Train the heterogeneous graph anomaly detector and save artifacts for testing.

This script reuses the model, graph construction, and objective definitions from
graph_hete_end_v2.py. It only performs training, training-score calibration, and
artifact saving.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config
import graph_hete_end_v2 as core


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train heterogeneous graph anomaly detector.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--min-frequency", type=int, default=1, help="Minimum token frequency kept in vocabularies.")
    parser.add_argument("--rebuild-vocab", action="store_true", help="Force rebuilding vocabularies.")
    parser.add_argument("--field-embedding-dim", type=int, default=16, help="Embedding dimension for each Event input field.")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension for nodes and R-GCN layers.")
    parser.add_argument("--rgcn-layers", type=int, default=2, help="Number of R-GCN layers.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout ratio.")
    parser.add_argument("--mask-ratio", type=float, default=0.3, help="Fraction of Event nodes masked in each training graph.")
    parser.add_argument("--lambda-rel", type=float, default=0.5, help="Weight for relation prediction loss.")
    parser.add_argument("--lambda-trans", type=float, default=0.5, help="Weight for transition prediction loss.")
    parser.add_argument("--score-mask-batch-size", type=int, default=32, help="Event nodes masked together in scoring.")
    parser.add_argument("--batch-size", type=int, default=16, help="Window graphs per optimization step.")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--nu", type=float, default=0.1, help="Normal training fraction above anomaly threshold.")
    parser.add_argument("--actor-field", default="actor_id", help="Field name used to build Actor nodes.")
    parser.add_argument("--window-score-reduction", choices=["max", "mean", "meanmax", "topk"], default="max")
    parser.add_argument("--top-k", type=int, default=3, help="Top-k for topk window score reduction.")
    parser.add_argument("--device", default="cuda:1", help="Torch device, for example cpu, cuda, or cuda:1.")
    parser.add_argument("--vocab-output", default=str(Path(config.arithmetic_path) / "graph_hete_train_vocab.json"))
    parser.add_argument("--model-output", default=str(Path(config.arithmetic_path) / "graph_hete_train_model.pt"))
    parser.add_argument("--threshold-output", default=str(Path(config.arithmetic_path) / "graph_hete_train_threshold.json"))
    parser.add_argument("--train-scores-output", default=str(Path(config.arithmetic_path) / "graph_hete_train_scores.jsonl"))
    parser.add_argument("--report-output", default=str(Path(config.arithmetic_path) / "graph_hete_train_report.txt"))
    return parser


def write_train_report(output_path: Path, args: argparse.Namespace, threshold: float, training_history) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("Heterogeneous graph train report\n")
        handle.write(f"model_output: {args.model_output}\n")
        handle.write(f"vocab_output: {args.vocab_output}\n")
        handle.write(f"threshold_output: {args.threshold_output}\n")
        handle.write(f"threshold: {threshold:.6f}\n")
        handle.write(f"window_score_reduction: {args.window_score_reduction}\n")
        handle.write(f"lambda_rel: {args.lambda_rel}\n")
        handle.write(f"lambda_trans: {args.lambda_trans}\n")
        handle.write("\ntraining_history\n")
        for item in training_history:
            handle.write(f"epoch {item['epoch']}: loss={item['loss']:.6f}\n")


def main() -> None:
    args = build_argument_parser().parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_specs = core.default_dataset_specs()
    training_rows = core.collect_training_rows(dataset_specs)
    if not training_rows:
        raise ValueError("No normal training rows found.")

    vocabularies = core.prepare_vocabularies(
        training_rows=training_rows,
        vocab_path=Path(args.vocab_output),
        min_frequency=args.min_frequency,
        rebuild_vocab=args.rebuild_vocab,
        actor_field=args.actor_field,
    )
    field_vocab_sizes = {
        field_name: len(vocabularies["fields"][field_name]["token_to_id"])
        for field_name in core.VOCAB_FIELDS
    }
    entity_vocab_sizes = {
        entity_type: len(vocabularies["entities"][entity_type]["token_to_id"])
        for entity_type in core.ENTITY_NODE_TYPES
    }

    samples = core.build_graph_samples(dataset_specs=dataset_specs, vocabularies=vocabularies, actor_field=args.actor_field)
    training_samples = [
        sample
        for sample in samples
        if sample["split"] == "train" and int(sample["label"]) == 0
    ]
    if not training_samples:
        raise ValueError("No normal training graph windows found.")

    device = torch.device(args.device)
    mask_token_ids = core.event_mask_token_ids(vocabularies).to(device)
    model = core.HeteroEventPredictor(
        field_vocab_sizes=field_vocab_sizes,
        entity_vocab_sizes=entity_vocab_sizes,
        field_embedding_dim=args.field_embedding_dim,
        hidden_dim=args.hidden_dim,
        rgcn_layers=args.rgcn_layers,
        dropout=args.dropout,
    ).to(device)

    training_history = core.train_model(
        model=model,
        training_samples=training_samples,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        mask_token_ids=mask_token_ids,
        mask_ratio=args.mask_ratio,
        lambda_rel=args.lambda_rel,
        lambda_trans=args.lambda_trans,
    )

    scored_training = core.score_all_samples(
        model=model,
        samples=training_samples,
        device=device,
        window_score_reduction=args.window_score_reduction,
        top_k=args.top_k,
        mask_token_ids=mask_token_ids,
        score_mask_batch_size=args.score_mask_batch_size,
        lambda_rel=args.lambda_rel,
        lambda_trans=args.lambda_trans,
    )
    training_scores = [float(sample["score"]) for sample in scored_training]
    threshold = core.quantile_threshold(training_scores, nu=args.nu)

    model_path = Path(args.model_output)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "node_types": list(core.NODE_TYPES),
            "event_input_fields": list(core.EVENT_INPUT_FIELDS),
            "target_fields": list(core.TARGET_FIELDS),
            "edge_types": list(core.EDGE_TYPES),
            "objective": "two_view_field_relation_transition_prediction",
            "mask_ratio": args.mask_ratio,
            "lambda_rel": args.lambda_rel,
            "lambda_trans": args.lambda_trans,
            "score_mask_batch_size": args.score_mask_batch_size,
            "resource_key_schema": core.RESOURCE_KEY_SCHEMA,
            "field_embedding_dim": args.field_embedding_dim,
            "hidden_dim": args.hidden_dim,
            "rgcn_layers": args.rgcn_layers,
            "dropout": args.dropout,
            "actor_field": args.actor_field,
            "training_history": training_history,
        },
        model_path,
    )

    threshold_path = Path(args.threshold_output)
    threshold_path.parent.mkdir(parents=True, exist_ok=True)
    with threshold_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "threshold": threshold,
                "nu": args.nu,
                "mask_ratio": args.mask_ratio,
                "lambda_rel": args.lambda_rel,
                "lambda_trans": args.lambda_trans,
                "score_mask_batch_size": args.score_mask_batch_size,
                "window_score_reduction": args.window_score_reduction,
                "top_k": args.top_k,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    core.write_window_scores(Path(args.train_scores_output), scored_training)
    write_train_report(Path(args.report_output), args=args, threshold=threshold, training_history=training_history)
    print(f"[graph_hete_train] saved model: {model_path}")
    print(f"[graph_hete_train] saved threshold: {threshold_path} ({threshold:.6f})")


if __name__ == "__main__":
    main()
