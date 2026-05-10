"""
Load a trained heterogeneous graph model, evaluate test windows, and export one
case study for the highest-scoring correctly detected anomalous window.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config
import graph_hete_end_v2 as core


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test heterogeneous graph anomaly detector and export a case study.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", default="cuda:1", help="Torch device, for example cpu, cuda, or cuda:1.")
    parser.add_argument("--actor-field", default=None, help="Override Actor field; default comes from checkpoint.")
    parser.add_argument("--vocab-input", default=str(Path(config.arithmetic_path) / "graph_hete_train_vocab.json"))
    parser.add_argument("--model-input", default=str(Path(config.arithmetic_path) / "graph_hete_train_model.pt"))
    parser.add_argument("--threshold-input", default=str(Path(config.arithmetic_path) / "graph_hete_train_threshold.json"))
    parser.add_argument("--window-scores-output", default=str(Path(config.arithmetic_path) / "graph_hete_test_window_scores.jsonl"))
    parser.add_argument("--metrics-output", default=str(Path(config.arithmetic_path) / "graph_hete_test_metrics.csv"))
    parser.add_argument("--case-output-dir", default=str(Path(config.arithmetic_path) / "case_study"))
    parser.add_argument("--case-report-output", default=str(Path(config.arithmetic_path) / "case_study" / "case_study.txt"))
    parser.add_argument("--score-mask-batch-size", type=int, default=None, help="Override scoring batch size.")
    parser.add_argument("--window-score-reduction", choices=["max", "mean", "meanmax", "topk"], default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--lambda-rel", type=float, default=None)
    parser.add_argument("--lambda-trans", type=float, default=None)
    parser.add_argument("--max-case-nodes", type=int, default=40, help="Maximum nodes included in the exported local graph.")
    parser.add_argument(
        "--case-per-dataset",
        action="store_true",
        help="Export one highest-scoring case-study graph for each test dataset.",
    )
    return parser


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_model_and_config(args: argparse.Namespace, device: torch.device):
    checkpoint = torch.load(args.model_input, map_location=device)
    threshold_config = load_json(Path(args.threshold_input))
    vocabularies = core.load_vocabularies(Path(args.vocab_input))

    field_vocab_sizes = {
        field_name: len(vocabularies["fields"][field_name]["token_to_id"])
        for field_name in core.VOCAB_FIELDS
    }
    entity_vocab_sizes = {
        entity_type: len(vocabularies["entities"][entity_type]["token_to_id"])
        for entity_type in core.ENTITY_NODE_TYPES
    }

    model = core.HeteroEventPredictor(
        field_vocab_sizes=field_vocab_sizes,
        entity_vocab_sizes=entity_vocab_sizes,
        field_embedding_dim=int(checkpoint["field_embedding_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        rgcn_layers=int(checkpoint["rgcn_layers"]),
        dropout=float(checkpoint["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    runtime_config = {
        "threshold": float(threshold_config["threshold"]),
        "lambda_rel": float(args.lambda_rel if args.lambda_rel is not None else threshold_config.get("lambda_rel", checkpoint.get("lambda_rel", 0.5))),
        "lambda_trans": float(args.lambda_trans if args.lambda_trans is not None else threshold_config.get("lambda_trans", checkpoint.get("lambda_trans", 0.5))),
        "score_mask_batch_size": int(args.score_mask_batch_size if args.score_mask_batch_size is not None else threshold_config.get("score_mask_batch_size", checkpoint.get("score_mask_batch_size", 32))),
        "window_score_reduction": str(args.window_score_reduction if args.window_score_reduction is not None else threshold_config.get("window_score_reduction", "max")),
        "top_k": int(args.top_k if args.top_k is not None else threshold_config.get("top_k", 3)),
        "actor_field": str(args.actor_field if args.actor_field is not None else checkpoint.get("actor_field", "actor_id")),
    }
    return model, vocabularies, checkpoint, runtime_config


def score_graph_sample_with_components(
    model: core.HeteroEventPredictor,
    sample: Dict[str, object],
    device: torch.device,
    mask_token_ids: torch.Tensor,
    score_mask_batch_size: int,
    lambda_rel: float,
    lambda_trans: float,
) -> List[Dict[str, float]]:
    device_sample = core.graph_sample_to_device(sample, device)
    event_count = device_sample["event_input_ids"].size(0)
    results = [
        {"field_loss": 0.0, "relation_loss": 0.0, "transition_loss": 0.0, "score": 0.0}
        for _ in range(event_count)
    ]
    batch_size = max(1, int(score_mask_batch_size))

    with torch.no_grad():
        for start in range(0, event_count, batch_size):
            end = min(start + batch_size, event_count)
            masked_event_indices = torch.arange(start, end, dtype=torch.long, device=device)

            field_event_input_ids, field_edge_index, field_edge_type = core.prepare_field_relation_view(
                device_sample=device_sample,
                masked_event_indices=masked_event_indices,
                mask_token_ids=mask_token_ids,
            )
            field_outputs = model(
                event_input_ids=field_event_input_ids,
                node_type_ids=device_sample["node_type_ids"],
                node_token_ids=device_sample["node_token_ids"],
                event_node_indices=device_sample["event_node_indices"],
                edge_index=field_edge_index,
                edge_type=field_edge_type,
            )
            field_loss = core.field_prediction_loss(
                model_outputs=field_outputs,
                device_sample=device_sample,
                masked_event_indices=masked_event_indices,
                reduction="none",
            )
            relation_loss = core.relation_prediction_loss(
                model=model,
                model_outputs=field_outputs,
                device_sample=device_sample,
                masked_event_indices=masked_event_indices,
                reduction="none",
            )

            transition_event_input_ids, transition_edge_index, transition_edge_type = core.prepare_transition_view(
                device_sample=device_sample,
                masked_event_indices=masked_event_indices,
                mask_token_ids=mask_token_ids,
            )
            transition_outputs = model(
                event_input_ids=transition_event_input_ids,
                node_type_ids=device_sample["node_type_ids"],
                node_token_ids=device_sample["node_token_ids"],
                event_node_indices=device_sample["event_node_indices"],
                edge_index=transition_edge_index,
                edge_type=transition_edge_type,
            )
            transition_loss = core.transition_prediction_loss(
                model_outputs=transition_outputs,
                device_sample=device_sample,
                masked_event_indices=masked_event_indices,
                reduction="none",
            )

            combined = field_loss + float(lambda_rel) * relation_loss + float(lambda_trans) * transition_loss
            for offset in range(end - start):
                event_idx = start + offset
                results[event_idx] = {
                    "field_loss": float(field_loss[offset].detach().cpu().item()),
                    "relation_loss": float(relation_loss[offset].detach().cpu().item()),
                    "transition_loss": float(transition_loss[offset].detach().cpu().item()),
                    "score": float(combined[offset].detach().cpu().item()),
                }
    return results


def read_rows_by_dataset(dataset_specs: Sequence[Dict[str, Path]]) -> Dict[str, List[Dict[str, str]]]:
    return {str(spec["name"]): core.read_csv_rows(spec["csv_path"]) for spec in dataset_specs}


def dot_escape(text: object) -> str:
    return str(text).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def token_for_node(sample: Dict[str, object], vocabularies: Dict[str, Dict], node_id: int) -> str:
    node_type = core.NODE_TYPES[int(sample["node_type_ids"][node_id].item())]
    token_id = int(sample["node_token_ids"][node_id].item())
    if node_type == "Event":
        return f"event_{node_id}"
    id_to_token = vocabularies["entities"][node_type]["id_to_token"]
    return str(id_to_token.get(str(token_id), core.UNK_TOKEN))


def event_label(row: Dict[str, str], event_index: int, component: Dict[str, float]) -> str:
    pieces = [
        f"E{event_index}",
        f"score={component['score']:.4f}",
        f"verb={row.get('verb', '')}",
        f"ns={row.get('namespace', '')}",
        f"resource={row.get('resource_type', '')}",
        f"decision={row.get('decision', '')}",
        f"status={row.get('status_code', '')}",
    ]
    return "\\n".join(pieces)


def write_case_graph(
    sample: Dict[str, object],
    scored_sample: Dict[str, object],
    components: List[Dict[str, float]],
    vocabularies: Dict[str, Dict],
    dataset_rows: List[Dict[str, str]],
    case_event_index: int,
    output_dir: Path,
    max_nodes: int,
    graph_title: str,
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dot_path = output_dir / "case_study_local_graph.dot"
    svg_path = output_dir / "case_study_local_graph.svg"

    edge_index = sample["edge_index"]
    edge_type = sample["edge_type"]
    event_count = int(sample["event_count"])
    selected_nodes = {int(case_event_index)}
    incident_edges = []

    for edge_pos in range(edge_type.size(0)):
        src = int(edge_index[0, edge_pos].item())
        dst = int(edge_index[1, edge_pos].item())
        if src == case_event_index or dst == case_event_index:
            incident_edges.append(edge_pos)
            selected_nodes.add(src)
            selected_nodes.add(dst)

    if len(selected_nodes) > max_nodes:
        selected_nodes = {case_event_index} | set(sorted(selected_nodes - {case_event_index})[: max_nodes - 1])

    selected_edges = []
    for edge_pos in range(edge_type.size(0)):
        src = int(edge_index[0, edge_pos].item())
        dst = int(edge_index[1, edge_pos].item())
        if src in selected_nodes and dst in selected_nodes:
            selected_edges.append(edge_pos)

    start_idx = int(sample["start_idx"])
    with dot_path.open("w", encoding="utf-8") as handle:
        handle.write("digraph case_study {\n")
        handle.write('  rankdir=LR;\n')
        handle.write(f'  graph [fontsize=10, labelloc="t", label="{dot_escape(graph_title)}"];\n')
        handle.write('  node [shape=box, style="rounded,filled", fontsize=9];\n')
        handle.write('  edge [fontsize=8];\n')
        for node_id in sorted(selected_nodes):
            node_type = core.NODE_TYPES[int(sample["node_type_ids"][node_id].item())]
            if node_id < event_count:
                row = dataset_rows[start_idx + node_id]
                label = event_label(row, node_id, components[node_id])
                fill = "#ffb3b3" if node_id == case_event_index else "#ffe0b2"
            else:
                token = token_for_node(sample, vocabularies, node_id)
                label = f"{node_type}\\n{token}"
                fill = "#d9ecff"
            handle.write(f'  n{node_id} [label="{dot_escape(label)}", fillcolor="{fill}"];\n')
        for edge_pos in selected_edges:
            src = int(edge_index[0, edge_pos].item())
            dst = int(edge_index[1, edge_pos].item())
            relation = core.EDGE_TYPES[int(edge_type[edge_pos].item())]
            handle.write(f'  n{src} -> n{dst} [label="{dot_escape(relation)}"];\n')
        handle.write("}\n")

    if shutil.which("dot"):
        subprocess.run(["dot", "-Tsvg", str(dot_path), "-o", str(svg_path)], check=False)

    return {"dot": str(dot_path), "svg": str(svg_path) if svg_path.exists() else ""}


def write_case_report(
    output_path: Path,
    case_sample: Dict[str, object],
    scored_sample: Dict[str, object],
    components: List[Dict[str, float]],
    dataset_rows: List[Dict[str, str]],
    case_event_index: int,
    threshold: float,
    runtime_config: Dict[str, object],
    graph_paths: Dict[str, str],
    selection_label: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    case_component = components[case_event_index]
    row = dataset_rows[int(case_sample["start_idx"]) + case_event_index]
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(f"case study: {selection_label}\n")
        handle.write("\n判定标准\n")
        handle.write("- window is a true positive when label=1 and prediction=1.\n")
        handle.write("- prediction=1 iff window_score > threshold.\n")
        handle.write(
            "- event_score = field_loss + lambda_rel * relation_loss + lambda_trans * transition_loss.\n"
        )
        handle.write(f"- window_score_reduction: {runtime_config['window_score_reduction']}\n")
        handle.write(f"- threshold: {threshold:.6f}\n")
        handle.write(f"- lambda_rel: {runtime_config['lambda_rel']}\n")
        handle.write(f"- lambda_trans: {runtime_config['lambda_trans']}\n")

        handle.write("\n窗口信息\n")
        handle.write(f"- selection: {selection_label}\n")
        handle.write(f"- dataset: {case_sample['dataset']}\n")
        handle.write(f"- window_id: {case_sample['window_id']}\n")
        handle.write(f"- start_idx: {case_sample['start_idx']}\n")
        handle.write(f"- end_idx: {case_sample['end_idx']}\n")
        handle.write(f"- label: {case_sample['label']}\n")
        handle.write(f"- prediction: {scored_sample['prediction']}\n")
        handle.write(f"- window_score: {float(scored_sample['score']):.6f}\n")
        handle.write(f"- event_score_max: {float(scored_sample['event_score_max']):.6f}\n")
        handle.write(f"- event_score_mean: {float(scored_sample['event_score_mean']):.6f}\n")

        handle.write("\n最高分局部事件\n")
        handle.write(f"- local_event_index: {case_event_index}\n")
        handle.write(f"- global_row_index: {int(case_sample['start_idx']) + case_event_index}\n")
        handle.write(f"- combined_event_score: {case_component['score']:.6f}\n")
        handle.write(f"- field_loss: {case_component['field_loss']:.6f}\n")
        handle.write(f"- relation_loss: {case_component['relation_loss']:.6f}\n")
        handle.write(f"- transition_loss: {case_component['transition_loss']:.6f}\n")
        for field_name in ["verb", "namespace", "resource_type", "resource", "subresource", "decision", "status_code", "actor_id", "request_uri"]:
            handle.write(f"- {field_name}: {row.get(field_name, '')}\n")

        handle.write("\n图文件\n")
        handle.write(f"- dot: {graph_paths.get('dot', '')}\n")
        if graph_paths.get("svg"):
            handle.write(f"- svg: {graph_paths['svg']}\n")
        else:
            handle.write("- svg: not generated because graphviz dot was unavailable\n")


def select_global_case(predicted_samples: Sequence[Dict[str, object]]) -> Tuple[Dict[str, object], str]:
    true_positives = [
        sample
        for sample in predicted_samples
        if int(sample["label"]) == 1 and int(sample["prediction"]) == 1
    ]
    if not true_positives:
        raise ValueError("No true-positive anomalous window found; case study not generated.")
    return max(true_positives, key=lambda item: float(item["score"])), "highest-score correctly detected anomaly window"


def select_dataset_cases(
    predicted_samples: Sequence[Dict[str, object]],
    dataset_order: Sequence[str],
) -> List[Tuple[str, Dict[str, object], str]]:
    cases: List[Tuple[str, Dict[str, object], str]] = []
    for dataset_name in dataset_order:
        dataset_samples = [sample for sample in predicted_samples if str(sample["dataset"]) == dataset_name]
        if not dataset_samples:
            continue

        true_positives = [
            sample
            for sample in dataset_samples
            if int(sample["label"]) == 1 and int(sample["prediction"]) == 1
        ]
        if true_positives:
            cases.append(
                (
                    dataset_name,
                    max(true_positives, key=lambda item: float(item["score"])),
                    "highest-score correctly detected anomaly window",
                )
            )
            continue

        anomalous_samples = [sample for sample in dataset_samples if int(sample["label"]) == 1]
        if anomalous_samples:
            cases.append(
                (
                    dataset_name,
                    max(anomalous_samples, key=lambda item: float(item["score"])),
                    "highest-score anomaly window without true-positive detection",
                )
            )
            continue

        print(f"[graph_hete_test] dataset={dataset_name} has no anomalous window; case study skipped.")
    return cases


def export_case_study(
    case_scored_sample: Dict[str, object],
    selection_label: str,
    sample_lookup: Dict[Tuple[str, int], Dict[str, object]],
    rows_by_dataset: Dict[str, List[Dict[str, str]]],
    model: core.HeteroEventPredictor,
    vocabularies: Dict[str, Dict],
    device: torch.device,
    mask_token_ids: torch.Tensor,
    runtime_config: Dict[str, object],
    threshold: float,
    output_dir: Path,
    report_output: Path,
    max_case_nodes: int,
) -> Dict[str, str]:
    case_sample = sample_lookup[(str(case_scored_sample["dataset"]), int(case_scored_sample["window_id"]))]
    components = score_graph_sample_with_components(
        model=model,
        sample=case_sample,
        device=device,
        mask_token_ids=mask_token_ids,
        score_mask_batch_size=int(runtime_config["score_mask_batch_size"]),
        lambda_rel=float(runtime_config["lambda_rel"]),
        lambda_trans=float(runtime_config["lambda_trans"]),
    )
    case_event_index = max(range(len(components)), key=lambda index: components[index]["score"])
    dataset_rows = rows_by_dataset[str(case_sample["dataset"])]
    graph_title = f"{case_sample['dataset']}: {selection_label}"
    graph_paths = write_case_graph(
        sample=case_sample,
        scored_sample=case_scored_sample,
        components=components,
        vocabularies=vocabularies,
        dataset_rows=dataset_rows,
        case_event_index=case_event_index,
        output_dir=output_dir,
        max_nodes=max_case_nodes,
        graph_title=graph_title,
    )
    write_case_report(
        output_path=report_output,
        case_sample=case_sample,
        scored_sample=case_scored_sample,
        components=components,
        dataset_rows=dataset_rows,
        case_event_index=case_event_index,
        threshold=threshold,
        runtime_config=runtime_config,
        graph_paths=graph_paths,
        selection_label=selection_label,
    )
    return graph_paths


def main() -> None:
    args = build_argument_parser().parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    model, vocabularies, _checkpoint, runtime_config = load_model_and_config(args, device)
    threshold = float(runtime_config["threshold"])
    mask_token_ids = core.event_mask_token_ids(vocabularies).to(device)

    dataset_specs = core.default_dataset_specs()
    samples = core.build_graph_samples(
        dataset_specs=dataset_specs,
        vocabularies=vocabularies,
        actor_field=str(runtime_config["actor_field"]),
    )
    test_samples = [sample for sample in samples if sample["split"] == "test"]
    scored_samples = core.score_all_samples(
        model=model,
        samples=test_samples,
        device=device,
        window_score_reduction=str(runtime_config["window_score_reduction"]),
        top_k=int(runtime_config["top_k"]),
        mask_token_ids=mask_token_ids,
        score_mask_batch_size=int(runtime_config["score_mask_batch_size"]),
        lambda_rel=float(runtime_config["lambda_rel"]),
        lambda_trans=float(runtime_config["lambda_trans"]),
    )
    predicted_samples = core.attach_predictions(scored_samples=scored_samples, threshold=threshold)
    metrics_rows = core.summarize_metrics(predicted_samples, threshold=threshold)
    core.write_window_scores(Path(args.window_scores_output), predicted_samples)
    core.write_metrics(Path(args.metrics_output), metrics_rows)

    sample_lookup = {
        (str(sample["dataset"]), int(sample["window_id"])): sample
        for sample in test_samples
    }
    rows_by_dataset = read_rows_by_dataset(dataset_specs)

    exported_graphs: List[Tuple[str, Dict[str, str]]] = []
    if args.case_per_dataset:
        dataset_order = [
            str(spec["name"])
            for spec in dataset_specs
            if str(spec.get("split", "")) == "test"
        ]
        dataset_cases = select_dataset_cases(predicted_samples, dataset_order=dataset_order)
        if not dataset_cases:
            print("[graph_hete_test] no anomalous windows found; case studies not generated.")
        for dataset_name, case_scored_sample, selection_label in dataset_cases:
            dataset_output_dir = Path(args.case_output_dir) / dataset_name
            dataset_report_output = dataset_output_dir / "case_study.txt"
            graph_paths = export_case_study(
                case_scored_sample=case_scored_sample,
                selection_label=selection_label,
                sample_lookup=sample_lookup,
                rows_by_dataset=rows_by_dataset,
                model=model,
                vocabularies=vocabularies,
                device=device,
                mask_token_ids=mask_token_ids,
                runtime_config=runtime_config,
                threshold=threshold,
                output_dir=dataset_output_dir,
                report_output=dataset_report_output,
                max_case_nodes=int(args.max_case_nodes),
            )
            exported_graphs.append((dataset_name, graph_paths))
    else:
        try:
            case_scored_sample, selection_label = select_global_case(predicted_samples)
        except ValueError as exc:
            print(f"[graph_hete_test] {exc}")
            return
        graph_paths = export_case_study(
            case_scored_sample=case_scored_sample,
            selection_label=selection_label,
            sample_lookup=sample_lookup,
            rows_by_dataset=rows_by_dataset,
            model=model,
            vocabularies=vocabularies,
            device=device,
            mask_token_ids=mask_token_ids,
            runtime_config=runtime_config,
            threshold=threshold,
            output_dir=Path(args.case_output_dir),
            report_output=Path(args.case_report_output),
            max_case_nodes=int(args.max_case_nodes),
        )
        exported_graphs.append(("global", graph_paths))

    for row in metrics_rows:
        print(
            f"[graph_hete_test] {row['dataset']}: precision={row['precision']:.4f}, "
            f"recall={row['recall']:.4f}, f1={row['f1']:.4f}, fpr={row['fpr']:.4f}"
        )
    for case_name, graph_paths in exported_graphs:
        print(f"[graph_hete_test] case={case_name} graph dot: {graph_paths.get('dot', '')}")
        if graph_paths.get("svg"):
            print(f"[graph_hete_test] case={case_name} graph svg: {graph_paths['svg']}")


if __name__ == "__main__":
    main()
