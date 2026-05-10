"""
Heterogeneous event graph anomaly detection for Kubernetes audit windows.

The graph makes audit-log relations explicit instead of leaving them only inside
Event node features. Node types:
- Event
- Actor
- Resource
- Namespace
- Verb

Logical edge types:
- Actor -> Event
- Event -> Actor
- Event -> Resource
- Resource -> Event
- Event -> Namespace
- Namespace -> Event
- Event -> Verb
- Verb -> Event
- Resource -> Namespace
- Namespace -> Resource
- Event -> Event same_actor_next / same_actor_prev

The model is trained with partial masked key-field prediction on normal windows.
When an Event is masked, its directly exposed target fields are masked and its
target-revealing Event -> Resource/Namespace/Verb edges are hidden for that
forward pass. It also predicts the masked Event's Verb/Resource/Namespace nodes
and the same-actor next Event behavior when such a successor exists.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch import nn

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config


UNK_TOKEN = "__UNK__"
EMPTY_TOKEN = "__EMPTY__"
MASK_TOKEN = "__MASK__"
UNKNOWN_ACTOR = "__UNKNOWN_ACTOR__"
UNKNOWN_RESOURCE = "__UNKNOWN_RESOURCE__"
RESOURCE_KEY_SCHEMA = "api_group|namespace|resource_type|resource|subresource|resource_instance_or_name"

NODE_TYPES: Tuple[str, ...] = ("Event", "Actor", "Resource", "Namespace", "Verb")
ENTITY_NODE_TYPES: Tuple[str, ...] = ("Actor", "Resource", "Namespace", "Verb")
NODE_TYPE_TO_ID = {node_type: index for index, node_type in enumerate(NODE_TYPES)}

EVENT_INPUT_FIELDS: Tuple[str, ...] = (
    "api_group",
    "is_resource_request",
    "subresource",
    "decision",
    "status_code",
    "stage",
    "verb",
)

TARGET_FIELDS: Tuple[str, ...] = (
    "verb",
    "decision",
    "status_code",
)
EVENT_TARGET_FIELD_INDICES: Tuple[int, ...] = tuple(
    EVENT_INPUT_FIELDS.index(field_name)
    for field_name in TARGET_FIELDS
    if field_name in EVENT_INPUT_FIELDS
)

VOCAB_FIELDS: Tuple[str, ...] = tuple(dict.fromkeys(EVENT_INPUT_FIELDS + TARGET_FIELDS))

EDGE_TYPES: Tuple[str, ...] = (
    "actor_to_event",
    "event_to_actor",
    "event_to_resource",
    "resource_to_event",
    "event_to_namespace",
    "namespace_to_event",
    "event_to_verb",
    "verb_to_event",
    "resource_to_namespace",
    "namespace_to_resource",
    "same_actor_next",
    "same_actor_prev",
)
EDGE_TYPE_TO_ID = {edge_type: index for index, edge_type in enumerate(EDGE_TYPES)}
TARGET_REVEALING_EDGE_TYPES = {
    EDGE_TYPE_TO_ID["event_to_resource"],
    EDGE_TYPE_TO_ID["resource_to_event"],
    EDGE_TYPE_TO_ID["event_to_namespace"],
    EDGE_TYPE_TO_ID["namespace_to_event"],
    EDGE_TYPE_TO_ID["event_to_verb"],
    EDGE_TYPE_TO_ID["verb_to_event"],
}


def mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def pstdev(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        return 0.0
    avg = mean(items)
    return math.sqrt(sum((value - avg) ** 2 for value in items) / len(items))


def default_dataset_specs() -> List[Dict[str, Path]]:
    return [
        {
            "name": "normal_log_new",
            "split": "train",
            "csv_path": ROOT_DIR / config.csv_path / config.normal_log_new_filter,
            "window_path": ROOT_DIR / config.traintest_path / config.normal_log_new_window,
        },
        {
            "name": "audit_abuse_dev",
            "split": "test",
            "csv_path": ROOT_DIR / config.csv_path / config.audit_abuse_dev_filter,
            "window_path": ROOT_DIR / config.traintest_path / config.audit_abuse_dev_window,
        },
        {
            "name": "audit_abuse_ops",
            "split": "test",
            "csv_path": ROOT_DIR / config.csv_path / config.audit_abuse_ops_filter,
            "window_path": ROOT_DIR / config.traintest_path / config.audit_abuse_ops_window,
        },
        {
            "name": "audit_abuse_sys",
            "split": "test",
            "csv_path": ROOT_DIR / config.csv_path / config.audit_abuse_sys_filter,
            "window_path": ROOT_DIR / config.traintest_path / config.audit_abuse_sys_window,
        },
        {
            "name": "dev_escape",
            "split": "test",
            "csv_path": ROOT_DIR / config.csv_path / config.dev_escape_filter,
            "window_path": ROOT_DIR / config.traintest_path / config.dev_escape_window,
        },
        {
            "name": "ops_escape",
            "split": "test",
            "csv_path": ROOT_DIR / config.csv_path / config.ops_escape_filter,
            "window_path": ROOT_DIR / config.traintest_path / config.ops_escape_window,
        },
        {
            "name": "kubeconfig_node2",
            "split": "test",
            "csv_path": ROOT_DIR / config.csv_path / config.kubeconfig_node2_filter,
            "window_path": ROOT_DIR / config.traintest_path / config.kubeconfig_node2_window,
        },
        {
            "name": "kubeconfig_node3",
            "split": "test",
            "csv_path": ROOT_DIR / config.csv_path / config.kubeconfig_node3_filter,
            "window_path": ROOT_DIR / config.traintest_path / config.kubeconfig_node3_window,
        },
    ]


def read_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def read_window_indices(window_path: Path) -> List[Dict[str, int]]:
    windows: List[Dict[str, int]] = []
    with window_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            parts = [item for item in stripped.replace(",", " ").split() if item]
            if len(parts) < 3:
                continue
            windows.append(
                {
                    "start_idx": int(parts[0]),
                    "end_idx": int(parts[1]),
                    "label": int(parts[2]),
                }
            )
    return windows


def normalize_value(value: str) -> str:
    text = (value or "").strip()
    return text if text else EMPTY_TOKEN


def actor_key_from_row(row: Dict[str, str], actor_field: str) -> str:
    actor_value = normalize_value(row.get(actor_field, ""))
    return UNKNOWN_ACTOR if actor_value == EMPTY_TOKEN else actor_value


def resource_key_from_row(row: Dict[str, str]) -> str:
    resource_instance_or_name = normalize_value(row.get("resource_instance", ""))
    if resource_instance_or_name == EMPTY_TOKEN:
        resource_instance_or_name = normalize_value(row.get("resource_name", ""))

    pieces = [
        normalize_value(row.get("api_group", "")),
        normalize_value(row.get("namespace", "")),
        normalize_value(row.get("resource_type", "")),
        normalize_value(row.get("resource", "")),
        normalize_value(row.get("subresource", "")),
        resource_instance_or_name,
    ]
    if all(piece == EMPTY_TOKEN for piece in pieces):
        return UNKNOWN_RESOURCE
    return "|".join(pieces)


def namespace_key_from_row(row: Dict[str, str]) -> str:
    return normalize_value(row.get("namespace", ""))


def verb_key_from_row(row: Dict[str, str]) -> str:
    return normalize_value(row.get("verb", ""))


def entity_value_from_row(row: Dict[str, str], entity_type: str, actor_field: str) -> str:
    if entity_type == "Actor":
        return actor_key_from_row(row, actor_field)
    if entity_type == "Resource":
        return resource_key_from_row(row)
    if entity_type == "Namespace":
        return namespace_key_from_row(row)
    if entity_type == "Verb":
        return verb_key_from_row(row)
    raise ValueError(f"Unsupported entity type: {entity_type}")


def collect_training_rows(dataset_specs: Sequence[Dict[str, Path]]) -> List[Dict[str, str]]:
    training_rows: List[Dict[str, str]] = []
    for spec in dataset_specs:
        if spec["split"] != "train":
            continue
        rows = read_csv_rows(spec["csv_path"])
        windows = read_window_indices(spec["window_path"])
        for window in windows:
            if int(window["label"]) != 0:
                continue
            start_idx = max(0, int(window["start_idx"]))
            end_idx = min(len(rows) - 1, int(window["end_idx"]))
            if end_idx >= start_idx:
                training_rows.extend(rows[start_idx : end_idx + 1])
    return training_rows


def build_vocabularies(
    training_rows: Sequence[Dict[str, str]],
    min_frequency: int,
    actor_field: str,
) -> Dict[str, Dict]:
    field_counters: Dict[str, Counter] = {field_name: Counter() for field_name in VOCAB_FIELDS}
    entity_counters: Dict[str, Counter] = {entity_type: Counter() for entity_type in ENTITY_NODE_TYPES}

    for row in training_rows:
        for field_name in VOCAB_FIELDS:
            field_counters[field_name][normalize_value(row.get(field_name, ""))] += 1
        for entity_type in ENTITY_NODE_TYPES:
            entity_counters[entity_type][entity_value_from_row(row, entity_type, actor_field)] += 1

    vocabularies: Dict[str, Dict] = {
        "event_input_fields": list(EVENT_INPUT_FIELDS),
        "target_fields": list(TARGET_FIELDS),
        "node_types": list(NODE_TYPES),
        "edge_types": list(EDGE_TYPES),
        "fields": {},
        "entities": {},
        "metadata": {
            "min_frequency": int(min_frequency),
            "resource_key_schema": RESOURCE_KEY_SCHEMA,
        },
    }

    for field_name in VOCAB_FIELDS:
        token_to_id = {UNK_TOKEN: 0, EMPTY_TOKEN: 1, MASK_TOKEN: 2}
        for token, count in field_counters[field_name].most_common():
            if token in token_to_id or count < min_frequency:
                continue
            token_to_id[token] = len(token_to_id)
        vocabularies["fields"][field_name] = {
            "token_to_id": token_to_id,
            "id_to_token": {str(index): token for token, index in token_to_id.items()},
        }

    for entity_type in ENTITY_NODE_TYPES:
        token_to_id = {UNK_TOKEN: 0, EMPTY_TOKEN: 1}
        for token, count in entity_counters[entity_type].most_common():
            if token in token_to_id or count < min_frequency:
                continue
            token_to_id[token] = len(token_to_id)
        vocabularies["entities"][entity_type] = {
            "token_to_id": token_to_id,
            "id_to_token": {str(index): token for token, index in token_to_id.items()},
        }

    return vocabularies


def load_vocabularies(vocab_path: Path) -> Dict[str, Dict]:
    with vocab_path.open("r", encoding="utf-8") as handle:
        vocabularies = json.load(handle)
    if tuple(vocabularies.get("event_input_fields", [])) != EVENT_INPUT_FIELDS:
        raise ValueError("Saved event input fields do not match graph_hete EVENT_INPUT_FIELDS.")
    if tuple(vocabularies.get("target_fields", [])) != TARGET_FIELDS:
        raise ValueError("Saved target fields do not match graph_hete TARGET_FIELDS.")
    if tuple(vocabularies.get("node_types", [])) != NODE_TYPES:
        raise ValueError("Saved node types do not match graph_hete NODE_TYPES.")
    if tuple(vocabularies.get("edge_types", [])) != EDGE_TYPES:
        raise ValueError("Saved edge types do not match graph_hete EDGE_TYPES.")
    saved_resource_key_schema = vocabularies.get("metadata", {}).get("resource_key_schema")
    if saved_resource_key_schema != RESOURCE_KEY_SCHEMA:
        raise ValueError("Saved resource key schema does not match current graph_hete resource key schema.")
    for field_name in VOCAB_FIELDS:
        token_to_id = vocabularies["fields"][field_name]["token_to_id"]
        if MASK_TOKEN not in token_to_id:
            raise ValueError("Saved vocabulary does not contain MASK_TOKEN.")
    return vocabularies


def save_vocabularies(vocab_path: Path, vocabularies: Dict[str, Dict]) -> None:
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    with vocab_path.open("w", encoding="utf-8") as handle:
        json.dump(vocabularies, handle, ensure_ascii=False, indent=2)


def prepare_vocabularies(
    training_rows: Sequence[Dict[str, str]],
    vocab_path: Path,
    min_frequency: int,
    rebuild_vocab: bool,
    actor_field: str,
) -> Dict[str, Dict]:
    if not rebuild_vocab and vocab_path.exists():
        try:
            vocabularies = load_vocabularies(vocab_path)
            saved_min_frequency = vocabularies.get("metadata", {}).get("min_frequency")
            if saved_min_frequency is not None and int(saved_min_frequency) != int(min_frequency):
                raise ValueError("Saved min_frequency does not match current min_frequency.")
            return vocabularies
        except Exception as exc:
            print(f"[graph_hete_add] Existing vocabulary is not reusable: {exc}. Rebuilding vocabulary.")

    vocabularies = build_vocabularies(
        training_rows=training_rows,
        min_frequency=min_frequency,
        actor_field=actor_field,
    )
    save_vocabularies(vocab_path, vocabularies)
    return vocabularies


def encode_event_input_fields(row: Dict[str, str], vocabularies: Dict[str, Dict]) -> List[int]:
    field_ids: List[int] = []
    for field_name in EVENT_INPUT_FIELDS:
        token_to_id = vocabularies["fields"][field_name]["token_to_id"]
        token = normalize_value(row.get(field_name, ""))
        field_ids.append(int(token_to_id.get(token, token_to_id[UNK_TOKEN])))
    return field_ids


def encode_target_fields(row: Dict[str, str], vocabularies: Dict[str, Dict]) -> Dict[str, int]:
    target_ids: Dict[str, int] = {}
    for field_name in TARGET_FIELDS:
        token_to_id = vocabularies["fields"][field_name]["token_to_id"]
        token = normalize_value(row.get(field_name, ""))
        target_ids[field_name] = int(token_to_id.get(token, token_to_id[UNK_TOKEN]))
    return target_ids


def event_mask_token_ids(vocabularies: Dict[str, Dict]) -> torch.Tensor:
    mask_ids = [0 for _ in EVENT_INPUT_FIELDS]
    for field_index, field_name in enumerate(EVENT_INPUT_FIELDS):
        if field_name in TARGET_FIELDS:
            mask_ids[field_index] = int(vocabularies["fields"][field_name]["token_to_id"][MASK_TOKEN])
    return torch.tensor(mask_ids, dtype=torch.long)


def apply_event_field_mask(
    event_input_ids: torch.Tensor,
    masked_event_indices: torch.Tensor,
    mask_token_ids: torch.Tensor,
) -> None:
    if masked_event_indices.numel() == 0:
        return
    for field_index in EVENT_TARGET_FIELD_INDICES:
        event_input_ids[masked_event_indices, field_index] = mask_token_ids[field_index]


def add_edge(
    edges: List[Tuple[int, int, int, int]],
    src: int,
    dst: int,
    edge_type: str,
    mask_owner: int = -1,
) -> None:
    if src == dst:
        return
    edges.append((src, dst, EDGE_TYPE_TO_ID[edge_type], int(mask_owner)))


def build_graph_sample(
    dataset_name: str,
    split: str,
    window_id: int,
    window: Dict[str, int],
    rows: Sequence[Dict[str, str]],
    vocabularies: Dict[str, Dict],
    actor_field: str,
) -> Dict[str, object]:
    start_idx = max(0, int(window["start_idx"]))
    end_idx = min(len(rows) - 1, int(window["end_idx"]))
    window_rows = rows[start_idx : end_idx + 1] if end_idx >= start_idx else []

    node_type_ids: List[int] = [NODE_TYPE_TO_ID["Event"] for _ in window_rows]
    node_token_ids: List[int] = [0 for _ in window_rows]
    entity_nodes: Dict[Tuple[str, str], int] = {}

    def get_entity_node(entity_type: str, value: str) -> int:
        key = (entity_type, value)
        if key in entity_nodes:
            return entity_nodes[key]
        token_to_id = vocabularies["entities"][entity_type]["token_to_id"]
        node_id = len(node_type_ids)
        entity_nodes[key] = node_id
        node_type_ids.append(NODE_TYPE_TO_ID[entity_type])
        node_token_ids.append(int(token_to_id.get(value, token_to_id[UNK_TOKEN])))
        return node_id

    event_input_ids = [encode_event_input_fields(row, vocabularies) for row in window_rows]
    event_target_ids = [encode_target_fields(row, vocabularies) for row in window_rows]

    edges: List[Tuple[int, int, int, int]] = []
    dedup_edges = set()
    actor_positions: Dict[str, List[int]] = defaultdict(list)
    event_relation_node_ids: Dict[str, List[int]] = {
        "verb": [],
        "resource": [],
        "namespace": [],
    }
    next_event_exists = [0 for _ in window_rows]
    next_event_index = [-1 for _ in window_rows]
    next_event_targets: Dict[str, List[int]] = {
        field_name: [0 for _ in window_rows]
        for field_name in TARGET_FIELDS
    }

    def add_unique_edge(src: int, dst: int, edge_type: str) -> None:
        key = (src, dst, EDGE_TYPE_TO_ID[edge_type])
        if key in dedup_edges:
            return
        dedup_edges.add(key)
        add_edge(edges, src, dst, edge_type)

    for event_index, row in enumerate(window_rows):
        actor_key = actor_key_from_row(row, actor_field)
        actor_node = get_entity_node("Actor", actor_key)
        resource_node = get_entity_node("Resource", resource_key_from_row(row))
        namespace_node = get_entity_node("Namespace", namespace_key_from_row(row))
        verb_node = get_entity_node("Verb", verb_key_from_row(row))

        actor_positions[actor_key].append(event_index)
        event_relation_node_ids["verb"].append(verb_node)
        event_relation_node_ids["resource"].append(resource_node)
        event_relation_node_ids["namespace"].append(namespace_node)

        add_edge(edges, actor_node, event_index, "actor_to_event")
        add_edge(edges, event_index, actor_node, "event_to_actor")
        add_edge(edges, event_index, resource_node, "event_to_resource", mask_owner=event_index)
        add_edge(edges, resource_node, event_index, "resource_to_event", mask_owner=event_index)
        add_edge(edges, event_index, namespace_node, "event_to_namespace", mask_owner=event_index)
        add_edge(edges, namespace_node, event_index, "namespace_to_event", mask_owner=event_index)
        add_edge(edges, event_index, verb_node, "event_to_verb", mask_owner=event_index)
        add_edge(edges, verb_node, event_index, "verb_to_event", mask_owner=event_index)
        add_unique_edge(resource_node, namespace_node, "resource_to_namespace")
        add_unique_edge(namespace_node, resource_node, "namespace_to_resource")

    for positions in actor_positions.values():
        for left, right in zip(positions, positions[1:]):
            add_edge(edges, left, right, "same_actor_next")
            add_edge(edges, right, left, "same_actor_prev")
            next_event_exists[left] = 1
            next_event_index[left] = right
            for field_name in TARGET_FIELDS:
                next_event_targets[field_name][left] = event_target_ids[right][field_name]

    if edges:
        edge_index = torch.tensor([[src, dst] for src, dst, _, _ in edges], dtype=torch.long).t().contiguous()
        edge_type = torch.tensor([relation for _, _, relation, _ in edges], dtype=torch.long)
        edge_mask_owner = torch.tensor([owner for _, _, _, owner in edges], dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_type = torch.empty((0,), dtype=torch.long)
        edge_mask_owner = torch.empty((0,), dtype=torch.long)

    return {
        "dataset": dataset_name,
        "split": split,
        "window_id": window_id,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "label": int(window["label"]),
        "event_input_ids": torch.tensor(event_input_ids, dtype=torch.long),
        "event_target_ids": {
            field_name: torch.tensor([target[field_name] for target in event_target_ids], dtype=torch.long)
            for field_name in TARGET_FIELDS
        },
        "event_relation_node_ids": {
            relation_name: torch.tensor(node_ids, dtype=torch.long)
            for relation_name, node_ids in event_relation_node_ids.items()
        },
        "next_event_exists": torch.tensor(next_event_exists, dtype=torch.bool),
        "next_event_index": torch.tensor(next_event_index, dtype=torch.long),
        "next_event_targets": {
            field_name: torch.tensor(targets, dtype=torch.long)
            for field_name, targets in next_event_targets.items()
        },
        "node_type_ids": torch.tensor(node_type_ids, dtype=torch.long),
        "node_token_ids": torch.tensor(node_token_ids, dtype=torch.long),
        "event_node_indices": torch.arange(len(window_rows), dtype=torch.long),
        "edge_index": edge_index,
        "edge_type": edge_type,
        "edge_mask_owner": edge_mask_owner,
        "event_count": len(window_rows),
        "node_count": len(node_type_ids),
        "edge_count": len(edges),
        "entity_count": {
            entity_type: sum(1 for key in entity_nodes if key[0] == entity_type)
            for entity_type in ENTITY_NODE_TYPES
        },
        "edge_type_count": {
            edge_type_name: sum(1 for _, _, relation_id, _ in edges if relation_id == EDGE_TYPE_TO_ID[edge_type_name])
            for edge_type_name in EDGE_TYPES
        },
    }


def build_graph_samples(
    dataset_specs: Sequence[Dict[str, Path]],
    vocabularies: Dict[str, Dict],
    actor_field: str,
) -> List[Dict[str, object]]:
    samples: List[Dict[str, object]] = []
    for spec in dataset_specs:
        rows = read_csv_rows(spec["csv_path"])
        windows = read_window_indices(spec["window_path"])
        for window_id, window in enumerate(windows):
            if int(window["start_idx"]) >= len(rows) or int(window["end_idx"]) < 0:
                continue
            sample = build_graph_sample(
                dataset_name=str(spec["name"]),
                split=str(spec["split"]),
                window_id=window_id,
                window=window,
                rows=rows,
                vocabularies=vocabularies,
                actor_field=actor_field,
            )
            if int(sample["event_count"]) > 0:
                samples.append(sample)
    return samples


class EventFieldEncoder(nn.Module):
    def __init__(
        self,
        field_vocab_sizes: Dict[str, int],
        field_embedding_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.fields = list(EVENT_INPUT_FIELDS)
        self.field_embeddings = nn.ModuleDict(
            {
                field_name: nn.Embedding(
                    num_embeddings=field_vocab_sizes[field_name],
                    embedding_dim=field_embedding_dim,
                )
                for field_name in self.fields
            }
        )
        self.projection = nn.Sequential(
            nn.Linear(len(self.fields) * field_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, event_input_ids: torch.Tensor) -> torch.Tensor:
        field_vectors = [
            self.field_embeddings[field_name](event_input_ids[:, field_index])
            for field_index, field_name in enumerate(self.fields)
        ]
        return self.projection(torch.cat(field_vectors, dim=-1))


class HeteroRGCNLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_relations: int, dropout: float) -> None:
        super().__init__()
        self.relation_linears = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_relations)]
        )
        self.self_linear = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        num_nodes = node_features.size(0)
        aggregated = torch.zeros_like(node_features)

        for relation_id, relation_linear in enumerate(self.relation_linears):
            relation_mask = edge_type == relation_id
            if not bool(relation_mask.any()):
                continue
            relation_edges = edge_index[:, relation_mask]
            src_index = relation_edges[0]
            dst_index = relation_edges[1]
            messages = relation_linear(node_features[src_index])
            relation_agg = torch.zeros_like(node_features)
            relation_deg = torch.zeros((num_nodes, 1), dtype=node_features.dtype, device=node_features.device)
            relation_agg.index_add_(0, dst_index, messages)
            relation_deg.index_add_(
                0,
                dst_index,
                torch.ones((dst_index.numel(), 1), dtype=node_features.dtype, device=node_features.device),
            )
            # Normalize inside each relation before summing relation-specific messages.
            aggregated = aggregated + relation_agg / relation_deg.clamp_min(1.0)

        updated = self.self_linear(node_features) + aggregated
        return self.dropout(self.activation(self.norm(updated)))


class HeteroEventPredictor(nn.Module):
    def __init__(
        self,
        field_vocab_sizes: Dict[str, int],
        entity_vocab_sizes: Dict[str, int],
        field_embedding_dim: int,
        hidden_dim: int,
        rgcn_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.event_encoder = EventFieldEncoder(
            field_vocab_sizes=field_vocab_sizes,
            field_embedding_dim=field_embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.entity_embeddings = nn.ModuleDict(
            {
                entity_type: nn.Embedding(
                    num_embeddings=entity_vocab_sizes[entity_type],
                    embedding_dim=hidden_dim,
                )
                for entity_type in ENTITY_NODE_TYPES
            }
        )
        self.node_type_embedding = nn.Embedding(len(NODE_TYPES), hidden_dim)
        self.rgcn_layers = nn.ModuleList(
            [HeteroRGCNLayer(hidden_dim=hidden_dim, num_relations=len(EDGE_TYPES), dropout=dropout) for _ in range(rgcn_layers)]
        )
        self.target_heads = nn.ModuleDict(
            {
                field_name: nn.Linear(hidden_dim, field_vocab_sizes[field_name])
                for field_name in TARGET_FIELDS
            }
        )
        self.relation_query_heads = nn.ModuleDict(
            {
                "predict_verb_node": nn.Linear(hidden_dim, hidden_dim),
                "predict_resource_node": nn.Linear(hidden_dim, hidden_dim),
                "predict_namespace_node": nn.Linear(hidden_dim, hidden_dim),
            }
        )
        self.transition_heads = nn.ModuleDict(
            {
                f"next_{field_name}": nn.Linear(hidden_dim, field_vocab_sizes[field_name])
                for field_name in TARGET_FIELDS
            }
        )

    def initial_node_features(
        self,
        event_input_ids: torch.Tensor,
        node_type_ids: torch.Tensor,
        node_token_ids: torch.Tensor,
        event_node_indices: torch.Tensor,
    ) -> torch.Tensor:
        node_features = torch.zeros(
            (node_type_ids.size(0), self.node_type_embedding.embedding_dim),
            dtype=torch.float32,
            device=node_type_ids.device,
        )
        node_features[event_node_indices] = self.event_encoder(event_input_ids)
        for entity_type in ENTITY_NODE_TYPES:
            type_id = NODE_TYPE_TO_ID[entity_type]
            node_mask = node_type_ids == type_id
            if not bool(node_mask.any()):
                continue
            node_features[node_mask] = self.entity_embeddings[entity_type](node_token_ids[node_mask])
        return node_features + self.node_type_embedding(node_type_ids)

    def forward(
        self,
        event_input_ids: torch.Tensor,
        node_type_ids: torch.Tensor,
        node_token_ids: torch.Tensor,
        event_node_indices: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> Dict[str, object]:
        node_features = self.initial_node_features(
            event_input_ids=event_input_ids,
            node_type_ids=node_type_ids,
            node_token_ids=node_token_ids,
            event_node_indices=event_node_indices,
        )
        for rgcn_layer in self.rgcn_layers:
            node_features = rgcn_layer(node_features, edge_index=edge_index, edge_type=edge_type)
        event_features = node_features[event_node_indices]
        return {
            "field_logits": {
                field_name: head(event_features)
                for field_name, head in self.target_heads.items()
            },
            "transition_logits": {
                field_name: self.transition_heads[f"next_{field_name}"](event_features)
                for field_name in TARGET_FIELDS
            },
            "event_features": event_features,
            "node_features": node_features,
        }


def graph_sample_to_device(sample: Dict[str, object], device: torch.device) -> Dict[str, object]:
    return {
        "event_input_ids": sample["event_input_ids"].to(device),
        "event_target_ids": {field_name: tensor.to(device) for field_name, tensor in sample["event_target_ids"].items()},
        "event_relation_node_ids": {relation_name: tensor.to(device) for relation_name, tensor in sample["event_relation_node_ids"].items()},
        "next_event_exists": sample["next_event_exists"].to(device),
        "next_event_index": sample["next_event_index"].to(device),
        "next_event_targets": {field_name: tensor.to(device) for field_name, tensor in sample["next_event_targets"].items()},
        "node_type_ids": sample["node_type_ids"].to(device),
        "node_token_ids": sample["node_token_ids"].to(device),
        "event_node_indices": sample["event_node_indices"].to(device),
        "edge_index": sample["edge_index"].to(device),
        "edge_type": sample["edge_type"].to(device),
        "edge_mask_owner": sample["edge_mask_owner"].to(device),
    }


def random_masked_event_indices(num_events: int, mask_ratio: float, device: torch.device) -> torch.Tensor:
    masked_count = int(math.ceil(num_events * max(0.0, min(1.0, float(mask_ratio)))))
    masked_count = max(1, min(num_events, masked_count))
    return torch.randperm(num_events, device=device)[:masked_count]


def filter_edges_for_mask(
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    edge_mask_owner: torch.Tensor,
    masked_event_indices: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    keep_mask = torch.ones(edge_type.size(0), dtype=torch.bool, device=edge_type.device)
    for event_index in masked_event_indices:
        keep_mask &= edge_mask_owner != int(event_index.detach().cpu().item())
    return edge_index[:, keep_mask], edge_type[keep_mask], edge_mask_owner[keep_mask]


def apply_successor_leakage_block(
    device_sample: Dict[str, object],
    masked_event_indices: torch.Tensor,
    event_input_ids: torch.Tensor,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    edge_mask_owner: torch.Tensor,
    mask_token_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply strong successor blocking for same-actor transition prediction.

    For each masked current Event, this masks the target-visible fields of its
    same_actor_next successor, removes the current<->successor actor-local edges,
    removes the successor's own target-revealing entity edges, and disconnects
    that successor from Actor nodes for this transition-only view.
    """
    successor_indices = device_sample["next_event_index"][masked_event_indices]
    valid_successors = successor_indices[successor_indices >= 0]
    if valid_successors.numel() == 0:
        return event_input_ids, edge_index, edge_type

    apply_event_field_mask(
        event_input_ids=event_input_ids,
        masked_event_indices=valid_successors.unique(),
        mask_token_ids=mask_token_ids,
    )

    keep_mask = torch.ones(edge_type.size(0), dtype=torch.bool, device=edge_type.device)
    same_actor_next_id = EDGE_TYPE_TO_ID["same_actor_next"]
    same_actor_prev_id = EDGE_TYPE_TO_ID["same_actor_prev"]
    actor_to_event_id = EDGE_TYPE_TO_ID["actor_to_event"]
    event_to_actor_id = EDGE_TYPE_TO_ID["event_to_actor"]

    for current_index, successor_index in zip(masked_event_indices, successor_indices):
        if int(successor_index.detach().cpu().item()) < 0:
            continue
        current = int(current_index.detach().cpu().item())
        successor = int(successor_index.detach().cpu().item())
        keep_mask &= ~(
            (edge_index[0] == current)
            & (edge_index[1] == successor)
            & (edge_type == same_actor_next_id)
        )
        keep_mask &= ~(
            (edge_index[0] == successor)
            & (edge_index[1] == current)
            & (edge_type == same_actor_prev_id)
        )
        keep_mask &= edge_mask_owner != successor
        keep_mask &= ~((edge_index[1] == successor) & (edge_type == actor_to_event_id))
        keep_mask &= ~((edge_index[0] == successor) & (edge_type == event_to_actor_id))

    return event_input_ids, edge_index[:, keep_mask], edge_type[keep_mask]


def prepare_current_event_masked_view(
    device_sample: Dict[str, object],
    masked_event_indices: torch.Tensor,
    mask_token_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the common current-Event masked view used by both task views."""
    event_input_ids = device_sample["event_input_ids"].clone()
    apply_event_field_mask(
        event_input_ids=event_input_ids,
        masked_event_indices=masked_event_indices,
        mask_token_ids=mask_token_ids,
    )
    edge_index, edge_type, edge_mask_owner = filter_edges_for_mask(
        edge_index=device_sample["edge_index"],
        edge_type=device_sample["edge_type"],
        edge_mask_owner=device_sample["edge_mask_owner"],
        masked_event_indices=masked_event_indices,
    )
    return event_input_ids, edge_index, edge_type, edge_mask_owner


def prepare_field_relation_view(
    device_sample: Dict[str, object],
    masked_event_indices: torch.Tensor,
    mask_token_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare View A for current Event field and relation prediction only."""
    event_input_ids, edge_index, edge_type, _ = prepare_current_event_masked_view(
        device_sample=device_sample,
        masked_event_indices=masked_event_indices,
        mask_token_ids=mask_token_ids,
    )
    return event_input_ids, edge_index, edge_type


def prepare_transition_view(
    device_sample: Dict[str, object],
    masked_event_indices: torch.Tensor,
    mask_token_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare View B with stronger successor blocking for transition loss."""
    event_input_ids, edge_index, edge_type, edge_mask_owner = prepare_current_event_masked_view(
        device_sample=device_sample,
        masked_event_indices=masked_event_indices,
        mask_token_ids=mask_token_ids,
    )
    return apply_successor_leakage_block(
        device_sample=device_sample,
        masked_event_indices=masked_event_indices,
        event_input_ids=event_input_ids,
        edge_index=edge_index,
        edge_type=edge_type,
        edge_mask_owner=edge_mask_owner,
        mask_token_ids=mask_token_ids,
    )


def relation_prediction_loss(
    model: HeteroEventPredictor,
    model_outputs: Dict[str, object],
    device_sample: Dict[str, object],
    masked_event_indices: torch.Tensor,
    reduction: str,
) -> torch.Tensor:
    """Predict the true Verb/Resource/Namespace entity node for masked Events.

    Classification is restricted to entity nodes of the same type inside the
    current window, so the task stays local and does not require global entity
    ids across windows.
    """
    relation_specs = {
        "verb": ("Verb", "predict_verb_node"),
        "resource": ("Resource", "predict_resource_node"),
        "namespace": ("Namespace", "predict_namespace_node"),
    }
    relation_losses: List[torch.Tensor] = []
    event_features = model_outputs["event_features"][masked_event_indices]
    node_features = model_outputs["node_features"]
    node_type_ids = device_sample["node_type_ids"]

    for relation_name, (node_type, head_name) in relation_specs.items():
        candidate_indices = torch.nonzero(
            node_type_ids == NODE_TYPE_TO_ID[node_type],
            as_tuple=False,
        ).flatten()
        if candidate_indices.numel() == 0:
            if reduction == "none":
                relation_losses.append(
                    torch.zeros(masked_event_indices.size(0), dtype=torch.float32, device=masked_event_indices.device)
                )
            else:
                relation_losses.append(torch.tensor(0.0, dtype=torch.float32, device=masked_event_indices.device))
            continue

        query = model.relation_query_heads[head_name](event_features)
        candidate_features = node_features[candidate_indices]
        scores = query @ candidate_features.t()
        true_node_ids = device_sample["event_relation_node_ids"][relation_name][masked_event_indices]
        matches = true_node_ids.unsqueeze(1).eq(candidate_indices.unsqueeze(0))
        valid_mask = matches.any(dim=1)
        if not bool(valid_mask.any()):
            if reduction == "none":
                relation_losses.append(
                    torch.zeros(masked_event_indices.size(0), dtype=torch.float32, device=masked_event_indices.device)
                )
            else:
                relation_losses.append(torch.tensor(0.0, dtype=torch.float32, device=masked_event_indices.device))
            continue

        # Only matched true entity nodes are valid positives; unmatched ids may come from OOV/window filtering.
        valid_scores = scores[valid_mask]
        target_positions = matches[valid_mask].long().argmax(dim=1)
        valid_loss = nn.functional.cross_entropy(valid_scores, target_positions, reduction=reduction)
        if reduction == "none":
            relation_loss = torch.zeros(masked_event_indices.size(0), dtype=torch.float32, device=masked_event_indices.device)
            relation_loss[valid_mask] = valid_loss
            relation_losses.append(relation_loss)
        else:
            relation_losses.append(valid_loss)

    if not relation_losses:
        if reduction == "none":
            return torch.zeros(masked_event_indices.size(0), dtype=torch.float32, device=masked_event_indices.device)
        return torch.tensor(0.0, dtype=torch.float32, device=masked_event_indices.device)

    if reduction == "none":
        return torch.stack(relation_losses, dim=0).mean(dim=0)
    return torch.stack(relation_losses).mean()


def field_prediction_loss(
    model_outputs: Dict[str, object],
    device_sample: Dict[str, object],
    masked_event_indices: torch.Tensor,
    reduction: str,
) -> torch.Tensor:
    """Compute masked Event field prediction loss for TARGET_FIELDS."""
    losses: List[torch.Tensor] = []
    field_logits = model_outputs["field_logits"]
    for field_name in TARGET_FIELDS:
        losses.append(
            nn.functional.cross_entropy(
                field_logits[field_name][masked_event_indices],
                device_sample["event_target_ids"][field_name][masked_event_indices],
                reduction=reduction,
            )
        )
    if reduction == "none":
        return torch.stack(losses, dim=0).mean(dim=0)
    return torch.stack(losses).mean()


def transition_prediction_loss(
    model_outputs: Dict[str, object],
    device_sample: Dict[str, object],
    masked_event_indices: torch.Tensor,
    reduction: str,
) -> torch.Tensor:
    """Predict same-actor next Event behavior for masked Events.

    Only masked Events that have a same_actor_next successor contribute to this
    loss. Events without such a successor receive zero transition loss.
    """
    valid_mask = device_sample["next_event_exists"][masked_event_indices]
    if not bool(valid_mask.any()):
        if reduction == "none":
            return torch.zeros(masked_event_indices.size(0), dtype=torch.float32, device=masked_event_indices.device)
        return torch.tensor(0.0, dtype=torch.float32, device=masked_event_indices.device)

    valid_event_indices = masked_event_indices[valid_mask]
    transition_logits = model_outputs["transition_logits"]
    losses: List[torch.Tensor] = []
    for field_name in TARGET_FIELDS:
        losses.append(
            nn.functional.cross_entropy(
                transition_logits[field_name][valid_event_indices],
                device_sample["next_event_targets"][field_name][valid_event_indices],
                reduction=reduction,
            )
        )

    if reduction == "none":
        valid_losses = torch.stack(losses, dim=0).mean(dim=0)
        all_losses = torch.zeros(masked_event_indices.size(0), dtype=torch.float32, device=masked_event_indices.device)
        all_losses[valid_mask] = valid_losses
        return all_losses
    return torch.stack(losses).mean()


def compute_graph_loss(
    model: HeteroEventPredictor,
    sample: Dict[str, object],
    device: torch.device,
    mask_token_ids: torch.Tensor,
    mask_ratio: float,
    lambda_rel: float,
    lambda_trans: float,
) -> torch.Tensor:
    device_sample = graph_sample_to_device(sample, device)
    masked_event_indices = random_masked_event_indices(
        num_events=device_sample["event_input_ids"].size(0),
        mask_ratio=mask_ratio,
        device=device,
    )
    field_event_input_ids, field_edge_index, field_edge_type = prepare_field_relation_view(
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
    field_loss = field_prediction_loss(
        model_outputs=field_outputs,
        device_sample=device_sample,
        masked_event_indices=masked_event_indices,
        reduction="mean",
    )
    relation_loss = relation_prediction_loss(
        model=model,
        model_outputs=field_outputs,
        device_sample=device_sample,
        masked_event_indices=masked_event_indices,
        reduction="mean",
    )

    transition_event_input_ids, transition_edge_index, transition_edge_type = prepare_transition_view(
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
    transition_loss = transition_prediction_loss(
        model_outputs=transition_outputs,
        device_sample=device_sample,
        masked_event_indices=masked_event_indices,
        reduction="mean",
    )
    return field_loss + float(lambda_rel) * relation_loss + float(lambda_trans) * transition_loss


def train_model(
    model: HeteroEventPredictor,
    training_samples: Sequence[Dict[str, object]],
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    mask_token_ids: torch.Tensor,
    mask_ratio: float,
    lambda_rel: float,
    lambda_trans: float,
) -> List[Dict[str, float]]:
    history: List[Dict[str, float]] = []
    if not training_samples:
        return history

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for epoch in range(epochs):
        shuffled_samples = list(training_samples)
        random.shuffle(shuffled_samples)
        model.train()
        epoch_losses: List[float] = []

        for start in range(0, len(shuffled_samples), batch_size):
            batch_samples = shuffled_samples[start : start + batch_size]
            optimizer.zero_grad()
            batch_losses = [
                compute_graph_loss(
                    model=model,
                    sample=sample,
                    device=device,
                    mask_token_ids=mask_token_ids,
                    mask_ratio=mask_ratio,
                    lambda_rel=lambda_rel,
                    lambda_trans=lambda_trans,
                )
                for sample in batch_samples
            ]
            batch_loss = torch.stack(batch_losses).mean()
            batch_loss.backward()
            optimizer.step()
            epoch_losses.append(float(batch_loss.detach().cpu().item()))

        history.append({"epoch": epoch + 1, "loss": mean(epoch_losses)})
    return history


def score_graph_sample(
    model: HeteroEventPredictor,
    sample: Dict[str, object],
    device: torch.device,
    mask_token_ids: torch.Tensor,
    score_mask_batch_size: int,
    lambda_rel: float,
    lambda_trans: float,
) -> List[float]:
    device_sample = graph_sample_to_device(sample, device)
    model.eval()
    event_count = device_sample["event_input_ids"].size(0)
    event_scores = [0.0 for _ in range(event_count)]
    batch_size = max(1, int(score_mask_batch_size))
    with torch.no_grad():
        for start in range(0, event_count, batch_size):
            end = min(start + batch_size, event_count)
            masked_event_indices = torch.arange(start, end, dtype=torch.long, device=device)
            field_event_input_ids, field_edge_index, field_edge_type = prepare_field_relation_view(
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
            field_loss = field_prediction_loss(
                model_outputs=field_outputs,
                device_sample=device_sample,
                masked_event_indices=masked_event_indices,
                reduction="none",
            )
            relation_loss = relation_prediction_loss(
                model=model,
                model_outputs=field_outputs,
                device_sample=device_sample,
                masked_event_indices=masked_event_indices,
                reduction="none",
            )

            transition_event_input_ids, transition_edge_index, transition_edge_type = prepare_transition_view(
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
            transition_loss = transition_prediction_loss(
                model_outputs=transition_outputs,
                device_sample=device_sample,
                masked_event_indices=masked_event_indices,
                reduction="none",
            )
            batch_scores = (
                field_loss
                + float(lambda_rel) * relation_loss
                + float(lambda_trans) * transition_loss
            ).detach().cpu().tolist()
            for offset, score in enumerate(batch_scores):
                event_scores[start + offset] = float(score)
    return event_scores


def aggregate_scores(scores: Sequence[float], reduction: str, top_k: int) -> float:
    if not scores:
        return 0.0
    if reduction == "max":
        return float(max(scores))
    if reduction == "mean":
        return float(mean(scores))
    if reduction == "meanmax":
        return float((mean(scores) + max(scores)) / 2.0)
    if reduction == "topk":
        k = max(1, min(int(top_k), len(scores)))
        return float(mean(sorted(scores, reverse=True)[:k]))
    raise ValueError(f"Unsupported score reduction: {reduction}")


def quantile_threshold(scores: Sequence[float], nu: float) -> float:
    if not scores:
        return 0.0
    sorted_scores = sorted(float(score) for score in scores)
    percentile = max(0.0, min(1.0, 1.0 - float(nu)))
    if len(sorted_scores) == 1:
        return sorted_scores[0]
    position = percentile * (len(sorted_scores) - 1)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return sorted_scores[lower_index]
    lower_value = sorted_scores[lower_index]
    upper_value = sorted_scores[upper_index]
    weight = position - lower_index
    return lower_value * (1.0 - weight) + upper_value * weight


def compute_metrics(labels: Sequence[int], predictions: Sequence[int]) -> Dict[str, float]:
    tp = sum(1 for label, pred in zip(labels, predictions) if label == 1 and pred == 1)
    tn = sum(1 for label, pred in zip(labels, predictions) if label == 0 and pred == 0)
    fp = sum(1 for label, pred in zip(labels, predictions) if label == 0 and pred == 1)
    fn = sum(1 for label, pred in zip(labels, predictions) if label == 1 and pred == 0)
    total = len(labels)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    accuracy = (tp + tn) / total if total else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    return {
        "total": total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "accuracy": accuracy,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "tnr": tnr,
        "fnr": fnr,
    }


def score_all_samples(
    model: HeteroEventPredictor,
    samples: Sequence[Dict[str, object]],
    device: torch.device,
    window_score_reduction: str,
    top_k: int,
    mask_token_ids: torch.Tensor,
    score_mask_batch_size: int,
    lambda_rel: float,
    lambda_trans: float,
) -> List[Dict[str, object]]:
    scored_samples: List[Dict[str, object]] = []
    for sample in samples:
        event_scores = score_graph_sample(
            model=model,
            sample=sample,
            device=device,
            mask_token_ids=mask_token_ids,
            score_mask_batch_size=score_mask_batch_size,
            lambda_rel=lambda_rel,
            lambda_trans=lambda_trans,
        )
        score = aggregate_scores(event_scores, reduction=window_score_reduction, top_k=top_k)
        scored_sample = {
            key: value
            for key, value in sample.items()
            if key not in {
                "event_input_ids",
                "event_target_ids",
                "event_relation_node_ids",
                "next_event_exists",
                "next_event_index",
                "next_event_targets",
                "node_type_ids",
                "node_token_ids",
                "event_node_indices",
                "edge_index",
                "edge_type",
                "edge_mask_owner",
            }
        }
        scored_sample["event_score_mean"] = mean(event_scores)
        scored_sample["event_score_max"] = max(event_scores) if event_scores else 0.0
        scored_sample["event_score_std"] = pstdev(event_scores) if len(event_scores) > 1 else 0.0
        scored_sample["event_scores"] = event_scores
        scored_sample["score"] = float(score)
        scored_samples.append(scored_sample)
    return scored_samples


def attach_predictions(scored_samples: Sequence[Dict[str, object]], threshold: float) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    for sample in scored_samples:
        new_sample = dict(sample)
        new_sample["threshold"] = float(threshold)
        new_sample["prediction"] = 1 if float(sample["score"]) > float(threshold) else 0
        results.append(new_sample)
    return results


def summarize_metrics(scored_samples: Sequence[Dict[str, object]], threshold: float) -> List[Dict[str, float]]:
    grouped_samples: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for sample in scored_samples:
        grouped_samples[str(sample["dataset"])].append(sample)

    metrics_rows: List[Dict[str, float]] = []
    for dataset_name, samples in grouped_samples.items():
        labels = [int(sample["label"]) for sample in samples]
        predictions = [int(sample["prediction"]) for sample in samples]
        dataset_metrics = compute_metrics(labels, predictions)
        dataset_metrics["dataset"] = dataset_name
        dataset_metrics["threshold"] = float(threshold)
        dataset_metrics["score_mean"] = mean(float(sample["score"]) for sample in samples)
        dataset_metrics["score_std"] = pstdev(float(sample["score"]) for sample in samples) if len(samples) > 1 else 0.0
        metrics_rows.append(dataset_metrics)

    test_samples = [sample for sample in scored_samples if sample["split"] == "test"]
    test_metrics = compute_metrics(
        [int(sample["label"]) for sample in test_samples],
        [int(sample["prediction"]) for sample in test_samples],
    )
    test_metrics["dataset"] = "test_overall"
    test_metrics["threshold"] = float(threshold)
    test_metrics["score_mean"] = mean(float(sample["score"]) for sample in test_samples)
    test_metrics["score_std"] = pstdev(float(sample["score"]) for sample in test_samples) if len(test_samples) > 1 else 0.0
    metrics_rows.append(test_metrics)
    return metrics_rows


def write_window_scores(output_path: Path, samples: Sequence[Dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")


def write_metrics(output_path: Path, metrics_rows: Sequence[Dict[str, float]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "dataset",
                "threshold",
                "total",
                "tp",
                "tn",
                "fp",
                "fn",
                "precision",
                "accuracy",
                "recall",
                "f1",
                "fpr",
                "tnr",
                "fnr",
                "score_mean",
                "score_std",
            ]
        )
        for row in metrics_rows:
            writer.writerow(
                [
                    row["dataset"],
                    f"{row['threshold']:.6f}",
                    row["total"],
                    row["tp"],
                    row["tn"],
                    row["fp"],
                    row["fn"],
                    f"{row['precision']:.6f}",
                    f"{row['accuracy']:.6f}",
                    f"{row['recall']:.6f}",
                    f"{row['f1']:.6f}",
                    f"{row['fpr']:.6f}",
                    f"{row['tnr']:.6f}",
                    f"{row['fnr']:.6f}",
                    f"{row['score_mean']:.6f}",
                    f"{row['score_std']:.6f}",
                ]
            )


def write_report(
    output_path: Path,
    args: argparse.Namespace,
    vocabulary_size: int,
    threshold: float,
    training_history: Sequence[Dict[str, float]],
    metrics_rows: Sequence[Dict[str, float]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("Heterogeneous Event R-GCN anomaly detection report\n")
        handle.write("\n方法说明\n")
        handle.write("- 节点类型包含 Event、Actor、Resource、Namespace、Verb。\n")
        handle.write("- 显式构建 Actor/Event、Event/Resource、Event/Namespace、Event/Verb、Resource/Namespace 双向关系。\n")
        handle.write("- 该最小异构版本不包含 SourceIP 节点和 Event/Event temporal 边。\n")
        handle.write("- 额外加入 same_actor_next/same_actor_prev，表示同一 actor 的局部行为演化。\n")
        handle.write("- Resource/Namespace 这类实体边在窗口内去重。\n")
        handle.write("- 训练时随机 mask 部分 Event，隐藏其事件字段与目标泄漏边。\n")
        handle.write("- 模型同时学习 masked field prediction、relation prediction 和 same-actor next-event behavior prediction。\n")
        handle.write("- View A 用于 masked field prediction 和 relation prediction，仅对当前 Event 做普通 mask。\n")
        handle.write("- View B 用于 same-actor next-event transition prediction，并启用更强 successor leakage blocking。\n")
        handle.write("- View B 会额外隐藏 successor 的目标泄漏边、same_actor 直连边和 Actor/Event 连接边。\n")
        handle.write("- 打分时按小批 Event mask，使用三项损失的加权和作为事件异常分数。\n")

        handle.write("\n模型参数\n")
        handle.write(f"- vocabulary_size: {vocabulary_size}\n")
        handle.write(f"- node_types: {', '.join(NODE_TYPES)}\n")
        handle.write(f"- event_input_fields: {', '.join(EVENT_INPUT_FIELDS)}\n")
        handle.write(f"- target_fields: {', '.join(TARGET_FIELDS)}\n")
        handle.write(f"- edge_types: {', '.join(EDGE_TYPES)}\n")
        handle.write(f"- field_embedding_dim: {args.field_embedding_dim}\n")
        handle.write(f"- hidden_dim: {args.hidden_dim}\n")
        handle.write(f"- rgcn_layers: {args.rgcn_layers}\n")
        handle.write(f"- dropout: {args.dropout}\n")
        handle.write(f"- mask_ratio: {args.mask_ratio}\n")
        handle.write(f"- lambda_rel: {args.lambda_rel}\n")
        handle.write(f"- lambda_trans: {args.lambda_trans}\n")
        handle.write(f"- score_mask_batch_size: {args.score_mask_batch_size}\n")
        handle.write(f"- resource_key_schema: {RESOURCE_KEY_SCHEMA}\n")
        handle.write(f"- window_score_reduction: {args.window_score_reduction}\n")
        handle.write(f"- top_k: {args.top_k}\n")
        handle.write(f"- nu: {args.nu}\n")
        handle.write(f"- threshold: {threshold:.6f}\n")

        handle.write("\n训练历史\n")
        if training_history:
            for item in training_history:
                handle.write(f"- epoch {item['epoch']}: loss={item['loss']:.6f}\n")
        else:
            handle.write("- 没有可用训练样本，未执行训练。\n")

        handle.write("\n评估结果\n")
        for row in metrics_rows:
            handle.write(
                f"- {row['dataset']}: precision={row['precision']:.4f}, accuracy={row['accuracy']:.4f}, "
                f"recall={row['recall']:.4f}, f1={row['f1']:.4f}, fpr={row['fpr']:.4f}, "
                f"tnr={row['tnr']:.4f}, fnr={row['fnr']:.4f}\n"
            )


def total_vocabulary_size(vocabularies: Dict[str, Dict]) -> int:
    field_size = sum(len(vocabularies["fields"][field_name]["token_to_id"]) for field_name in VOCAB_FIELDS)
    entity_size = sum(len(vocabularies["entities"][entity_type]["token_to_id"]) for entity_type in ENTITY_NODE_TYPES)
    return field_size + entity_size


def run_experiment(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_specs = default_dataset_specs()
    training_rows = collect_training_rows(dataset_specs)
    if not training_rows:
        raise ValueError("No normal training rows found for graph_hete.")

    vocabularies = prepare_vocabularies(
        training_rows=training_rows,
        vocab_path=Path(args.vocab_output),
        min_frequency=args.min_frequency,
        rebuild_vocab=args.rebuild_vocab,
        actor_field=args.actor_field,
    )
    field_vocab_sizes = {
        field_name: len(vocabularies["fields"][field_name]["token_to_id"])
        for field_name in VOCAB_FIELDS
    }
    entity_vocab_sizes = {
        entity_type: len(vocabularies["entities"][entity_type]["token_to_id"])
        for entity_type in ENTITY_NODE_TYPES
    }
    mask_token_ids = event_mask_token_ids(vocabularies)

    samples = build_graph_samples(dataset_specs=dataset_specs, vocabularies=vocabularies, actor_field=args.actor_field)
    training_samples = [
        sample
        for sample in samples
        if sample["split"] == "train" and int(sample["label"]) == 0
    ]
    if not training_samples:
        raise ValueError("No normal training graph windows found for graph_hete.")

    device = torch.device(args.device)
    mask_token_ids = mask_token_ids.to(device)
    model = HeteroEventPredictor(
        field_vocab_sizes=field_vocab_sizes,
        entity_vocab_sizes=entity_vocab_sizes,
        field_embedding_dim=args.field_embedding_dim,
        hidden_dim=args.hidden_dim,
        rgcn_layers=args.rgcn_layers,
        dropout=args.dropout,
    ).to(device)

    training_history = train_model(
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

    model_path = Path(args.model_output)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "node_types": list(NODE_TYPES),
            "event_input_fields": list(EVENT_INPUT_FIELDS),
            "target_fields": list(TARGET_FIELDS),
            "edge_types": list(EDGE_TYPES),
            "objective": "actor_centered_heterogeneous_temporal_field_and_relation_prediction",
            "mask_ratio": args.mask_ratio,
            "lambda_rel": args.lambda_rel,
            "lambda_trans": args.lambda_trans,
            "score_mask_batch_size": args.score_mask_batch_size,
            "resource_key_schema": RESOURCE_KEY_SCHEMA,
            "field_embedding_dim": args.field_embedding_dim,
            "hidden_dim": args.hidden_dim,
            "rgcn_layers": args.rgcn_layers,
            "dropout": args.dropout,
            "actor_field": args.actor_field,
            "training_history": training_history,
        },
        model_path,
    )

    scored_samples = score_all_samples(
        model=model,
        samples=samples,
        device=device,
        window_score_reduction=args.window_score_reduction,
        top_k=args.top_k,
        mask_token_ids=mask_token_ids,
        score_mask_batch_size=args.score_mask_batch_size,
        lambda_rel=args.lambda_rel,
        lambda_trans=args.lambda_trans,
    )
    training_scores = [
        float(sample["score"])
        for sample in scored_samples
        if sample["split"] == "train" and int(sample["label"]) == 0
    ]
    threshold = quantile_threshold(training_scores, nu=args.nu)

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
                "resource_key_schema": RESOURCE_KEY_SCHEMA,
                "window_score_reduction": args.window_score_reduction,
                "top_k": args.top_k,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    predicted_samples = attach_predictions(scored_samples=scored_samples, threshold=threshold)
    metrics_rows = summarize_metrics(predicted_samples, threshold=threshold)

    write_window_scores(Path(args.window_scores_output), predicted_samples)
    write_metrics(Path(args.metrics_output), metrics_rows)
    write_report(
        output_path=Path(args.report_output),
        args=args,
        vocabulary_size=total_vocabulary_size(vocabularies),
        threshold=threshold,
        training_history=training_history,
        metrics_rows=metrics_rows,
    )

    for row in metrics_rows:
        print(
            f"[graph_hete_add] {row['dataset']}: precision={row['precision']:.4f}, accuracy={row['accuracy']:.4f}, "
            f"recall={row['recall']:.4f}, f1={row['f1']:.4f}, fpr={row['fpr']:.4f}"
        )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Heterogeneous R-GCN anomaly detection for Kubernetes audit windows.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--min-frequency", type=int, default=1, help="Minimum token frequency kept in vocabularies.")
    parser.add_argument("--rebuild-vocab", action="store_true", help="Force rebuilding heterogeneous graph vocabularies.")
    parser.add_argument("--field-embedding-dim", type=int, default=16, help="Embedding dimension for each Event input field.")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension for nodes and R-GCN layers.")
    parser.add_argument("--rgcn-layers", type=int, default=2, help="Number of R-GCN layers.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout ratio used in encoders and R-GCN layers.")
    parser.add_argument("--mask-ratio", type=float, default=0.3, help="Fraction of Event nodes masked in each training graph.")
    parser.add_argument("--lambda-rel", type=float, default=0.5, help="Weight for Verb/Resource/Namespace relation prediction loss.")
    parser.add_argument("--lambda-trans", type=float, default=0.5, help="Weight for same-actor next-event behavior prediction loss.")
    parser.add_argument("--score-mask-batch-size", type=int, default=32, help="Number of Event nodes masked together in each scoring forward pass.")
    parser.add_argument("--batch-size", type=int, default=16, help="Number of window graphs per optimization step.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs on normal graph windows.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate for graph model training.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for optimizer regularization.")
    parser.add_argument("--nu", type=float, default=0.1, help="Fraction of normal training windows allowed above the anomaly threshold.")
    parser.add_argument("--actor-field", default="actor_id", help="Field name used to build Actor nodes.")
    parser.add_argument("--window-score-reduction", choices=["max", "mean", "meanmax", "topk"], default="max", help="How to aggregate event prediction losses into one window score.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of highest event scores averaged when window-score-reduction is topk.")
    parser.add_argument("--device", default="cpu", help="Torch device, for example cpu or cuda.")
    parser.add_argument("--vocab-output", default=str(Path(config.arithmetic_path) / "graph_hete_add_vocab.json"), help="Output JSON path for heterogeneous graph vocabularies.")
    parser.add_argument("--model-output", default=str(Path(config.arithmetic_path) / "graph_hete_add_model.pt"), help="Output path for trained heterogeneous graph model checkpoint.")
    parser.add_argument("--threshold-output", default=str(Path(config.arithmetic_path) / "graph_hete_add_threshold.json"), help="Output JSON path for learned anomaly threshold.")
    parser.add_argument("--window-scores-output", default=str(Path(config.arithmetic_path) / "graph_hete_add_window_scores.jsonl"), help="Output JSONL path for window-level graph scores.")
    parser.add_argument("--metrics-output", default=str(Path(config.arithmetic_path) / "graph_hete_add_metrics.csv"), help="Output CSV path for evaluation metrics.")
    parser.add_argument("--report-output", default=str(Path(config.arithmetic_path) / "graph_hete_add_report.txt"), help="Output text path for experiment report.")
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
