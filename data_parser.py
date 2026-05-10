import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
sys.path.append("../")

# Define HEADERS.
HEADERS = [
    "Event_ID",
    "window_ts",
    "stage_ts",
    "stage",
    "request_uri",
    "is_resource_request",
    "non_resource_path",
    "actor_id",
    "actor_uid",
    "actor_groups",
    "actor_type",
    "verb",
    "resource_type",
    "resource_name",
    "resource_uid",
    "api_group",
    "resource",
    "subresource",
    "namespace",
    "resource_instance",
    "status_code",
    "decision",
    "reason",
    "source_ip",
    "source_ips_all",
    "user_agent_raw",
    "label",
]

# Define the parse timestamp function.
# Define the parse timestamp function.
# Define the parse timestamp function.
def _parse_timestamp(raw):
    if not raw:
        return None
    value = raw.strip()
    if not value:
        return None
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed

# Parse the required value.
'''
  将角色分为三类：
    human
    serviceaccount
    node
    system-controller
    other-system
'''
def _actor_type(username):
    if not username:
        return ""
    if username.startswith("system:serviceaccount:"):
        return "serviceaccount"
    if username.startswith("system:node:"):
        return "node"
    if username in {
        "system:kube-scheduler",
        "system:kube-controller-manager",
        "system:apiserver",
    }:
        return "system-controller"
    if username.startswith("system:"):
        return "other-system"
    return "human-or-custom-user"


def _json_dump(value):
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=True)


def _normalize_path(request_uri):
    if not request_uri:
        return ""
    return request_uri.split("?", 1)[0]


def _is_discovery_path(path):
    if not path:
        return False
    if path in {"/api", "/apis"}:
        return True
    if path.startswith("/api/"):
        parts = [part for part in path.split("/") if part]
        if len(parts) == 2 and parts[0] == "api":
            return True
    if path.startswith("/apis/"):
        parts = [part for part in path.split("/") if part]
        if len(parts) == 3 and parts[0] == "apis":
            return True
    for prefix in ("/healthz", "/readyz", "/livez", "/version", "/metrics"):
        if path.startswith(prefix):
            return True
    return False


def _build_row(record):
    user = record.get("user") or {}
    obj_ref = record.get("objectRef") or {}
    resp_status = record.get("responseStatus") or {}
    annotations = record.get("annotations") or {}

    raw_api_group = obj_ref.get("apiGroup")
    api_group = raw_api_group or "core"
    resource = obj_ref.get("resource") or ""
    subresource = obj_ref.get("subresource") or ""
    namespace = obj_ref.get("namespace") or "__cluster__"
    resource_name = obj_ref.get("name") or ""

    request_uri = record.get("requestURI") or ""
    path = _normalize_path(request_uri)
    is_resource_request = 1 if resource else 0
    # is_resource_request = 1 if resource or uri_looks_like_resource_path(path) else 0
    non_resource_path = path if not is_resource_request else ""

    if is_resource_request:
        api_group = raw_api_group or "core"
        resource_type_parts = [api_group, resource]
        if subresource:
            resource_type_parts.append(subresource)
        resource_type = "/".join([part for part in resource_type_parts if part])
        # resource_instance = "/".join(
        #     [api_group, resource, namespace, resource_name]
        # )
        # request_kind = "resource_object" if resource_name else "resource_collection"
        if resource_name:
            request_kind = "resource_object"
            resource_instance = "/".join([api_group, resource, namespace, resource_name])
        else:
            request_kind = "resource_collection"
            resource_instance = "/".join([api_group, resource, namespace, "__LIST__"])
    else:
        if raw_api_group is None:
            api_group = ""
        if _is_discovery_path(path):
            resource_type = "__NON_RESOURCE__"
            resource_instance = (
                f"__NON_RESOURCE__:{path}" if path else "__NON_RESOURCE__"
            )
            namespace = "__NON_RESOURCE__"
            request_kind = "non_resource"
        else:
            resource_type = "__UNRESOLVED__"
            resource_instance = (
                f"__UNRESOLVED__:{path}" if path else "__UNRESOLVED__"
            )
            namespace = "__UNKNOWN__"
            request_kind = "unknown"

    if request_kind == "resource_object":
        resource_name = obj_ref.get("name") or ""
    elif request_kind == "resource_collection":
        resource_name = "__LIST__"
    else:
        resource_name = ""

    window_ts = record.get("requestReceivedTimestamp") or ""
    stage_ts = record.get("stageTimestamp") or ""
    # start_ts = _parse_timestamp(window_ts)
    # end_ts = _parse_timestamp(stage_ts)
    # latency_ms = ""
    # if start_ts and end_ts:
    #     latency_ms = int((end_ts - start_ts).total_seconds() * 1000)

    source_ips = record.get("sourceIPs") or []
    source_ip = source_ips[-1] if source_ips else ""

    actor_id = user.get("username") or ""

    return {
        "Event_ID": record.get("auditID") or "",
        "window_ts": window_ts,
        "stage_ts": stage_ts,
        "stage": record.get("stage") or "",
        "request_uri": request_uri,
        "is_resource_request": is_resource_request,
        "non_resource_path": non_resource_path,
        "actor_id": actor_id,
        "actor_uid": user.get("uid") or "",
        "actor_groups": _json_dump(user.get("groups")),
        "actor_type": _actor_type(actor_id),
        "verb": record.get("verb") or "",
        "resource_type": resource_type,
        "resource_name": resource_name,
        "resource_uid": obj_ref.get("uid") or "",
        "api_group": api_group,
        "resource": resource,
        "subresource": subresource,
        "namespace": namespace,
        "resource_instance": resource_instance,
        "status_code": resp_status.get("code") or "",
        "decision": annotations.get("authorization.k8s.io/decision") or "",
        "reason": annotations.get("authorization.k8s.io/reason") or "",
        "source_ip": source_ip,
        "source_ips_all": _json_dump(source_ips),
        "user_agent_raw": record.get("userAgent") or "",
    }


# Define the normalize audit log function.
def normalize_audit_log(input_path, output_path, label_input_path=None):
    input_path = Path(input_path)
    output_path = Path(output_path)
    label_in = Path(label_input_path) if label_input_path else None
    print(label_in)
    label_handle = label_in.open(encoding="utf-8") if label_in else None
    print(label_handle)

    with input_path.open(encoding="utf-8") as in_handle, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as out_handle:
        writer = csv.DictWriter(out_handle, fieldnames=HEADERS)
        writer.writeheader()
        for line in in_handle:
            if not line.strip():
                continue
            label_value = "0"
            if label_handle:
                label_line = label_handle.readline()
                if label_line == "":
                    raise ValueError(
                        "Label file has fewer lines than log lines."
                    )
                stripped = label_line.strip()
                if stripped != "":
                    label_value = stripped
            record = json.loads(line)
            row = _build_row(record)
            row["label"] = label_value
            writer.writerow(row)

    if label_handle:
        label_handle.close()

# Define the filter service accounts function.
def filter_service_accounts(
    input_path="normal_log_new.csv",
    output_path="normal_log_new_fliter.csv",
    label_input_path=None,
    label_output_path=None,
):
    input_path = Path(input_path)
    output_path = Path(output_path)

    label_in = Path(label_input_path) if label_input_path else None
    label_out = Path(label_output_path) if label_output_path else None

    label_handle = label_in.open(encoding="utf-8") if label_in else None
    label_out_handle = (
        label_out.open("w", encoding="utf-8") if label_out else None
    )

    with input_path.open(encoding="utf-8") as in_handle, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as out_handle:
        reader = csv.DictReader(in_handle)
        writer = csv.DictWriter(out_handle, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            label_line = None
            if label_handle:
                label_line = label_handle.readline()
                if label_line == "":
                    raise ValueError(
                        "Label file has fewer lines than CSV rows."
                    )
            actor_type = (row.get("actor_type") or "").strip()
            if actor_type in {"serviceaccount", "service_account"}:
                writer.writerow(row)
                if label_out_handle and label_line is not None:
                    label_out_handle.write(label_line)

    if label_handle:
        label_handle.close()
    if label_out_handle:
        label_out_handle.close()


def main():
    parser = argparse.ArgumentParser(
        description="Normalize k8s audit logs (JSON lines) into CSV."
    )
    parser.add_argument(
        "--input",
        default="dataset/normal/normal_log_new.log",
        help="Path to audit log JSON lines file.",
    )
    parser.add_argument(
        "--output",
        default="normal_log_new.csv",
        help="Path to output CSV file.",
    )
    parser.add_argument(
        "--filter_output",
        default="csv/normal_log_new_filter.csv",
        help="Path to filter output file.",
    )
    parser.add_argument(
        "--label_input",
        default=None,
        help="Path to label input file.",
    )
    parser.add_argument(
        "--label_output",
        default=None,
        help="Path to label output file.",
    )
    args = parser.parse_args()

    normalize_audit_log(args.input, args.output, args.label_input)
    filter_service_accounts(args.output, args.filter_output)


def main_from_config():
    import config

    csv_root = Path(config.csv_path)
    csv_root.mkdir(parents=True, exist_ok=True)

    '''
    datasets = [
        ("normal", config.log_normal, None, config.normal_log_new, config.normal_log_new_filter),
        ("abuse_dev", config.audit_abuse_dev, config.abuse_dev_label, config.audit_abuse_dev_csv, config.audit_abuse_dev_filter),
        ("abuse_ops", config.audit_abuse_ops, config.abuse_ops_label, config.audit_abuse_ops_csv, config.audit_abuse_ops_filter),
        ("abuse_sys", config.audit_abuse_sys, config.abuse_sys_label, config.audit_abuse_sys_csv, config.audit_abuse_sys_filter),
        ("kubeconfig_node2", config.audit_kubeconfig_node2, config.kubeconfig_node2_label, config.kubeconfig_node2_csv, config.kubeconfig_node2_filter),
        ("kubeconfig_node3", config.audit_kubeconfig_node3, config.kubeconfig_node3_label, config.kubeconfig_node3_csv, config.kubeconfig_node3_filter),
        ("dev_escape", config.dev_escape, config.dev_escape_label, config.dev_escape_csv, config.dev_escape_filter),
        ("ops_escape", config.ops_escape, config.ops_escape_label, config.ops_escape_csv, config.ops_escape_filter),
    ]
    '''
    datasets = [
        ("kubeconfig_node2", config.audit_kubeconfig_node2, config.kubeconfig_node2_label, config.kubeconfig_node2_csv, config.kubeconfig_node2_filter),
        ("kubeconfig_node3", config.audit_kubeconfig_node3, config.kubeconfig_node3_label, config.kubeconfig_node3_csv, config.kubeconfig_node3_filter),
        ("dev_escape", config.dev_escape, config.dev_escape_label, config.dev_escape_csv, config.dev_escape_filter),
        ("ops_escape", config.ops_escape, config.ops_escape_label, config.ops_escape_csv, config.ops_escape_filter),
    ]

    for _, log_path, label_path, csv_name, filter_name in datasets:
        output_csv = csv_root / csv_name
        output_filter = csv_root / filter_name
        normalize_audit_log(log_path, output_csv, label_path)
        filter_service_accounts(output_csv, output_filter)

# Define the cut some function.
def cut_some():
    with open("normal_log_new_fliter.csv", "r") as f:
        reader = csv.reader(f)
        i = 0
        for row in reader:
            if i <= 1000:
                with open("normal_log_new_1000.csv", "a") as f2:
                    writer = csv.writer(f2)
                    writer.writerow(row)
                i += 1
            else:
                break


if __name__ == "__main__":
    main_from_config()
