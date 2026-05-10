"""Split filtered CSV audit logs into sliding time windows."""

import argparse
import csv
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path


def _parse_timestamp(raw: str) -> datetime:
    value = (raw or "").strip()
    if not value:
        raise ValueError("Empty timestamp value.")
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _parse_label(raw: str) -> int:
    value = (raw or "0").strip()
    if not value:
        return 0
    return 1 if int(value) != 0 else 0


def _default_output_path(input_path: Path, output_dir: Path) -> Path:
    name = input_path.name
    if name.endswith("_filter.csv"):
        name = name[: -len("_filter.csv")] + ".txt"
    else:
        name = input_path.stem + ".txt"
    return output_dir / name


def build_window_index_labels(
    input_path,
    output_path,
    window_minutes=3,
    step_minutes=1,
):
    input_path = Path(input_path)
    output_path = Path(output_path)

    events = []
    with input_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            if "stage_ts" not in row:
                raise KeyError(f"Missing 'stage_ts' column in {input_path}")
            ts = _parse_timestamp(row["stage_ts"])
            label = _parse_label(row.get("label", "0"))
            events.append({"idx": idx, "ts": ts, "label": label})

    if not events:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")
        return []

    events_by_ts = sorted(events, key=lambda item: (item["ts"], item["idx"]))

    window = timedelta(minutes=window_minutes)
    step = timedelta(minutes=step_minutes)
    start = events_by_ts[0]["ts"]
    end_limit = events_by_ts[-1]["ts"]
    latest_full_window_start = end_limit - window

    if start > latest_full_window_start:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")
        return []

    left = 0
    right = 0
    anomaly_count = 0
    windows = []
    min_idx_queue = deque()
    max_idx_queue = deque()

    while start <= latest_full_window_start:
        window_end = start + window

        while right < len(events_by_ts) and events_by_ts[right]["ts"] < window_end:
            if events_by_ts[right]["label"] == 1:
                anomaly_count += 1

            while (
                min_idx_queue
                and events_by_ts[min_idx_queue[-1]]["idx"] >= events_by_ts[right]["idx"]
            ):
                min_idx_queue.pop()
            min_idx_queue.append(right)

            while (
                max_idx_queue
                and events_by_ts[max_idx_queue[-1]]["idx"] <= events_by_ts[right]["idx"]
            ):
                max_idx_queue.pop()
            max_idx_queue.append(right)

            right += 1

        while left < len(events_by_ts) and events_by_ts[left]["ts"] < start:
            if events_by_ts[left]["label"] == 1:
                anomaly_count -= 1

            if min_idx_queue and min_idx_queue[0] == left:
                min_idx_queue.popleft()
            if max_idx_queue and max_idx_queue[0] == left:
                max_idx_queue.popleft()

            left += 1

        if left < right and min_idx_queue and max_idx_queue:
            start_idx = events_by_ts[min_idx_queue[0]]["idx"]
            end_idx = events_by_ts[max_idx_queue[0]]["idx"]
            window_label = 1 if anomaly_count > 0 else 0
            windows.append((start_idx, end_idx, window_label))

        start += step

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        for start_idx, end_idx, window_label in windows:
            handle.write(f"{start_idx}\t{end_idx}\t{window_label}\n")

    return windows


def batch_build_window_index_labels(
    input_dir="csv",
    output_dir="traintest",
    pattern="*_filter.csv",
    window_minutes=3,
    step_minutes=1,
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    csv_files = sorted(input_dir.glob(pattern))
    if not csv_files:
        raise FileNotFoundError(f"No files matched {pattern} in {input_dir}")

    summary = []
    for csv_path in csv_files:
        output_path = _default_output_path(csv_path, output_dir)
        windows = build_window_index_labels(
            input_path=csv_path,
            output_path=output_path,
            window_minutes=window_minutes,
            step_minutes=step_minutes,
        )
        summary.append((csv_path, output_path, len(windows)))

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Build sliding-window train/test index files from filtered audit CSV."
    )
    parser.add_argument(
        "--input",
        help="Single input CSV file. If omitted, process all *_filter.csv under --input-dir.",
    )
    parser.add_argument(
        "--output",
        help="Single output txt file. If omitted, derive name from input file.",
    )
    parser.add_argument(
        "--input-dir",
        default="csv",
        help="Directory containing filtered CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        default="traintest",
        help="Directory to store generated txt files.",
    )
    parser.add_argument("--window-min", type=int, default=3, help="Window size in minutes.")
    parser.add_argument("--step-min", type=int, default=1, help="Sliding step in minutes.")
    args = parser.parse_args()

    if args.input:
        input_path = Path(args.input)
        output_dir = Path(args.output_dir)
        output_path = Path(args.output) if args.output else _default_output_path(input_path, output_dir)
        windows = build_window_index_labels(
            input_path=input_path,
            output_path=output_path,
            window_minutes=args.window_min,
            step_minutes=args.step_min,
        )
        print(f"{input_path} -> {output_path}: {len(windows)} windows")
        return

    summary = batch_build_window_index_labels(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        window_minutes=args.window_min,
        step_minutes=args.step_min,
    )
    for input_path, output_path, window_count in summary:
        print(f"{input_path} -> {output_path}: {window_count} windows")


if __name__ == "__main__":
    main()
