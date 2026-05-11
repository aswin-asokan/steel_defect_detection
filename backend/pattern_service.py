#!/usr/bin/env python3
"""Pattern analysis helpers for defect detection CSV logs."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


@dataclass
class PatternConfig:
    bucket_minutes: int = 5
    min_window_events: int = 3
    dominant_ratio_threshold: float = 0.6


def _parse_time(value: str) -> datetime | None:
    if not value:
        return None
    for fmt in ("%Y%m%d_%H%M%S_%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _split_labels(value: str) -> list[str]:
    if not value:
        return []
    return [p.strip() for p in value.split(";") if p.strip()]


def analyze_detection_csv(
    csv_path: str | Path,
    out_json_path: str | Path | None = None,
    config: PatternConfig | None = None,
) -> dict[str, Any]:
    cfg = config or PatternConfig()
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Detection CSV not found: {csv_path}")

    rows: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            ts = _parse_time(row.get("timestamp", ""))
            labels = _split_labels(row.get("yolo_labels", ""))
            rows.append(
                {
                    "timestamp": ts,
                    "labels": labels,
                    "frame_index": row.get("frame_index", ""),
                    "bbox_raw": row.get("yolo_bboxes", ""),
                }
            )

    defect_rows = [r for r in rows if r["labels"]]
    flat_labels = [label for r in defect_rows for label in r["labels"]]
    label_counts = Counter(flat_labels)

    windows: dict[datetime, list[dict[str, Any]]] = defaultdict(list)
    for row in defect_rows:
        ts = row["timestamp"]
        if ts is None:
            continue
        bucket_start = ts - timedelta(
            minutes=ts.minute % cfg.bucket_minutes,
            seconds=ts.second,
            microseconds=ts.microsecond,
        )
        windows[bucket_start].append(row)

    window_alerts: list[dict[str, Any]] = []
    for start in sorted(windows.keys()):
        entries = windows[start]
        labels = [label for entry in entries for label in entry["labels"]]
        if len(labels) < cfg.min_window_events:
            continue
        counts = Counter(labels)
        dominant_type, dominant_count = counts.most_common(1)[0]
        ratio = dominant_count / max(1, len(labels))
        if ratio >= cfg.dominant_ratio_threshold:
            window_alerts.append(
                {
                    "window_start": start.isoformat(),
                    "window_end": (start + timedelta(minutes=cfg.bucket_minutes)).isoformat(),
                    "total_defects": len(labels),
                    "dominant_type": dominant_type,
                    "dominant_ratio": round(ratio, 3),
                }
            )

    recurring_patterns = [
        {"defect_type": defect_type, "count": count}
        for defect_type, count in label_counts.most_common(5)
    ]

    summary: dict[str, Any] = {
        "source_csv": str(csv_path.resolve()),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "total_rows": len(rows),
        "rows_with_defects": len(defect_rows),
        "top_defect_types": recurring_patterns,
        "window_alerts": window_alerts,
        "notes": "Pattern summary generated from YOLO labels in the detection log.",
    }

    if out_json_path is not None:
        out_json_path = Path(out_json_path)
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        with out_json_path.open("w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2)
        summary["saved_to"] = str(out_json_path.resolve())

    return summary
