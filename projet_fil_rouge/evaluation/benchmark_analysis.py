"""Analysis helpers for aggressive benchmark CSV files."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path


DISPLAY_FIELDS = [
    "dataset",
    "fit_method",
    "preprocessor",
    "classifier",
    "score",
    "score_std",
    "n_features",
    "duration_seconds",
    "status",
]


def _as_float(value):
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def _mean(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _rounded(value, digits=6):
    return "" if value is None else round(value, digits)


def load_benchmark_rows(input_csv):
    """Read benchmark rows from a CSV file."""

    input_csv = Path(input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Benchmark CSV not found: {input_csv}")

    with input_csv.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def scored_rows(rows):
    """Return ok rows with a parseable score."""

    return [
        row
        for row in rows
        if row.get("status") == "ok" and _as_float(row.get("score")) is not None
    ]


def sort_by_score(rows, reverse=True):
    return sorted(
        rows,
        key=lambda row: _as_float(row.get("score")) or float("-inf"),
        reverse=reverse,
    )


def top_rows(rows, limit=10):
    """Return the best scored rows."""

    return sort_by_score(scored_rows(rows))[:limit]


def _group_key(row, keys):
    return tuple(row.get(key, "") for key in keys)


def best_by(rows, group_keys):
    """Return the best scored row for each group."""

    best = {}
    for row in scored_rows(rows):
        key = _group_key(row, group_keys)
        current = best.get(key)
        if current is None or _as_float(row["score"]) > _as_float(current["score"]):
            best[key] = row
    return sort_by_score(best.values())


def aggregate_by(rows, group_keys):
    """Aggregate scores and timings by one or more dimensions."""

    groups = defaultdict(list)
    for row in rows:
        groups[_group_key(row, group_keys)].append(row)

    aggregates = []
    for key, group_rows in groups.items():
        ok_rows = scored_rows(group_rows)
        error_rows = [row for row in group_rows if row.get("status") == "error"]
        scores = [_as_float(row.get("score")) for row in ok_rows]
        durations = [_as_float(row.get("duration_seconds")) for row in group_rows]
        best = sort_by_score(ok_rows)[:1]

        aggregate = {
            group_key: group_value
            for group_key, group_value in zip(group_keys, key)
        }
        aggregate.update(
            {
                "rows": len(group_rows),
                "ok_rows": len(ok_rows),
                "error_rows": len(error_rows),
                "mean_score": _rounded(_mean(scores)),
                "best_score": _rounded(max(scores) if scores else None),
                "mean_duration_seconds": _rounded(_mean(durations), digits=3),
            }
        )
        if best:
            best_row = best[0]
            aggregate.update(
                {
                    "best_dataset": best_row.get("dataset", ""),
                    "best_fit_method": best_row.get("fit_method", ""),
                    "best_preprocessor": best_row.get("preprocessor", ""),
                    "best_classifier": best_row.get("classifier", ""),
                    "best_task_id": best_row.get("task_id", ""),
                }
            )
        aggregates.append(aggregate)

    return sorted(
        aggregates,
        key=lambda row: (
            row["best_score"] == "",
            -(row["best_score"] or 0),
            -(row["mean_score"] or 0),
            row["error_rows"],
        ),
    )


def error_summary(rows):
    """Group error rows by dataset/method combination and error message."""

    groups = defaultdict(int)
    for row in rows:
        if row.get("status") != "error":
            continue
        key = (
            row.get("dataset", ""),
            row.get("fit_method", ""),
            row.get("preprocessor", ""),
            row.get("classifier", ""),
            row.get("error", ""),
        )
        groups[key] += 1

    error_rows = []
    for key, count in groups.items():
        dataset, fit_method, preprocessor, classifier, error = key
        error_rows.append(
            {
                "count": count,
                "dataset": dataset,
                "fit_method": fit_method,
                "preprocessor": preprocessor,
                "classifier": classifier,
                "error": error,
            }
        )
    return sorted(error_rows, key=lambda row: (-row["count"], row["error"]))


def summarize(rows):
    """Return global benchmark statistics."""

    ok_rows = scored_rows(rows)
    errors = [row for row in rows if row.get("status") == "error"]
    scores = [_as_float(row.get("score")) for row in ok_rows]
    durations = [_as_float(row.get("duration_seconds")) for row in rows]
    return {
        "rows": len(rows),
        "ok_rows": len(ok_rows),
        "error_rows": len(errors),
        "task_ids": len({row.get("task_id", "") for row in rows if row.get("task_id")}),
        "datasets": sorted({row.get("dataset", "") for row in rows if row.get("dataset")}),
        "preprocessors": sorted(
            {row.get("preprocessor", "") for row in rows if row.get("preprocessor")}
        ),
        "classifiers": sorted(
            {row.get("classifier", "") for row in rows if row.get("classifier")}
        ),
        "fit_methods": sorted(
            {row.get("fit_method", "") for row in rows if row.get("fit_method")}
        ),
        "mean_score": _rounded(_mean(scores)),
        "best_score": _rounded(max(scores) if scores else None),
        "mean_duration_seconds": _rounded(_mean(durations), digits=3),
        "total_duration_seconds": _rounded(
            sum(value for value in durations if value is not None),
            digits=3,
        ),
    }


def build_analysis(rows, top_n=10):
    """Build all reusable analysis tables for a benchmark CSV."""

    return {
        "summary": summarize(rows),
        "top_overall": top_rows(rows, limit=top_n),
        "best_by_dataset": best_by(rows, ["dataset"]),
        "aggregate_by_dataset": aggregate_by(rows, ["dataset"]),
        "best_by_dataset_fit_method": best_by(rows, ["dataset", "fit_method"]),
        "best_by_dataset_preprocessor": best_by(rows, ["dataset", "preprocessor"]),
        "best_by_dataset_classifier": best_by(rows, ["dataset", "classifier"]),
        "aggregate_by_preprocessor": aggregate_by(rows, ["preprocessor"]),
        "aggregate_by_classifier": aggregate_by(rows, ["classifier"]),
        "aggregate_by_fit_method": aggregate_by(rows, ["fit_method"]),
        "errors": error_summary(rows),
    }


def _fieldnames(rows):
    keys = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    return keys


def write_table_csv(rows, output_csv, fieldnames=None):
    """Write one analysis table to CSV."""

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fieldnames or _fieldnames(rows))
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return output_csv


def write_analysis_outputs(analysis, output_dir):
    """Write analysis tables and summary JSON to a directory."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(analysis["summary"], summary_file, ensure_ascii=False, indent=2)

    written_paths = {"summary": summary_path}
    for name, table in analysis.items():
        if name == "summary":
            continue
        written_paths[name] = write_table_csv(table, output_dir / f"{name}.csv")
    return written_paths


def compact_row(row):
    """Return the high-signal columns used in console output."""

    return {field: row.get(field, "") for field in DISPLAY_FIELDS}
