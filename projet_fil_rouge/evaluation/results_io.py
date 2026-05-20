"""CSV persistence helpers for benchmark result rows."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def serialize_cell(value):
    """Serialize Python/numpy values into CSV-friendly scalar values."""

    if value is None:
        return ""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=isinstance(value, dict),
            default=_json_default,
        )
    return value


def serialize_row(row, fieldnames):
    return {field: serialize_cell(row.get(field, "")) for field in fieldnames}


def write_rows_csv(rows, output_path, fieldnames, append=False):
    """Write rows to a CSV and return the output path."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists() and output_path.stat().st_size > 0
    mode = "a" if append and file_exists else "w"

    with output_path.open(mode, newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()
        for row in rows:
            writer.writerow(serialize_row(row, fieldnames))
        csv_file.flush()
    return output_path


def read_completed_ids(output_path, id_column="task_id"):
    """Return ids already present in a previous CSV."""

    output_path = Path(output_path)
    if not output_path.exists() or output_path.stat().st_size == 0:
        return set()

    with output_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        return {row[id_column] for row in reader if row.get(id_column)}


def initialize_output_csv(output_path, fieldnames, resume=False, overwrite=False):
    """Validate output mode and return whether the CSV already has a header."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    has_content = output_path.exists() and output_path.stat().st_size > 0
    if has_content and not resume and not overwrite:
        raise FileExistsError(
            f"Output CSV already exists: {output_path}. "
            "Use --resume or --overwrite."
        )
    if overwrite:
        write_rows_csv([], output_path=output_path, fieldnames=fieldnames, append=False)
        return True
    return has_content
