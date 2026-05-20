"""CLI to analyze the aggressive benchmark CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .config import PROJECT_DIR
    from .evaluation.benchmark_analysis import (
        build_analysis,
        compact_row,
        load_benchmark_rows,
        write_analysis_outputs,
    )
except ImportError:  # Allows running with: python3 analyse_csv_benchmark.py
    from config import PROJECT_DIR
    from evaluation.benchmark_analysis import (
        build_analysis,
        compact_row,
        load_benchmark_rows,
        write_analysis_outputs,
    )


DEFAULT_INPUT_CSV = PROJECT_DIR / "results" / "part5_benchmark_aggressive.csv"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "results" / "benchmark_analysis"


def _format_value(value):
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def print_table(rows, title, limit=10):
    print(f"\n{title}")
    rows = list(rows[:limit])
    if not rows:
        print("(empty)")
        return

    headers = list(rows[0])
    widths = [
        max(len(header), *(len(_format_value(row.get(header, ""))) for row in rows))
        for header in headers
    ]

    def format_row(values):
        return "| " + " | ".join(
            _format_value(value).ljust(width)
            for value, width in zip(values, widths)
        ) + " |"

    print(format_row(headers))
    print(format_row(["-" * width for width in widths]))
    for row in rows:
        print(format_row([row.get(header, "") for header in headers]))


def print_summary(summary):
    print("Benchmark CSV analysis")
    print(f"Rows: {summary['rows']}")
    print(f"Tasks: {summary['task_ids']}")
    print(f"OK rows: {summary['ok_rows']}")
    print(f"Error rows: {summary['error_rows']}")
    print(f"Best score: {summary['best_score']}")
    print(f"Mean score: {summary['mean_score']}")
    print(f"Total duration seconds: {summary['total_duration_seconds']}")
    print(f"Datasets: {', '.join(summary['datasets'])}")
    print(f"Preprocessors: {', '.join(summary['preprocessors'])}")
    print(f"Classifiers: {', '.join(summary['classifiers'])}")
    print(f"Fit methods: {', '.join(summary['fit_methods'])}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze the CSV produced by part5_benchmark_aggressive.py."
    )
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT_CSV))
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        default="",
        help=(
            "Optional directory where analysis CSVs and summary.json are written. "
            f"Default when --write is used: {DEFAULT_OUTPUT_DIR}"
        ),
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write analysis tables to --output-dir.",
    )
    parser.add_argument(
        "--errors",
        action="store_true",
        help="Print grouped error rows.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rows = load_benchmark_rows(args.input_csv)
    analysis = build_analysis(rows, top_n=args.top_n)

    print_summary(analysis["summary"])
    print_table(
        [compact_row(row) for row in analysis["top_overall"]],
        title=f"Top {args.top_n} overall",
        limit=args.top_n,
    )
    print_table(
        [compact_row(row) for row in analysis["best_by_dataset"]],
        title="Best by dataset",
        limit=args.top_n,
    )
    print_table(
        analysis["aggregate_by_preprocessor"],
        title="Aggregate by preprocessor",
        limit=args.top_n,
    )
    print_table(
        analysis["aggregate_by_classifier"],
        title="Aggregate by classifier",
        limit=args.top_n,
    )

    if args.errors:
        print_table(
            analysis["errors"],
            title="Grouped errors",
            limit=args.top_n,
        )

    if args.write:
        output_dir = Path(args.output_dir or DEFAULT_OUTPUT_DIR)
        written_paths = write_analysis_outputs(analysis, output_dir)
        print("\nWritten analysis files:")
        for name, path in written_paths.items():
            print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
