"""CLI for the aggressive Part V benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    from .classifiers import create_all_classifier_specs
    from .config import DEFAULT_SAMPLE_RATE, PROJECT_DIR, RANDOM_SEED, seed_everything
    from .data import list_datasets
    from .evaluation import (
        BENCHMARK_CSV_FIELDNAMES,
        create_fit_method_specs,
        initialize_output_csv,
        iter_benchmark_jobs,
        job_id,
        load_benchmark_datasets,
        read_completed_ids,
        run_benchmark_jobs,
        select_specs,
        write_rows_csv,
    )
    from .part5_compare_preprocess import make_preprocessor_specs
except ImportError:  # Allows running with: python3 part5_benchmark_aggressive.py
    from classifiers import create_all_classifier_specs
    from config import DEFAULT_SAMPLE_RATE, PROJECT_DIR, RANDOM_SEED, seed_everything
    from data import list_datasets
    from evaluation import (
        BENCHMARK_CSV_FIELDNAMES,
        create_fit_method_specs,
        initialize_output_csv,
        iter_benchmark_jobs,
        job_id,
        load_benchmark_datasets,
        read_completed_ids,
        run_benchmark_jobs,
        select_specs,
        write_rows_csv,
    )
    from part5_compare_preprocess import make_preprocessor_specs


DEFAULT_OUTPUT_CSV = PROJECT_DIR / "results" / "part5_benchmark_aggressive.csv"


def _known_keys(specs, attr):
    return ", ".join(str(getattr(spec, attr)) for spec in specs)


def build_specs(args):
    """Build and select the existing library specs requested by the CLI."""

    preprocessor_specs = select_specs(
        make_preprocessor_specs(fast=args.fast),
        requested_keys=args.preprocessors,
        key_attr="name",
        label="preprocessors",
    )
    classifier_specs = select_specs(
        create_all_classifier_specs(fast=args.fast),
        requested_keys=args.classifiers,
        key_attr="key",
        label="classifiers",
    )
    fit_method_specs = select_specs(
        create_fit_method_specs(),
        requested_keys=args.fit_methods,
        key_attr="key",
        label="fit methods",
    )
    return preprocessor_specs, classifier_specs, fit_method_specs


def build_jobs(args, completed_task_ids=None):
    """Load datasets and return selected benchmark tuple jobs."""

    datasets = load_benchmark_datasets(
        dataset_names=args.datasets,
        sample_rate=args.sample_rate,
        target_length=args.target_length,
        strict=args.strict_datasets,
    )
    preprocessor_specs, classifier_specs, fit_method_specs = build_specs(args)
    return list(
        iter_benchmark_jobs(
            datasets=datasets,
            preprocessor_specs=preprocessor_specs,
            classifier_specs=classifier_specs,
            fit_method_specs=fit_method_specs,
            completed_task_ids=completed_task_ids,
            limit=args.limit,
        )
    )


def print_dry_run(jobs):
    print(f"Benchmark tasks: {len(jobs)}")
    for job in jobs[:10]:
        print(f"- {job_id(*job)}")
    if len(jobs) > 10:
        print(f"... {len(jobs) - 10} more")


def print_summary(rows):
    if not rows:
        print("No benchmark row produced")
        return

    ok_rows = [row for row in rows if row.get("status") == "ok"]
    error_rows = [row for row in rows if row.get("status") == "error"]
    print(
        "\nSummary: "
        f"{len(rows)} rows, {len(ok_rows)} ok, {len(error_rows)} errors"
    )

    scored_rows = [
        row
        for row in ok_rows
        if row.get("score") is not None and not np.isnan(float(row["score"]))
    ]
    top_rows = sorted(scored_rows, key=lambda row: float(row["score"]), reverse=True)[:10]
    if not top_rows:
        return

    print("Top results:")
    for rank, row in enumerate(top_rows, start=1):
        print(
            f"{rank}. {row['dataset']} | {row['preprocessor']} | "
            f"{row['classifier']} | {row['fit_method']} | "
            f"score={float(row['score']):.3f}"
        )


def parse_args():
    all_preprocessors = make_preprocessor_specs(fast=False)
    all_classifiers = create_all_classifier_specs(fast=False)
    all_fit_methods = create_fit_method_specs()

    parser = argparse.ArgumentParser(
        description="Run Part V over dataset x preprocessor x classifier x fit method."
    )
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--cv", type=int, default=6)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--target-length", type=int, default=None)
    parser.add_argument("--scoring", default="accuracy")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help=(
            "Datasets to run. Defaults to available datasets. "
            f"Known: {', '.join(list_datasets(include_missing=True))}"
        ),
    )
    parser.add_argument(
        "--preprocessors",
        nargs="+",
        default=None,
        help=(
            "Preprocessor keys to run, or 'all'. Defaults to all. "
            f"Known: {_known_keys(all_preprocessors, 'name')}"
        ),
    )
    parser.add_argument(
        "--classifiers",
        nargs="+",
        default=None,
        help=(
            "Classifier spec keys to run, or 'all'. Defaults to all. "
            f"Known: {_known_keys(all_classifiers, 'key')}"
        ),
    )
    parser.add_argument(
        "--fit-methods",
        nargs="+",
        default=None,
        help=(
            "Fit method keys to run, or 'all'. Defaults to all. "
            f"Known: {_known_keys(all_fit_methods, 'key')}"
        ),
    )
    parser.add_argument(
        "--strict-datasets",
        action="store_true",
        help="Raise on missing datasets/files instead of skipping with a warning.",
    )
    parser.add_argument(
        "--output-csv",
        default=str(DEFAULT_OUTPUT_CSV),
        help=f"CSV output path. Default: {DEFAULT_OUTPUT_CSV}",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite cannot be used together")
    return args


def main():
    args = parse_args()
    seed_everything(RANDOM_SEED)

    output_path = Path(args.output_csv)
    completed_task_ids = read_completed_ids(output_path) if args.resume else set()
    jobs = build_jobs(args, completed_task_ids=completed_task_ids)

    if args.dry_run:
        print_dry_run(jobs)
        return

    print(f"Benchmark tasks: {len(jobs)}")
    csv_has_header = initialize_output_csv(
        output_path,
        fieldnames=BENCHMARK_CSV_FIELDNAMES,
        resume=args.resume,
        overwrite=args.overwrite,
    )

    rows = []
    for job_rows in run_benchmark_jobs(
        jobs,
        cv_splits=args.cv,
        n_jobs=args.n_jobs,
        test_size=args.test_size,
        random_state=RANDOM_SEED,
        scoring=args.scoring,
    ):
        write_rows_csv(
            job_rows,
            output_path=output_path,
            fieldnames=BENCHMARK_CSV_FIELDNAMES,
            append=csv_has_header,
        )
        csv_has_header = True
        rows.extend(job_rows)

    print_summary(rows)
    print(f"\nCSV written to: {output_path}")


if __name__ == "__main__":
    main()
