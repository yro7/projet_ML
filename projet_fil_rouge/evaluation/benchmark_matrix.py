"""Reusable benchmark matrix orchestration.

The matrix runner combines already-defined dataset bundles, preprocessor specs,
classifier specs and fit method specs. It does not define new experiment models:
jobs are plain tuples of existing library objects.
"""

from __future__ import annotations

import re

import numpy as np

try:
    from .fit_methods import run_fit_method
except ImportError:  # Allows importing from a notebook whose cwd is projet_fil_rouge/
    from evaluation.fit_methods import run_fit_method

try:
    from ..config import DATASET_CURRENT, DEFAULT_SAMPLE_RATE, RANDOM_SEED
    from ..data import list_datasets, load_named_dataset
except ImportError:  # Allows importing from a notebook whose cwd is projet_fil_rouge/
    from config import DATASET_CURRENT, DEFAULT_SAMPLE_RATE, RANDOM_SEED
    from data import list_datasets, load_named_dataset


BENCHMARK_CSV_FIELDNAMES = [
    "experiment_id",
    "task_id",
    "row_index",
    "dataset",
    "dataset_n_samples",
    "dataset_n_classes",
    "dataset_fs",
    "dataset_target_names",
    "dataset_class_counts",
    "fit_method",
    "fit_method_label",
    "preprocessor",
    "preprocessor_label",
    "classifier",
    "classifier_label",
    "classifier_family",
    "meta_classifier",
    "base_classifier",
    "score",
    "score_std",
    "n_features",
    "duration_seconds",
    "status",
    "error",
    "preprocessor_params",
    "classifier_params",
    "fit_method_comment",
    "preprocessor_comment",
    "classifier_comment",
]


def _slug(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")


def _spec_key(spec, attr):
    return getattr(spec, attr)


def select_specs(specs, requested_keys=None, key_attr="key", label="specs"):
    """Return specs selected by key while preserving requested order."""

    specs = list(specs)
    if not requested_keys or "all" in requested_keys:
        return specs

    by_key = {_spec_key(spec, key_attr): spec for spec in specs}
    unknown = [key for key in requested_keys if key not in by_key]
    if unknown:
        valid = ", ".join(sorted(by_key))
        raise ValueError(
            f"Unknown {label}: {', '.join(unknown)}. Expected one of: {valid}"
        )

    selected = []
    seen = set()
    for key in requested_keys:
        if key not in seen:
            selected.append(by_key[key])
            seen.add(key)
    return selected


def load_benchmark_datasets(
    dataset_names=None,
    sample_rate=DEFAULT_SAMPLE_RATE,
    target_length=None,
    strict=False,
):
    """Load benchmark datasets, skipping missing ones unless strict=True."""

    dataset_names = list(dataset_names or list_datasets(include_missing=False))
    if not dataset_names:
        dataset_names = [DATASET_CURRENT]

    bundles = []
    for dataset_name in dataset_names:
        try:
            bundle = load_named_dataset(
                name=dataset_name,
                sample_rate=sample_rate,
                target_length=target_length,
                strict=strict,
            )
        except (FileNotFoundError, ValueError) as exc:
            if strict:
                raise
            print(f"Warning: skipping dataset '{dataset_name}': {exc}")
            continue

        print(
            "Loaded dataset: "
            f"{bundle.name}, X={bundle.X.shape}, "
            f"classes={dict(enumerate(bundle.target_names))}, fs={bundle.fs} Hz"
        )
        bundles.append(bundle)

    if not bundles:
        raise RuntimeError("No dataset could be loaded for the benchmark")
    return bundles


def job_id(dataset, preprocessor_spec, classifier_spec, fit_method_spec):
    """Return a stable id for one benchmark combination."""

    return "__".join(
        _slug(value)
        for value in (
            dataset.name,
            preprocessor_spec.name,
            classifier_spec.key,
            fit_method_spec.key,
        )
    )


def iter_benchmark_jobs(
    datasets,
    preprocessor_specs,
    classifier_specs,
    fit_method_specs,
    completed_task_ids=None,
    limit=None,
):
    """Yield plain tuple jobs for the selected benchmark matrix."""

    completed_task_ids = set(completed_task_ids or ())
    skipped_count = 0
    yielded_count = 0

    for dataset in datasets:
        for preprocessor_spec in preprocessor_specs:
            for classifier_spec in classifier_specs:
                for fit_method_spec in fit_method_specs:
                    task_id = job_id(
                        dataset,
                        preprocessor_spec,
                        classifier_spec,
                        fit_method_spec,
                    )
                    if task_id in completed_task_ids:
                        skipped_count += 1
                        continue
                    if limit is not None and yielded_count >= int(limit):
                        if skipped_count:
                            print(f"Resume: skipping {skipped_count} recorded tasks")
                        return
                    yielded_count += 1
                    yield (
                        dataset,
                        preprocessor_spec,
                        classifier_spec,
                        fit_method_spec,
                    )

    if skipped_count:
        print(f"Resume: skipping {skipped_count} recorded tasks")


def _dataset_class_counts(dataset):
    return {
        str(target_name): int(np.sum(dataset.y == label))
        for label, target_name in enumerate(dataset.target_names)
    }


def _job_metadata(dataset, preprocessor_spec, classifier_spec, fit_method_spec):
    return {
        "task_id": job_id(
            dataset,
            preprocessor_spec,
            classifier_spec,
            fit_method_spec,
        ),
        "dataset": dataset.name,
        "dataset_n_samples": int(dataset.X.shape[0]),
        "dataset_n_classes": int(len(np.unique(dataset.y))),
        "dataset_fs": int(dataset.fs),
        "dataset_target_names": tuple(dataset.target_names),
        "dataset_class_counts": _dataset_class_counts(dataset),
        "fit_method_label": fit_method_spec.label,
        "fit_method_comment": fit_method_spec.comment,
        "preprocessor_comment": preprocessor_spec.comment,
        "classifier_comment": classifier_spec.comment,
    }


def _error_result_row(
    fit_method_spec,
    preprocessor_spec,
    classifier_spec,
    error,
    duration_seconds=0.0,
):
    return {
        "fit_method": fit_method_spec.key,
        "preprocessor": preprocessor_spec.name,
        "preprocessor_label": preprocessor_spec.label,
        "classifier": classifier_spec.key,
        "classifier_label": classifier_spec.label,
        "classifier_family": classifier_spec.family,
        "meta_classifier": classifier_spec.meta_key,
        "base_classifier": classifier_spec.base_key,
        "duration_seconds": float(duration_seconds),
        "status": "error",
        "error": f"{type(error).__name__}: {error}",
        "score": np.nan,
        "score_std": np.nan,
        "n_features": np.nan,
        "preprocessor_params": {},
        "classifier_params": {},
    }


def run_benchmark_job(
    dataset,
    preprocessor_spec,
    classifier_spec,
    fit_method_spec,
    cv_splits=6,
    n_jobs=1,
    test_size=0.2,
    random_state=RANDOM_SEED,
    scoring="accuracy",
    catch_errors=True,
):
    """Run one benchmark combination and return normalized rows."""

    try:
        result_rows = run_fit_method(
            fit_method_spec=fit_method_spec,
            X=dataset.X,
            y=dataset.y,
            preprocessor_spec=preprocessor_spec,
            classifier_spec=classifier_spec,
            cv_splits=cv_splits,
            n_jobs=n_jobs,
            test_size=test_size,
            random_state=random_state,
            scoring=scoring,
            catch_errors=catch_errors,
        )
    except Exception as exc:
        if not catch_errors:
            raise
        result_rows = [
            _error_result_row(
                fit_method_spec=fit_method_spec,
                preprocessor_spec=preprocessor_spec,
                classifier_spec=classifier_spec,
                error=exc,
            )
        ]

    metadata = _job_metadata(
        dataset,
        preprocessor_spec,
        classifier_spec,
        fit_method_spec,
    )
    rows = []
    for row_index, result_row in enumerate(result_rows, start=1):
        rows.append(
            {
                "experiment_id": f"{metadata['task_id']}__{row_index:04d}",
                "row_index": row_index,
                **metadata,
                **result_row,
            }
        )
    return rows


def run_benchmark_jobs(
    jobs,
    cv_splits=6,
    n_jobs=1,
    test_size=0.2,
    random_state=RANDOM_SEED,
    scoring="accuracy",
):
    """Yield normalized rows for each tuple job."""

    jobs = list(jobs)
    total_jobs = len(jobs)
    for job_index, job in enumerate(jobs, start=1):
        dataset, preprocessor_spec, classifier_spec, fit_method_spec = job
        print(
            f"[{job_index}/{total_jobs}] "
            f"{dataset.name} | "
            f"{preprocessor_spec.name} | "
            f"{classifier_spec.key} | "
            f"{fit_method_spec.key}"
        )
        yield run_benchmark_job(
            dataset=dataset,
            preprocessor_spec=preprocessor_spec,
            classifier_spec=classifier_spec,
            fit_method_spec=fit_method_spec,
            cv_splits=cv_splits,
            n_jobs=n_jobs,
            test_size=test_size,
            random_state=random_state,
            scoring=scoring,
            catch_errors=True,
        )
