"""Compare preprocessing methods for Part V.
Each method is called by its registry name, and each score is produced by the shared
run_grid_search helper.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

try:
    from .classifiers import get_classifier_param_grid
    from .config import RANDOM_SEED, WORDS, seed_everything
    from .data import load_dataset
    from .evaluation.benchmark import run_grid_search
except ImportError:  # Allows running with: python3 part5_compare_preprocess.py
    from classifiers import get_classifier_param_grid
    from config import RANDOM_SEED, WORDS, seed_everything
    from data import load_dataset
    from evaluation.benchmark import run_grid_search


@dataclass(frozen=True)
class PreprocessorSpec:
    """One preprocessing strategy and the compact grid used for comparison."""

    label: str
    name: str
    param_grid: dict
    comment: str


def make_preprocessor_specs(fast=False):
    """Return curated, comparable preprocessing grids.

    The grids are intentionally smaller than the default project grids because
    Part V should compare many preprocessing families without making the
    notebook painfully slow.
    """

    fft_max_values = [500] if fast else [500, 1_000]
    n_components = [5] if fast else [5, 10]
    wavelet_levels = [4] if fast else [3, 4]
    mfcc_values = [13] if fast else [13, 20]

    return [
        PreprocessorSpec(
            label="FFT + PCA",
            name="fft_pca",
            param_grid={
                "idx_frequence_max": fft_max_values,
                "n_components": n_components,
                "scale": [True],
            },
            comment="Baseline reduction lineaire non supervisee.",
        ),
        PreprocessorSpec(
            label="FFT + Kernel PCA",
            name="fft_kernel_pca",
            param_grid={
                "idx_frequence_max": fft_max_values,
                "n_components": n_components,
                "kernel": ["rbf"] if fast else ["rbf", "cosine"],
                "gamma": [None],
                "scale": [True],
            },
            comment="Reduction non lineaire compatible train/test.",
        ),
        PreprocessorSpec(
            label="FFT + LDA",
            name="fft_lda",
            param_grid={
                "idx_frequence_max": fft_max_values,
                "n_components": [None],
                "scale": [True],
            },
            comment="Reduction supervisee, au plus n_classes - 1 dimensions.",
        ),
        PreprocessorSpec(
            label="FFT + NMF",
            name="fft_nmf",
            param_grid={
                "idx_frequence_max": fft_max_values,
                "n_components": n_components,
                "scale": [True],
            },
            comment="Factorisation non negative sur magnitudes FFT.",
        ),
        PreprocessorSpec(
            label="FFT + SVD",
            name="fft_svd",
            param_grid={
                "idx_frequence_max": fft_max_values,
                "n_components": n_components,
                "scale": [True],
            },
            comment="Alternative lineaire proche PCA, sans centrage explicite.",
        ),
        PreprocessorSpec(
            label="STFT stats",
            name="stft",
            param_grid={
                "stat": ["mean"] if fast else ["mean", "max"],
                "idx_frequence_max": [500],
                "nperseg": [400],
            },
            comment="Resume temps-frequence par statistique temporelle.",
        ),
        PreprocessorSpec(
            label="MFCC stats",
            name="mfcc",
            param_grid={
                "stat": ["mean"] if fast else ["mean", "max"],
                "n_mfcc": mfcc_values,
            },
            comment="Baseline audio compacte.",
        ),
        PreprocessorSpec(
            label="MFCC summary",
            name="mfcc_summary",
            param_grid={
                "n_mfcc": mfcc_values,
                "stats": [("mean", "std")],
                "include_delta": [False] if fast else [False, True],
                "include_delta2": [False],
                "scale": [True],
            },
            comment="MFCC enrichis par moyenne/ecart-type et deltas optionnels.",
        ),
        PreprocessorSpec(
            label="Wavelet",
            name="wavelet",
            param_grid={
                "wavelet": ["db4"] if fast else ["db4", "sym5"],
                "level": wavelet_levels,
                "representation": ["packet_energy"] if fast else ["packet_energy", "stats"],
                "n_components": [None],
                "scale": [True],
            },
            comment="Representation par ondelettes discretes.",
        ),
    ]


def _classifier_grid(classifier_name, fast=False):
    if classifier_name == "svm":
        return {
            "C": [1.0] if fast else [0.1, 1.0, 10.0],
            "gamma": ["scale"],
        }
    if fast:
        default_grid = get_classifier_param_grid(classifier_name)
        return {name: values[:1] for name, values in default_grid.items()}
    return get_classifier_param_grid(classifier_name)


def _format_best_params(best_params):
    if not best_params:
        return "{}"
    return ", ".join(f"{name}={value}" for name, value in sorted(best_params.items()))


def _best_score_std(grid_search):
    best_index = int(grid_search.best_index_)
    return float(grid_search.cv_results_["std_test_score"][best_index])


def _feature_count(best_estimator, X):
    preprocessor = best_estimator.named_steps.get("preprocessor")
    if preprocessor is None:
        return X.shape[1]
    return int(preprocessor.transform(X).shape[1])


def compare_preprocessors(
    X,
    y,
    specs=None,
    classifier="svm",
    classifier_param_grid=None,
    cv_splits=6,
    random_state=RANDOM_SEED,
    n_jobs=1,
    fast=False,
):
    """Compare preprocessing methods with the same CV and classifier family."""

    specs = list(specs or make_preprocessor_specs(fast=fast))
    classifier_param_grid = classifier_param_grid or _classifier_grid(classifier, fast=fast)
    cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=random_state,
    )

    rows = []
    for spec in specs:
        started_at = time.perf_counter()
        try:
            result = run_grid_search(
                X_train=X,
                y_train=y,
                preprocessor=spec.name,
                preprocessor_param_grid=spec.param_grid,
                classifier=classifier,
                classifier_param_grid=classifier_param_grid,
                cv=cv,
                n_jobs=n_jobs,
            )
            grid_search = result["grid_search"]
            rows.append(
                {
                    "label": spec.label,
                    "preprocessor": spec.name,
                    "cv_mean_accuracy": float(result["best_score"]),
                    "cv_std_accuracy": _best_score_std(grid_search),
                    "n_features": _feature_count(result["best_estimator"], X),
                    "best_preprocessor_params": _format_best_params(
                        result["best_params"]["preprocessor"]
                    ),
                    "best_classifier_params": _format_best_params(
                        result["best_params"]["classifier"]
                    ),
                    "duration_seconds": time.perf_counter() - started_at,
                    "comment": spec.comment,
                    "error": "",
                }
            )
        except Exception as exc:  # Keep the comparison robust in notebooks.
            rows.append(
                {
                    "label": spec.label,
                    "preprocessor": spec.name,
                    "cv_mean_accuracy": np.nan,
                    "cv_std_accuracy": np.nan,
                    "n_features": np.nan,
                    "best_preprocessor_params": "",
                    "best_classifier_params": "",
                    "duration_seconds": time.perf_counter() - started_at,
                    "comment": spec.comment,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    return sorted(
        rows,
        key=lambda row: (
            -np.inf if np.isnan(row["cv_mean_accuracy"]) else row["cv_mean_accuracy"]
        ),
        reverse=True,
    )


def print_comparison_table(rows):
    """Print a compact markdown table."""

    headers = [
        "rank",
        "method",
        "score",
        "std",
        "features",
        "time_s",
        "best_preprocess",
    ]
    table_rows = []
    for rank, row in enumerate(rows, start=1):
        if row["error"]:
            score = "ERROR"
            std = ""
        else:
            score = f"{row['cv_mean_accuracy']:.3f}"
            std = f"{row['cv_std_accuracy']:.3f}"
        table_rows.append(
            [
                str(rank),
                row["label"],
                score,
                std,
                str(row["n_features"]),
                f"{row['duration_seconds']:.1f}",
                row["best_preprocessor_params"] or row["error"],
            ]
        )

    widths = [
        max(len(str(value)) for value in [header] + [row[index] for row in table_rows])
        for index, header in enumerate(headers)
    ]

    def format_row(values):
        return "| " + " | ".join(
            str(value).ljust(width) for value, width in zip(values, widths)
        ) + " |"

    print(format_row(headers))
    print(format_row(["-" * width for width in widths]))
    for row in table_rows:
        print(format_row(row))


def write_csv(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare Part V preprocessing methods with cross-validation."
    )
    parser.add_argument("--classifier", default="svm")
    parser.add_argument("--cv", type=int, default=6)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--output-csv", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(RANDOM_SEED)
    X, y, genres, fs = load_dataset()

    print(f"Dataset: X={X.shape}, classes={dict(enumerate(WORDS))}, fs={fs} Hz")
    print(f"Classifier: {args.classifier}, CV={args.cv} folds")

    rows = compare_preprocessors(
        X=X,
        y=y,
        classifier=args.classifier,
        cv_splits=args.cv,
        n_jobs=args.n_jobs,
        fast=args.fast,
    )
    print_comparison_table(rows)

    if args.output_csv:
        output_path = write_csv(rows, args.output_csv)
        print(f"\nCSV written to: {output_path}")


if __name__ == "__main__":
    main()
