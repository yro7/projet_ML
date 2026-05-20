"""Fit/evaluation method specifications for aggressive benchmarks."""

from dataclasses import dataclass
import time

import numpy as np
from sklearn.model_selection import ParameterGrid, StratifiedKFold

try:
    from .benchmark import run_grid_search, train_test_benchmark
    from .manual_cv import manual_loo_score
except ImportError:  # Allows importing from a notebook whose cwd is projet_fil_rouge/
    from evaluation.benchmark import run_grid_search, train_test_benchmark
    from evaluation.manual_cv import manual_loo_score

try:
    from ..config import RANDOM_SEED
    from ..utils.preprocessings import make_preprocessor
except ImportError:  # Allows importing from a notebook whose cwd is projet_fil_rouge/
    from config import RANDOM_SEED
    from utils.preprocessings import make_preprocessor


@dataclass(frozen=True)
class FitMethodSpec:
    """One strategy used to fit and evaluate a classifier pipeline."""

    key: str
    label: str
    runner: object
    comment: str = ""


def _spec_attr(spec, name, default=None):
    if spec is None:
        return default
    if isinstance(spec, dict):
        return spec.get(name, default)
    return getattr(spec, name, default)


def _params_grid(param_grid):
    return list(ParameterGrid(param_grid or {}))


def _params_repr(params):
    return dict(params or {})


def _best_score_std(grid_search):
    best_index = int(grid_search.best_index_)
    return float(grid_search.cv_results_["std_test_score"][best_index])


def _feature_count_from_estimator(estimator, X):
    preprocessor = estimator.named_steps.get("preprocessor")
    if preprocessor is None:
        return int(X.shape[1])
    return int(preprocessor.transform(X).shape[1])


def _feature_count_from_params(X, y, preprocessor, preprocessor_params):
    preprocessor_estimator = make_preprocessor(
        preprocessor,
        **(preprocessor_params or {}),
    )
    if preprocessor_estimator is None:
        return int(X.shape[1])
    features = preprocessor_estimator.fit_transform(X, y)
    return int(features.shape[1])


def _base_row(
    fit_method,
    preprocessor_spec,
    classifier_spec,
    duration_seconds,
    status="ok",
    error="",
):
    return {
        "fit_method": fit_method,
        "preprocessor": _spec_attr(preprocessor_spec, "name"),
        "preprocessor_label": _spec_attr(preprocessor_spec, "label"),
        "classifier": _spec_attr(classifier_spec, "key"),
        "classifier_label": _spec_attr(classifier_spec, "label"),
        "classifier_family": _spec_attr(classifier_spec, "family"),
        "meta_classifier": _spec_attr(classifier_spec, "meta_key"),
        "base_classifier": _spec_attr(classifier_spec, "base_key"),
        "duration_seconds": float(duration_seconds),
        "status": status,
        "error": error,
    }


def _result_row(
    fit_method,
    preprocessor_spec,
    classifier_spec,
    score,
    score_std,
    n_features,
    preprocessor_params,
    classifier_params,
    duration_seconds,
    status="ok",
    error="",
):
    row = _base_row(
        fit_method=fit_method,
        preprocessor_spec=preprocessor_spec,
        classifier_spec=classifier_spec,
        duration_seconds=duration_seconds,
        status=status,
        error=error,
    )
    row.update(
        {
            "score": np.nan if score is None else float(score),
            "score_std": np.nan if score_std is None else float(score_std),
            "n_features": np.nan if n_features is None else int(n_features),
            "preprocessor_params": _params_repr(preprocessor_params),
            "classifier_params": _params_repr(classifier_params),
        }
    )
    return row


def _error_row(
    fit_method,
    preprocessor_spec,
    classifier_spec,
    duration_seconds,
    error,
    preprocessor_params=None,
    classifier_params=None,
):
    return _result_row(
        fit_method=fit_method,
        preprocessor_spec=preprocessor_spec,
        classifier_spec=classifier_spec,
        score=None,
        score_std=None,
        n_features=None,
        preprocessor_params=preprocessor_params,
        classifier_params=classifier_params,
        duration_seconds=duration_seconds,
        status="error",
        error=f"{type(error).__name__}: {error}",
    )


def run_grid_search_fit_method(
    X,
    y,
    preprocessor_spec,
    classifier_spec,
    cv_splits=6,
    random_state=RANDOM_SEED,
    n_jobs=1,
    scoring="accuracy",
    catch_errors=True,
    **params,
):
    """Run GridSearchCV and return one normalized result row."""

    started_at = time.perf_counter()
    try:
        cv = StratifiedKFold(
            n_splits=cv_splits,
            shuffle=True,
            random_state=random_state,
        )
        result = run_grid_search(
            X_train=X,
            y_train=y,
            preprocessor=_spec_attr(preprocessor_spec, "name"),
            preprocessor_param_grid=_spec_attr(preprocessor_spec, "param_grid", {}),
            classifier=_spec_attr(classifier_spec, "classifier"),
            classifier_param_grid=_spec_attr(classifier_spec, "param_grid", {}),
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
        )
        return [
            _result_row(
                fit_method="grid_search_cv",
                preprocessor_spec=preprocessor_spec,
                classifier_spec=classifier_spec,
                score=result["best_score"],
                score_std=_best_score_std(result["grid_search"]),
                n_features=_feature_count_from_estimator(result["best_estimator"], X),
                preprocessor_params=result["best_params"]["preprocessor"],
                classifier_params=result["best_params"]["classifier"],
                duration_seconds=time.perf_counter() - started_at,
            )
        ]
    except Exception as exc:
        if not catch_errors:
            raise
        return [
            _error_row(
                fit_method="grid_search_cv",
                preprocessor_spec=preprocessor_spec,
                classifier_spec=classifier_spec,
                duration_seconds=time.perf_counter() - started_at,
                error=exc,
            )
        ]


def run_train_test_fit_method(
    X,
    y,
    preprocessor_spec,
    classifier_spec,
    test_size=0.2,
    random_state=RANDOM_SEED,
    catch_errors=True,
    **params,
):
    """Run one train/test benchmark per explicit parameter combination."""

    rows = []
    for preprocessor_params in _params_grid(_spec_attr(preprocessor_spec, "param_grid", {})):
        for classifier_params in _params_grid(_spec_attr(classifier_spec, "param_grid", {})):
            started_at = time.perf_counter()
            try:
                result = train_test_benchmark(
                    X_raw=X,
                    y=y,
                    preprocessor=_spec_attr(preprocessor_spec, "name"),
                    preprocessor_params=preprocessor_params,
                    classifier=_spec_attr(classifier_spec, "classifier"),
                    classifier_params=classifier_params,
                    test_size=test_size,
                    random_state=random_state,
                )
                rows.append(
                    _result_row(
                        fit_method="train_test",
                        preprocessor_spec=preprocessor_spec,
                        classifier_spec=classifier_spec,
                        score=result["test_metrics"]["accuracy"],
                        score_std=None,
                        n_features=result["X_train"].shape[1],
                        preprocessor_params=preprocessor_params,
                        classifier_params=classifier_params,
                        duration_seconds=time.perf_counter() - started_at,
                    )
                )
            except Exception as exc:
                if not catch_errors:
                    raise
                rows.append(
                    _error_row(
                        fit_method="train_test",
                        preprocessor_spec=preprocessor_spec,
                        classifier_spec=classifier_spec,
                        duration_seconds=time.perf_counter() - started_at,
                        error=exc,
                        preprocessor_params=preprocessor_params,
                        classifier_params=classifier_params,
                    )
                )
    return rows


def run_manual_loo_fit_method(
    X,
    y,
    preprocessor_spec,
    classifier_spec,
    catch_errors=True,
    **params,
):
    """Run manual leave-one-out per explicit parameter combination."""

    rows = []
    for preprocessor_params in _params_grid(_spec_attr(preprocessor_spec, "param_grid", {})):
        for classifier_params in _params_grid(_spec_attr(classifier_spec, "param_grid", {})):
            started_at = time.perf_counter()
            try:
                result = manual_loo_score(
                    X_raw=X,
                    y=y,
                    preprocessor=_spec_attr(preprocessor_spec, "name"),
                    preprocessor_params=preprocessor_params,
                    classifier=_spec_attr(classifier_spec, "classifier"),
                    classifier_params=classifier_params,
                )
                rows.append(
                    _result_row(
                        fit_method="manual_loo",
                        preprocessor_spec=preprocessor_spec,
                        classifier_spec=classifier_spec,
                        score=result["score"],
                        score_std=np.std(result["fold_scores"]),
                        n_features=_feature_count_from_params(
                            X,
                            y,
                            _spec_attr(preprocessor_spec, "name"),
                            preprocessor_params,
                        ),
                        preprocessor_params=preprocessor_params,
                        classifier_params=classifier_params,
                        duration_seconds=time.perf_counter() - started_at,
                    )
                )
            except Exception as exc:
                if not catch_errors:
                    raise
                rows.append(
                    _error_row(
                        fit_method="manual_loo",
                        preprocessor_spec=preprocessor_spec,
                        classifier_spec=classifier_spec,
                        duration_seconds=time.perf_counter() - started_at,
                        error=exc,
                        preprocessor_params=preprocessor_params,
                        classifier_params=classifier_params,
                    )
                )
    return rows


def create_fit_method_specs():
    """Return all fit/evaluation method specs used by aggressive benchmarks."""

    return [
        FitMethodSpec(
            key="grid_search_cv",
            label="GridSearchCV",
            runner=run_grid_search_fit_method,
            comment="Cross-validated grid search over preprocessing and classifier params.",
        ),
        FitMethodSpec(
            key="train_test",
            label="Train/Test",
            runner=run_train_test_fit_method,
            comment="Single train/test split for each explicit parameter combination.",
        ),
        FitMethodSpec(
            key="manual_loo",
            label="Manual LOO",
            runner=run_manual_loo_fit_method,
            comment="Pedagogical leave-one-out loop for each explicit parameter combination.",
        ),
    ]


def fit_method_specs_by_key():
    """Return fit method specs indexed by key."""

    return {spec.key: spec for spec in create_fit_method_specs()}


def run_fit_method(fit_method_spec, X, y, preprocessor_spec, classifier_spec, **params):
    """Run a fit method spec and return normalized result rows."""

    return fit_method_spec.runner(
        X=X,
        y=y,
        preprocessor_spec=preprocessor_spec,
        classifier_spec=classifier_spec,
        **params,
    )
