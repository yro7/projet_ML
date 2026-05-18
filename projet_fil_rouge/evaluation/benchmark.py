"""Benchmark and GridSearch orchestration.

This module owns split/CV timing so preprocessing is fitted on train folds only.
"""

import numpy as np
from sklearn.model_selection import GridSearchCV, LeaveOneOut, train_test_split
from sklearn.pipeline import Pipeline

try:
    from .metrics import classification_summary
except ImportError:  # Allows importing from a notebook whose cwd is projet_fil_rouge/
    from evaluation.metrics import classification_summary

try:
    from ..utils.preprocessings import make_preprocessor
except ImportError:  # Allows importing from a notebook whose cwd is projet_fil_rouge/
    from utils.preprocessings import make_preprocessor

try:
    from ..config import RANDOM_SEED
except ImportError:  # Allows importing from a notebook whose cwd is projet_fil_rouge/
    from config import RANDOM_SEED


def _prefixed_grid(param_grid, prefix):
    return {f"{prefix}__{name}": values for name, values in (param_grid or {}).items()}


def _split_best_params(best_params):
    grouped = {"preprocessor": {}, "classifier": {}, "raw": dict(best_params)}
    for name, value in best_params.items():
        if name.startswith("preprocessor__"):
            grouped["preprocessor"][name.removeprefix("preprocessor__")] = value
        elif name.startswith("classifier__"):
            grouped["classifier"][name.removeprefix("classifier__")] = value
    return grouped


def build_estimator_pipeline(classifier_factory, preprocessor=None):
    """Build a sklearn Pipeline from optional preprocessor and classifier."""

    steps = []
    preprocessor_estimator = make_preprocessor(preprocessor)
    if preprocessor_estimator is not None:
        steps.append(("preprocessor", preprocessor_estimator))
    steps.append(("classifier", classifier_factory()))
    return Pipeline(steps)


def run_grid_search(
    classifier_factory,
    classifier_param_grid,
    X_train,
    y_train,
    X_test=None,
    y_test=None,
    preprocessor=None,
    preprocessor_param_grid=None,
    cv=None,
    scoring="accuracy",
    n_jobs=-1,
    refit=True,
):
    """Run GridSearchCV over preprocessing and classifier parameters."""

    if preprocessor is None and preprocessor_param_grid:
        raise ValueError("preprocessor_param_grid requires a preprocessor")

    estimator = build_estimator_pipeline(
        classifier_factory=classifier_factory,
        preprocessor=preprocessor,
    )
    param_grid = {}
    param_grid.update(_prefixed_grid(preprocessor_param_grid, "preprocessor"))
    param_grid.update(_prefixed_grid(classifier_param_grid, "classifier"))

    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid or {},
        cv=cv or LeaveOneOut(),
        scoring=scoring,
        n_jobs=n_jobs,
        refit=refit,
    )
    grid_search.fit(X_train, y_train)

    result = {
        "grid_search": grid_search,
        "best_estimator": grid_search.best_estimator_,
        "best_params": _split_best_params(grid_search.best_params_),
        "best_score": grid_search.best_score_,
        "cv_results": grid_search.cv_results_,
    }

    if X_test is not None and y_test is not None:
        y_pred = grid_search.predict(X_test)
        result.update(
            {
                "test_score": grid_search.score(X_test, y_test),
                "y_test_pred": y_pred,
                "test_metrics": classification_summary(y_test, y_pred),
            }
        )
    return result


def train_test_benchmark(
    X_raw,
    y,
    classifier_factory,
    classifier_params=None,
    preprocessor=None,
    preprocessor_params=None,
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=True,
):
    """Simple train/test benchmark with train-only preprocessing."""

    y = np.asarray(y)
    stratify_values = y if stratify else None
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_values,
    )

    preprocessor_estimator = make_preprocessor(preprocessor, **(preprocessor_params or {}))
    if preprocessor_estimator is None:
        X_train = X_train_raw
        X_test = X_test_raw
    else:
        X_train = preprocessor_estimator.fit_transform(X_train_raw)
        X_test = preprocessor_estimator.transform(X_test_raw)

    classifier = classifier_factory(**(classifier_params or {}))
    classifier.fit(X_train, y_train)
    y_train_pred = classifier.predict(X_train)
    y_test_pred = classifier.predict(X_test)

    return {
        "preprocessor": preprocessor,
        "preprocessor_estimator": preprocessor_estimator,
        "classifier": classifier,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_train_pred": y_train_pred,
        "y_test_pred": y_test_pred,
        "train_metrics": classification_summary(y_train, y_train_pred),
        "test_metrics": classification_summary(y_test, y_test_pred),
    }
