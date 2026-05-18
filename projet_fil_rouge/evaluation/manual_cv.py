"""Manual cross-validation and grid-search loops for pedagogical exercises."""

import numpy as np
from sklearn.model_selection import ParameterGrid


def _fit_transform_preprocessor(preprocessor, X_train, X_valid):
    if preprocessor is None:
        return X_train, X_valid

    if hasattr(preprocessor, "fit_transform"):
        X_train_features = preprocessor.fit_transform(X_train)
    else:
        preprocessor.fit(X_train)
        X_train_features = preprocessor.transform(X_train)
    X_valid_features = preprocessor.transform(X_valid)
    return X_train_features, X_valid_features


def manual_loo_score(
    X_raw,
    y,
    preprocessor_factory,
    classifier_factory,
    preprocessor_params=None,
    classifier_params=None,
    verbose=False,
):
    """Leave-One-Out score with explicit train-only preprocessing per fold."""

    X_raw = np.asarray(X_raw)
    y = np.asarray(y)
    preprocessor_params = preprocessor_params or {}
    classifier_params = classifier_params or {}

    y_pred = []
    fold_scores = []

    for fold_index in range(X_raw.shape[0]):
        train_mask = np.ones(X_raw.shape[0], dtype=bool)
        train_mask[fold_index] = False

        X_train_raw = X_raw[train_mask]
        X_valid_raw = X_raw[~train_mask]
        y_train = y[train_mask]
        y_valid = y[~train_mask]

        preprocessor = preprocessor_factory(**preprocessor_params) if preprocessor_factory else None
        X_train, X_valid = _fit_transform_preprocessor(preprocessor, X_train_raw, X_valid_raw)

        classifier = classifier_factory(**classifier_params)
        classifier.fit(X_train, y_train)
        pred = classifier.predict(X_valid)

        y_pred.extend(pred)
        fold_score = float(np.mean(pred == y_valid))
        fold_scores.append(fold_score)
        if verbose:
            print(f"Fold {fold_index + 1}: accuracy={fold_score:.1f}")

    score = float(np.mean(fold_scores))
    return {
        "score": score,
        "y_true": y.copy(),
        "y_pred": np.asarray(y_pred),
        "fold_scores": np.asarray(fold_scores),
        "preprocessor_params": dict(preprocessor_params),
        "classifier_params": dict(classifier_params),
    }


def manual_grid_search(
    X_raw,
    y,
    preprocessor_factory,
    classifier_factory,
    preprocessor_param_grid=None,
    classifier_param_grid=None,
    verbose=False,
):
    """Manual grid search using manual_loo_score for each parameter pair."""

    preprocessor_grid = list(ParameterGrid(preprocessor_param_grid or {}))
    classifier_grid = list(ParameterGrid(classifier_param_grid or {}))
    results = []
    best_result = None

    for preprocessor_params in preprocessor_grid:
        for classifier_params in classifier_grid:
            result = manual_loo_score(
                X_raw=X_raw,
                y=y,
                preprocessor_factory=preprocessor_factory,
                classifier_factory=classifier_factory,
                preprocessor_params=preprocessor_params,
                classifier_params=classifier_params,
                verbose=False,
            )
            results.append(result)
            if best_result is None or result["score"] > best_result["score"]:
                best_result = result
            if verbose:
                print(
                    "score={:.4f} preprocessor={} classifier={}".format(
                        result["score"],
                        preprocessor_params,
                        classifier_params,
                    )
                )

    return {
        "best_score": best_result["score"],
        "best_preprocessor_params": best_result["preprocessor_params"],
        "best_classifier_params": best_result["classifier_params"],
        "results": results,
    }
