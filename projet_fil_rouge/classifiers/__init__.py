"""Classifier registry and resolver.

Classifier modules only create sklearn-like estimators.
"""

from copy import deepcopy

from sklearn.base import clone

from .ensembles.adaboost import (
    ScratchAdaBoostClassifier,
    default_param_grid as default_adaboost_param_grid,
    default_scratch_param_grid,
    make_classifier as make_adaboost_classifier,
    make_scratch_classifier,
)
from .ensembles.bagging import (
    default_param_grid as default_bagging_tree_param_grid,
    make_tree_bagging_classifier,
)
from .ensembles.gradient_boosting import (
    default_param_grid as default_gradient_boosting_param_grid,
    make_classifier as make_gradient_boosting_classifier,
)
from .ensembles.random_forest import (
    default_param_grid as default_random_forest_param_grid,
    make_classifier as make_random_forest_classifier,
)
from .logistic_regression import (
    default_param_grid as default_logistic_regression_param_grid,
    make_classifier as make_logistic_regression_classifier,
)
from .neural_network import (
    default_param_grid as default_neural_network_param_grid,
    make_classifier as make_neural_network_classifier,
)
from .svm import default_param_grid as default_svm_param_grid
from .svm import make_classifier as make_svm_classifier


CLASSIFIERS = {
    "logistic_regression": make_logistic_regression_classifier,
    "svm": make_svm_classifier,
    "neural_network": make_neural_network_classifier,
    "adaboost": make_adaboost_classifier,
    "scratch_adaboost": make_scratch_classifier,
    "gradient_boosting": make_gradient_boosting_classifier,
    "bagging_tree": make_tree_bagging_classifier,
    "random_forest": make_random_forest_classifier,
}


CLASSIFIER_PARAM_GRIDS = {
    "logistic_regression": default_logistic_regression_param_grid,
    "svm": default_svm_param_grid,
    "neural_network": default_neural_network_param_grid,
    "adaboost": default_adaboost_param_grid,
    "scratch_adaboost": default_scratch_param_grid,
    "gradient_boosting": default_gradient_boosting_param_grid,
    "bagging_tree": default_bagging_tree_param_grid,
    "random_forest": default_random_forest_param_grid,
}


def make_classifier(classifier, **params):
    """Create a fresh classifier from a registry name, class, factory, or estimator."""

    if classifier is None:
        raise ValueError("classifier is required")

    if isinstance(classifier, str):
        if classifier not in CLASSIFIERS:
            valid_names = ", ".join(sorted(CLASSIFIERS))
            raise ValueError(f"Unknown classifier '{classifier}'. Expected one of: {valid_names}")
        return CLASSIFIERS[classifier](**params)

    if isinstance(classifier, type):
        return classifier(**params)

    if callable(classifier) and not hasattr(classifier, "fit"):
        return classifier(**params)

    estimator = clone(classifier)
    if params:
        estimator.set_params(**params)
    return estimator


def get_classifier_param_grid(name):
    """Return the default unprefixed parameter grid for a registered classifier."""

    if name not in CLASSIFIER_PARAM_GRIDS:
        valid_names = ", ".join(sorted(CLASSIFIER_PARAM_GRIDS))
        raise ValueError(f"Unknown classifier grid '{name}'. Expected one of: {valid_names}")
    return deepcopy(CLASSIFIER_PARAM_GRIDS[name]())
