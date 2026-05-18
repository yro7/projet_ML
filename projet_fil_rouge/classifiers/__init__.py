"""Classifier registry and resolver.

Classifier modules only create sklearn-like estimators.
"""

from sklearn.base import clone

from .ensembles.adaboost import make_classifier as make_adaboost_classifier
from .ensembles.bagging import make_tree_bagging_classifier
from .ensembles.gradient_boosting import make_classifier as make_gradient_boosting_classifier
from .logistic_regression import make_classifier as make_logistic_regression_classifier
from .neural_network import make_classifier as make_neural_network_classifier
from .svm import make_classifier as make_svm_classifier


CLASSIFIERS = {
    "logistic_regression": make_logistic_regression_classifier,
    "svm": make_svm_classifier,
    "neural_network": make_neural_network_classifier,
    "adaboost": make_adaboost_classifier,
    "gradient_boosting": make_gradient_boosting_classifier,
    "bagging_tree": make_tree_bagging_classifier,
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
