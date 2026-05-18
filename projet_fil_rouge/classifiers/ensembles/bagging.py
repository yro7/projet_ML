"""Composable bagging meta-classifier."""

import copy

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone

try:
    from ...config import RANDOM_SEED
except ImportError:  # Allows importing from a notebook whose cwd is projet_fil_rouge/
    from config import RANDOM_SEED


class BaggingClassifier(BaseEstimator, ClassifierMixin):
    """Bootstrap aggregation around any sklearn-like base classifier."""

    def __init__(self, base_classifier, n_estimators=100, random_state=RANDOM_SEED):
        self.base_classifier = base_classifier
        self.n_estimators = n_estimators
        self.random_state = random_state

    def _new_base_classifier(self, estimator_index):
        if callable(self.base_classifier) and not hasattr(self.base_classifier, "fit"):
            estimator = self.base_classifier()
        else:
            try:
                estimator = clone(self.base_classifier)
            except TypeError:
                estimator = copy.deepcopy(self.base_classifier)

        if hasattr(estimator, "get_params") and "random_state" in estimator.get_params():
            estimator.set_params(random_state=None if self.random_state is None else self.random_state + estimator_index)
        return estimator

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples")

        rng = np.random.default_rng(self.random_state)
        self.classes_ = np.unique(y)
        self.estimators_ = []
        self.bootstrap_indices_ = []

        for estimator_index in range(int(self.n_estimators)):
            indices = rng.choice(X.shape[0], size=X.shape[0], replace=True)
            estimator = self._new_base_classifier(estimator_index)
            estimator.fit(X[indices], y[indices])
            self.estimators_.append(estimator)
            self.bootstrap_indices_.append(indices)
        return self

    def predict(self, X):
        if not hasattr(self, "estimators_"):
            raise ValueError("This BaggingClassifier instance is not fitted yet")

        predictions = np.asarray([estimator.predict(X) for estimator in self.estimators_])
        majority_predictions = []
        for sample_predictions in predictions.T:
            values, counts = np.unique(sample_predictions, return_counts=True)
            majority_predictions.append(values[np.argmax(counts)])
        return np.asarray(majority_predictions)

    def score(self, X, y):
        return np.mean(self.predict(X) == np.asarray(y))


def make_classifier(base_classifier, n_estimators=100, random_state=RANDOM_SEED):
    return BaggingClassifier(
        base_classifier=base_classifier,
        n_estimators=n_estimators,
        random_state=random_state,
    )


def default_param_grid():
    return {
        "n_estimators": [50, 100, 200],
    }
