"""AdaBoost classifiers"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

try:
    from ...config import RANDOM_SEED
except ImportError:  # Allows importing from a notebook whose cwd is projet_fil_rouge/
    from config import RANDOM_SEED


def make_classifier(
    n_estimators=100,
    learning_rate=1.0,
    max_depth=2,
    random_state=RANDOM_SEED,
    **params,
):
    base_estimator = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    try:
        return AdaBoostClassifier(
            estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state,
            **params,
        )
    except TypeError:
        return AdaBoostClassifier(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state,
            **params,
        )


def default_param_grid():
    return {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.1, 0.5, 1.0],
    }


class ScratchAdaBoostClassifier(BaseEstimator, ClassifierMixin):
    """Binary AdaBoost implementation for the pedagogical exercise."""

    def __init__(self, base_classifier=None, n_estimators=100, random_state=RANDOM_SEED):
        self.base_classifier = base_classifier
        self.n_estimators = n_estimators
        self.random_state = random_state

    def _make_base_classifier(self, estimator_index):
        if self.base_classifier is None:
            return DecisionTreeClassifier(max_depth=2, random_state=self.random_state + estimator_index)
        estimator = clone(self.base_classifier)
        if hasattr(estimator, "get_params") and "random_state" in estimator.get_params():
            estimator.set_params(random_state=self.random_state + estimator_index)
        return estimator

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("ScratchAdaBoostClassifier supports binary classification only")

        y_signed = np.where(y == self.classes_[0], -1, 1)
        weights = np.ones(X.shape[0]) / X.shape[0]
        self.estimators_ = []
        self.alphas_ = []

        for estimator_index in range(int(self.n_estimators)):
            estimator = self._make_base_classifier(estimator_index)
            estimator.fit(X, y_signed, sample_weight=weights)
            pred = estimator.predict(X)
            errors = (pred != y_signed).astype(float)
            epsilon = np.sum(weights * errors) / np.sum(weights)
            epsilon = np.clip(epsilon, 1e-12, 1 - 1e-12)

            if epsilon >= 0.5:
                continue

            alpha = np.log((1 - epsilon) / epsilon)
            weights = weights * np.exp(alpha * errors)
            weights = weights / weights.sum()
            self.estimators_.append(estimator)
            self.alphas_.append(alpha)

        if not self.estimators_:
            raise ValueError("No useful weak learner was fitted")
        return self

    def predict(self, X):
        scores = np.zeros(np.asarray(X).shape[0])
        for alpha, estimator in zip(self.alphas_, self.estimators_):
            scores += alpha * estimator.predict(X)
        signed_pred = np.where(scores >= 0, 1, -1)
        return np.where(signed_pred == -1, self.classes_[0], self.classes_[1])


def make_scratch_classifier(base_classifier=None, n_estimators=100, random_state=RANDOM_SEED):
    return ScratchAdaBoostClassifier(
        base_classifier=base_classifier,
        n_estimators=n_estimators,
        random_state=random_state,
    )


def default_scratch_param_grid():
    return {
        "n_estimators": [25, 50, 100],
    }
