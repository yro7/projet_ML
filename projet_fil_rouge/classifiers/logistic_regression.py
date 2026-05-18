"""Logistic regression classifier definition."""

from sklearn.linear_model import LogisticRegression

try:
    from ..config import RANDOM_SEED
except ImportError:  # Allows importing from a notebook whose cwd is projet_fil_rouge/
    from config import RANDOM_SEED


def make_classifier(C=1.0, max_iter=1_000, random_state=RANDOM_SEED, **params):
    return LogisticRegression(
        C=C,
        max_iter=max_iter,
        random_state=random_state,
        **params,
    )


def default_param_grid():
    return {
        "C": [0.01, 0.1, 1.0, 10.0, 100.0],
    }
