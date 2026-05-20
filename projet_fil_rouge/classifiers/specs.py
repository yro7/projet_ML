"""Classifier experiment specifications.

This layer is intentionally above the existing classifier registry: the registry
keeps the simple notebook API, while specs describe benchmark combinations such
as "bagging + svm".
"""

from dataclasses import dataclass

from sklearn.base import clone

try:
    from ..config import RANDOM_SEED
    from .ensembles.bagging import BaggingClassifier
    from .logistic_regression import make_classifier as make_logistic_regression
    from .neural_network import make_classifier as make_neural_network
    from .svm import make_classifier as make_svm
except ImportError:  # Allows importing from a notebook whose cwd is projet_fil_rouge/
    from config import RANDOM_SEED
    from classifiers.ensembles.bagging import BaggingClassifier
    from classifiers.logistic_regression import make_classifier as make_logistic_regression
    from classifiers.neural_network import make_classifier as make_neural_network
    from classifiers.svm import make_classifier as make_svm


@dataclass(frozen=True)
class ClassifierSpec:
    """One classifier candidate for benchmark orchestration."""

    key: str
    label: str
    classifier: object
    param_grid: dict
    family: str
    base_key: str = None
    meta_key: str = None
    comment: str = ""


def _take(values, fast):
    return list(values[:1] if fast else values)


def _prefixed_grid(param_grid, prefix):
    return {f"{prefix}__{name}": values for name, values in param_grid.items()}


def _base_estimators():
    return {
        "logistic_regression": make_logistic_regression(max_iter=1_000),
        "svm": make_svm(),
        "neural_network": make_neural_network(max_iter=500),
    }


def _base_param_grids(fast=False):
    return {
        "logistic_regression": {
            "C": _take([1.0, 0.1, 10.0], fast),
        },
        "svm": {
            "C": _take([1.0, 0.1, 10.0], fast),
            "gamma": _take(["scale", 0.01], fast),
        },
        "neural_network": {
            "hidden_layer_sizes": _take([(32,), (16,), (32, 16)], fast),
            "alpha": _take([0.001, 0.0001], fast),
            "learning_rate_init": _take([0.001], fast),
        },
    }


def create_base_classifier_specs(fast=False):
    """Return standalone base classifiers."""

    estimators = _base_estimators()
    grids = _base_param_grids(fast=fast)
    return [
        ClassifierSpec(
            key="logistic_regression",
            label="Logistic Regression",
            classifier=estimators["logistic_regression"],
            param_grid=grids["logistic_regression"],
            family="base",
            base_key="logistic_regression",
            comment="Linear baseline.",
        ),
        ClassifierSpec(
            key="svm",
            label="SVM",
            classifier=estimators["svm"],
            param_grid=grids["svm"],
            family="base",
            base_key="svm",
            comment="Kernel method baseline.",
        ),
        ClassifierSpec(
            key="neural_network",
            label="Neural Network",
            classifier=estimators["neural_network"],
            param_grid=grids["neural_network"],
            family="base",
            base_key="neural_network",
            comment="MLPClassifier baseline.",
        ),
    ]


def create_bagging_classifier_specs(fast=False):
    """Return bagging meta-classifiers for each base classifier."""

    n_estimators_grid = [5] if fast else [10, 25]
    specs = []
    for base_spec in create_base_classifier_specs(fast=fast):
        bagging = BaggingClassifier(
            base_classifier=clone(base_spec.classifier),
            n_estimators=n_estimators_grid[0],
            random_state=RANDOM_SEED,
        )
        specs.append(
            ClassifierSpec(
                key=f"bagging_{base_spec.key}",
                label=f"Bagging {base_spec.label}",
                classifier=bagging,
                param_grid={
                    "n_estimators": n_estimators_grid,
                    **_prefixed_grid(base_spec.param_grid, "base_classifier"),
                },
                family="meta",
                base_key=base_spec.key,
                meta_key="bagging",
                comment=f"Bootstrap aggregation around {base_spec.label}.",
            )
        )
    return specs


def create_preset_classifier_specs(fast=False):
    """Return ensemble presets that are not generic meta x base combinations."""

    return [
        ClassifierSpec(
            key="bagging_tree",
            label="Bagging Tree",
            classifier="bagging_tree",
            param_grid={
                "n_estimators": [10] if fast else [25, 50],
                "max_depth": [2] if fast else [1, 2, 3],
            },
            family="preset",
            base_key="decision_tree",
            meta_key="bagging",
            comment="Existing shallow-tree bagging preset.",
        ),
        ClassifierSpec(
            key="random_forest",
            label="Random Forest",
            classifier="random_forest",
            param_grid={
                "n_estimators": [50] if fast else [100, 200],
                "max_depth": [None] if fast else [None, 5],
                "min_samples_leaf": [1] if fast else [1, 2],
            },
            family="preset",
            base_key="decision_tree",
            meta_key="random_forest",
            comment="Random forest preset.",
        ),
        ClassifierSpec(
            key="gradient_boosting",
            label="Gradient Boosting",
            classifier="gradient_boosting",
            param_grid={
                "n_estimators": [50] if fast else [50, 100],
                "learning_rate": [0.1] if fast else [0.01, 0.1],
            },
            family="preset",
            base_key="decision_tree",
            meta_key="gradient_boosting",
            comment="Gradient boosting preset.",
        ),
        ClassifierSpec(
            key="adaboost",
            label="AdaBoost",
            classifier="adaboost",
            param_grid={
                "n_estimators": [50] if fast else [50, 100],
                "learning_rate": [1.0] if fast else [0.1, 1.0],
            },
            family="preset",
            base_key="decision_tree",
            meta_key="adaboost",
            comment="sklearn AdaBoost preset with shallow decision trees.",
        ),
    ]


def create_all_classifier_specs(
    fast=False,
    include_bagging=True,
    include_presets=True,
):
    """Return all classifier specs used by aggressive benchmarks."""

    specs = create_base_classifier_specs(fast=fast)
    if include_bagging:
        specs.extend(create_bagging_classifier_specs(fast=fast))
    if include_presets:
        specs.extend(create_preset_classifier_specs(fast=fast))
    return specs


def classifier_specs_by_key(
    fast=False,
    include_bagging=True,
    include_presets=True,
):
    """Return classifier specs indexed by key."""

    return {
        spec.key: spec
        for spec in create_all_classifier_specs(
            fast=fast,
            include_bagging=include_bagging,
            include_presets=include_presets,
        )
    }
