"""Main fonction qui orchestre le reste du projet"""

from torch.utils.hipify.hipify_python import preprocessor
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

try:
    from .classifiers.ensembles.bagging import BaggingClassifier
    from .classifiers.logistic_regression import make_classifier as make_logistic_regression
    from .classifiers.svm import make_classifier as make_svm
    from .config import RANDOM_SEED, WORDS, seed_everything
    from .data import load_dataset
    from .evaluation.benchmark import run_grid_search, train_test_benchmark
    from .evaluation.manual_cv import manual_loo_score
except ImportError:  # Pr faire python3 main.py from projet_fil_rouge/
    from classifiers.ensembles.bagging import BaggingClassifier
    from classifiers.logistic_regression import make_classifier as make_logistic_regression
    from classifiers.svm import make_classifier as make_svm
    from config import RANDOM_SEED, WORDS, seed_everything
    from data import load_dataset
    from evaluation.benchmark import run_grid_search, train_test_benchmark
    from evaluation.manual_cv import manual_loo_score


def print_section(title):
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


def main():
    seed_everything(RANDOM_SEED)

    print_section("1. Load dataset")
    X, y, genres, fs = load_dataset()
    print(f"X={X.shape} | y={y.shape} | genres={genres.shape} | fs={fs} Hz")
    print(f"Labels: {dict(enumerate(WORDS))}")

    preprocessor_params = {
        "idx_frequence_max": 1_000,
        "n_components": 10,
        "scale": True,
    }

    print_section("2. Train/test benchmark: FFT + PCA + Logistic Regression")
    logistic_result = train_test_benchmark(
        X_raw=X,
        y=y,
        preprocessor="fft_pca",
        preprocessor_params=preprocessor_params,
        classifier="logistic_regression",
        classifier_params={"C": 1.0},
        test_size=0.2,
        random_state=RANDOM_SEED,
    )
    print(f"Train accuracy: {logistic_result['train_metrics']['accuracy']:.3f}")
    print(f"Test accuracy : {logistic_result['test_metrics']['accuracy']:.3f}")
    print("Test confusion matrix:")
    print(logistic_result["test_metrics"]["confusion_matrix"])

    print_section("3. Quick GridSearch: preprocessing params + SVM params")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    svm_grid = run_grid_search(
        X_train=X,
        y_train=y,
        preprocessor="fft_pca",
        preprocessor_param_grid={
            "idx_frequence_max": [500, 1_000],
            "n_components": [5, 10],
            "scale": [True],
        },
        classifier=make_svm,
        classifier_param_grid={
            "C": [0.1, 1.0],
            "gamma": ["scale"],
        },
        cv=cv,
        n_jobs=1,
    )
    print(f"Best CV score: {svm_grid['best_score']:.3f}")
    print(f"Best preprocessing params: {svm_grid['best_params']['preprocessor']}")
    print(f"Best classifier params   : {svm_grid['best_params']['classifier']}")

    print_section("4. Manual LOO exercise: same separation, no sklearn Pipeline")
    loo_result = manual_loo_score(
        X_raw=X,
        y=y,
        preprocessor="fft_pca",
        preprocessor_params={
            "idx_frequence_max": 500,
            "n_components": 5,
            "scale": False,
        },
        classifier="logistic_regression",
        classifier_params={"C": 0.1, "max_iter": 500},
    )
    print(f"Manual LOO accuracy: {loo_result['score']:.3f}")

    print_section("5. Binary bagging meta-classifier")
    y_binary = np.where(y == 1, 1, 0)

    bagging_result = train_test_benchmark(
        X_raw=X,
        y=y_binary,
        preprocessor="fft_pca",
        preprocessor_params={
            "idx_frequence_max": 500,
            "n_components": 5,
            "scale": False,
        },
        classifier="bagging_tree",
        test_size=0.2,
        random_state=RANDOM_SEED,
    )
    print("Binary target: recule=1, others=0")
    print(f"Bagging test accuracy: {bagging_result['test_metrics']['accuracy']:.3f}")
    print("Bagging test confusion matrix:")
    print(bagging_result["test_metrics"]["confusion_matrix"])


if __name__ == "__main__":
    main()
