"""SVM classifier definition."""

from sklearn.svm import SVC


def make_classifier(C=1.0, gamma="scale", kernel="rbf", **params):
    return SVC(C=C, gamma=gamma, kernel=kernel, **params)


def default_param_grid():
    return {
        "C": [0.1, 1.0, 10.0, 100.0],
        "gamma": [0.001, 0.01, 0.1, "scale"],
    }
