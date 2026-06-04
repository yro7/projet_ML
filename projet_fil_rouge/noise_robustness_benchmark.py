"""Benchmark the noise robustness of classifiers.

This script injects white Gaussian noise into the raw audio signals at different
SNR levels, extracts MFCC Summary features, runs Leave-One-Out cross-validation,
saves the results, and plots a comparison graph.

Run from project root:
    python3 projet_fil_rouge/noise_robustness_benchmark.py
"""
import sys
import time
from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier, BaggingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, LeaveOneOut

# Setup path to import from project
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from projet_fil_rouge.config import RANDOM_SEED, seed_everything
from projet_fil_rouge.data import load_dataset
from projet_fil_rouge.utils.preprocessings import preprocess_mfcc_summary

seed_everything(RANDOM_SEED)

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def add_noise_to_dataset(X, snr_db, random_state=RANDOM_SEED):
    """Add White Gaussian Noise to raw audio matrix X based on target SNR in dB."""
    if snr_db is None or np.isinf(snr_db):
        return X.copy()

    np.random.seed(random_state)
    X_noisy = np.zeros_like(X)
    for i in range(X.shape[0]):
        x = X[i, :]
        power = np.mean(x ** 2)
        if power <= 0:
            X_noisy[i, :] = x
            continue
        noise_power = power / (10 ** (snr_db / 10.0))
        noise = np.random.normal(0, np.sqrt(noise_power), size=x.shape)
        X_noisy[i, :] = x + noise
    return X_noisy


def main():
    print("=== Noise Robustness Benchmark ===")
    print("Loading raw dataset...")
    X, y, genres, fs = load_dataset()
    print(f"Loaded {X.shape[0]} audio samples (length: {X.shape[1]}, fs: {fs}Hz)")

    # Define noise levels (SNR in dB)
    noise_levels = {
        "Clean": None,
        "Low (20dB)": 20.0,
        "Medium (10dB)": 10.0,
        "High (0dB)": 0.0
    }

    # Define classifiers to evaluate
    classifiers = {
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_SEED),
        "SVM (RBF)": SVC(C=10.0, kernel="rbf", gamma="scale", random_state=RANDOM_SEED),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED),
        "Bagging Tree": BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=5, random_state=RANDOM_SEED),
            n_estimators=50,
            random_state=RANDOM_SEED
        ),
        "AdaBoost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=2, random_state=RANDOM_SEED),
            n_estimators=100,
            learning_rate=0.1,
            random_state=RANDOM_SEED
        ),
        "Neural Network": MLPClassifier(
            hidden_layer_sizes=(32,),
            alpha=0.001,
            max_iter=1000,
            random_state=RANDOM_SEED
        )
    }

    results = {clf_name: [] for clf_name in classifiers}
    csv_rows = []

    # Iterate over noise levels
    for level_name, snr_val in noise_levels.items():
        print(f"\nEvaluating noise level: {level_name}...")
        # 1. Add noise to raw signal
        X_noisy = add_noise_to_dataset(X, snr_val)
        
        # 2. Extract MFCC Summary features (13 mfcc, mean + std)
        print("  Extracting MFCC Summary features...")
        X_features = preprocess_mfcc_summary(X_noisy, sr=fs, n_mfcc=13)
        
        # 3. Evaluate each classifier
        for clf_name, clf in classifiers.items():
            t0 = time.perf_counter()
            
            # Setup a standard scaling pipeline
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", clf)
            ])
            
            # Leave-One-Out CV
            scores = cross_val_score(pipeline, X_features, y, cv=LeaveOneOut(), n_jobs=-1)
            mean_acc = np.mean(scores)
            duration = time.perf_counter() - t0
            
            results[clf_name].append(mean_acc)
            print(f"  - {clf_name}: Accuracy = {mean_acc:.2%} (took {duration:.2f}s)")
            
            csv_rows.append({
                "Noise Level": level_name,
                "SNR (dB)": str(snr_val) if snr_val is not None else "inf",
                "Classifier": clf_name,
                "Accuracy": f"{mean_acc:.4f}",
                "Duration (s)": f"{duration:.3f}"
            })

    # Save results to CSV
    csv_path = RESULTS_DIR / "noise_robustness.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Noise Level", "SNR (dB)", "Classifier", "Accuracy", "Duration (s)"])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nSaved CSV results to: {csv_path}")

    # Plot results
    print("\nGenerating graph...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Modern color palette
    colors = {
        "Logistic Regression": "#34495e",  # Slate Blue
        "SVM (RBF)": "#e74c3c",            # Coral Red
        "Random Forest": "#2ecc71",        # Emerald Green
        "Bagging Tree": "#3498db",         # Sky Blue
        "AdaBoost": "#e67e22",             # Orange
        "Neural Network": "#9b59b6"        # Amethyst Purple
    }
    
    markers = {
        "Logistic Regression": "o",
        "SVM (RBF)": "s",
        "Random Forest": "^",
        "Bagging Tree": "D",
        "AdaBoost": "v",
        "Neural Network": "p"
    }

    x_labels = list(noise_levels.keys())
    x_indices = np.arange(len(x_labels))

    for clf_name, accs in results.items():
        ax.plot(
            x_indices,
            accs,
            marker=markers[clf_name],
            color=colors[clf_name],
            label=clf_name,
            linewidth=2,
            markersize=8
        )
        
        # Add labels to the endpoints
        ax.text(
            x_indices[-1] + 0.05,
            accs[-1],
            f"{accs[-1]:.1%}",
            color=colors[clf_name],
            va="center",
            fontweight="bold",
            fontsize=9
        )

    ax.set_xticks(x_indices)
    ax.set_xticklabels(x_labels, fontsize=11)
    ax.set_xlabel("Niveau de Bruit (SNR)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy (LOO CV)", fontsize=12, fontweight="bold")
    ax.set_title("Résilience au Bruit des Classifieurs (MFCC Summary)", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylim(0.3, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower left", fontsize=10, framealpha=0.9)
    
    # Make layout adjustments to fit text on the right
    ax.set_xlim(-0.2, len(x_labels) - 0.7)
    plt.tight_layout()

    # Save figures
    pdf_path = FIGURES_DIR / "noise_robustness.pdf"
    png_path = FIGURES_DIR / "noise_robustness.png"
    fig.savefig(pdf_path, dpi=150, bbox_inches="tight")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved graph: {pdf_path}")
    print(f"Saved graph: {png_path}")
    print("Done!")


if __name__ == "__main__":
    main()
