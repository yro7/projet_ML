"""Benchmark the noise robustness of classifiers with multiple realizations.

This script injects white Gaussian noise into the raw audio signals at different
SNR levels over 10 independent realizations, extracts MFCC Summary features,
runs Leave-One-Out cross-validation, saves the results, and plots a comparison
graph with standard deviation shaded areas.

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
    print("=== Robust Noise Robustness Benchmark ===")
    print("Loading raw dataset...")
    X, y, genres, fs = load_dataset()
    print(f"Loaded {X.shape[0]} audio samples (length: {X.shape[1]}, fs: {fs}Hz)")

    # Define noise levels (SNR in dB)
    snr_levels = {
        "Clean": None,
        "25dB": 25.0,
        "20dB": 20.0,
        "15dB": 15.0,
        "10dB": 10.0,
        "5dB": 5.0,
        "0dB": 0.0
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

    n_realizations = 10
    results_mean = {clf_name: [] for clf_name in classifiers}
    results_std = {clf_name: [] for clf_name in classifiers}
    csv_rows = []

    # Iterate over SNR levels
    for level_name, snr_val in snr_levels.items():
        print(f"\nEvaluating SNR level: {level_name}...")
        
        if snr_val is None:
            # Clean: no noise realization needed
            print("  Running clean baseline...")
            X_features = preprocess_mfcc_summary(X, sr=fs, n_mfcc=13)
            for clf_name, clf in classifiers.items():
                t0 = time.perf_counter()
                pipeline = Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", clf)
                ])
                scores = cross_val_score(pipeline, X_features, y, cv=LeaveOneOut(), n_jobs=-1)
                mean_acc = np.mean(scores)
                duration = time.perf_counter() - t0
                
                results_mean[clf_name].append(mean_acc)
                results_std[clf_name].append(0.0)
                print(f"    - {clf_name}: Accuracy = {mean_acc:.2%} (took {duration:.2f}s)")
                
                csv_rows.append({
                    "Noise Level": level_name,
                    "SNR (dB)": "inf",
                    "Classifier": clf_name,
                    "Mean Accuracy": f"{mean_acc:.4f}",
                    "Std Accuracy": "0.0000",
                    "Realizations": "1"
                })
        else:
            # Noisy: run multiple realizations
            all_scores = {clf_name: [] for clf_name in classifiers}
            t0 = time.perf_counter()
            
            for r in range(n_realizations):
                # Generate noise with seed dependent on realization
                seed = RANDOM_SEED + r
                X_noisy = add_noise_to_dataset(X, snr_val, random_state=seed)
                X_features = preprocess_mfcc_summary(X_noisy, sr=fs, n_mfcc=13)
                
                for clf_name, clf in classifiers.items():
                    pipeline = Pipeline([
                        ("scaler", StandardScaler()),
                        ("clf", clf)
                    ])
                    scores = cross_val_score(pipeline, X_features, y, cv=LeaveOneOut(), n_jobs=-1)
                    all_scores[clf_name].append(np.mean(scores))
            
            duration = time.perf_counter() - t0
            print(f"  Completed {n_realizations} realizations in {duration:.2f}s")
            
            for clf_name in classifiers:
                mean_acc = np.mean(all_scores[clf_name])
                std_acc = np.std(all_scores[clf_name])
                results_mean[clf_name].append(mean_acc)
                results_std[clf_name].append(std_acc)
                print(f"    - {clf_name}: Mean = {mean_acc:.2%} (std = {std_acc:.2%})")
                
                csv_rows.append({
                    "Noise Level": level_name,
                    "SNR (dB)": str(snr_val),
                    "Classifier": clf_name,
                    "Mean Accuracy": f"{mean_acc:.4f}",
                    "Std Accuracy": f"{std_acc:.4f}",
                    "Realizations": str(n_realizations)
                })

    # Save results to CSV
    csv_path = RESULTS_DIR / "noise_robustness.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Noise Level", "SNR (dB)", "Classifier", "Mean Accuracy", "Std Accuracy", "Realizations"])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nSaved CSV results to: {csv_path}")

    # Plot results
    print("\nGenerating graph with standard deviation bands...")
    fig, ax = plt.subplots(figsize=(11, 7))
    
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

    x_labels = list(snr_levels.keys())
    x_indices = np.arange(len(x_labels))

    for clf_name in classifiers:
        means = np.array(results_mean[clf_name])
        stds = np.array(results_std[clf_name])
        
        # Plot mean line
        ax.plot(
            x_indices,
            means,
            marker=markers[clf_name],
            color=colors[clf_name],
            label=clf_name,
            linewidth=2,
            markersize=7
        )
        
        # Add shaded area for standard deviation
        ax.fill_between(
            x_indices,
            means - stds,
            means + stds,
            color=colors[clf_name],
            alpha=0.15
        )
        
        # Add label to the final point
        final_val = means[-1]
        final_std = stds[-1]
        ax.text(
            x_indices[-1] + 0.08,
            final_val,
            f"{final_val:.1%} ± {final_std:.1%}" if final_std > 0 else f"{final_val:.1%}",
            color=colors[clf_name],
            va="center",
            fontweight="bold",
            fontsize=8
        )

    ax.set_xticks(x_indices)
    ax.set_xticklabels(x_labels, fontsize=11)
    ax.set_xlabel("Niveau de Bruit (SNR)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy (LOO CV)", fontsize=12, fontweight="bold")
    ax.set_title("Résilience au Bruit des Classifieurs avec Intervalles de Confiance (MFCC Summary)", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylim(0.25, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower left", fontsize=10, framealpha=0.9)
    
    # Adjust X limits to make room for labels on the right
    ax.set_xlim(-0.25, len(x_labels) - 0.5)
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
