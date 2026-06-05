"""Benchmark classifiers on the 13-class dataset (sons_ia).

This script loads the 130 personal audio recordings from 13 classes,
extracts MFCC Summary features, runs Leave-One-Out cross-validation,
saves the results, and plots a comparison bar chart.

Run from project root:
    python3 projet_fil_rouge/multi_class_benchmark.py
"""
import sys
import time
from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import librosa

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
from projet_fil_rouge.data import energy_trim
from projet_fil_rouge.utils.preprocessings import preprocess_mfcc_summary

seed_everything(RANDOM_SEED)

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_13_class_dataset():
    """Load WAV files from sons_perso/sons_ia and return stacked signals and labels."""
    dataset_dir = Path(__file__).resolve().parent / "sons_perso" / "sons_ia"
    wav_paths = sorted(dataset_dir.glob("**/*.wav"))
    
    # Class names are the parent subdirectory names
    words = sorted(list(set(p.parent.name for p in wav_paths)))
    
    records = []
    labels = []
    for p in wav_paths:
        label = words.index(p.parent.name)
        # Load at 16 kHz
        y, sr = librosa.load(str(p), sr=16000)
        records.append(y)
        labels.append(label)
        
    # Trim to smallest sample length
    min_len = min(len(r) for r in records)
    X = np.vstack([energy_trim(r, min_len) for r in records])
    y = np.array(labels)
    
    return X, y, words


def main():
    print("=== 13-Class Dataset Benchmark ===")
    print("Loading sons_ia dataset...")
    X, y, words = load_13_class_dataset()
    print(f"Loaded {X.shape[0]} audio samples across {len(words)} classes.")
    print(f"Classes: {words}")

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

    # Extract MFCC Summary features at 16 kHz
    print("Extracting MFCC Summary features...")
    X_features = preprocess_mfcc_summary(X, sr=16000, n_mfcc=13)
    print(f"Features shape: {X_features.shape}")

    results = []
    csv_rows = []

    # Evaluate each classifier
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
        std_acc = np.std(scores)
        duration = time.perf_counter() - t0
        
        results.append((clf_name, mean_acc))
        print(f"  - {clf_name}: Accuracy = {mean_acc:.2%} (took {duration:.2f}s)")
        
        csv_rows.append({
            "Classifier": clf_name,
            "Accuracy": f"{mean_acc:.4f}",
            "Std Accuracy": f"{std_acc:.4f}",
            "Duration (s)": f"{duration:.3f}"
        })

    # Save results to CSV
    csv_path = RESULTS_DIR / "multi_class_benchmark.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Classifier", "Accuracy", "Std Accuracy", "Duration (s)"])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nSaved CSV results to: {csv_path}")

    # Plot results
    print("\nGenerating graph...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort results for nicer plotting
    results_sorted = sorted(results, key=lambda x: x[1], reverse=True)
    clf_names = [r[0] for r in results_sorted]
    clf_accs = [r[1] for r in results_sorted]
    
    # Modern color palette
    colors = ["#2ecc71", "#3498db", "#9b59b6", "#34495e", "#e67e22", "#e74c3c"]
    
    bars = ax.bar(clf_names, clf_accs, color=colors[:len(clf_names)], width=0.5, edgecolor="black", linewidth=0.5)
    
    for bar, acc in zip(bars, clf_accs):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.01,
            f"{acc:.2%}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10
        )

    ax.set_ylabel("Accuracy (LOO CV)", fontsize=12, fontweight="bold")
    ax.set_title("Reconnaissance de Commandes Vocales (13 Classes × 10 échantillons)", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    
    plt.tight_layout()

    # Save figures
    pdf_path = FIGURES_DIR / "multi_class_benchmark.pdf"
    png_path = FIGURES_DIR / "multi_class_benchmark.png"
    fig.savefig(pdf_path, dpi=150, bbox_inches="tight")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved graph: {pdf_path}")
    print(f"Saved graph: {png_path}")
    print("Done!")


if __name__ == "__main__":
    main()
