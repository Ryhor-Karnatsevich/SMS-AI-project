"""Compare one neural network trained on ten amounts of SMS data."""
from pathlib import Path
import csv
import argparse
import json
import warnings
from sklearn.exceptions import ConvergenceWarning

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
SEED = 42


def load_data(dataset):
    path = ROOT / "data" / "clean" / f"{dataset}.csv"
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    return [r["text"] for r in rows], [int(r["label"] == "spam") for r in rows]


def plot_accuracy(results, dataset):
    output = ROOT / "results" / dataset
    output.mkdir(exist_ok=True)
    percentages = [int(row["training_percent"]) for row in results]
    accuracies = [float(row["accuracy"]) * 100 for row in results]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(percentages, accuracies, marker="o", linewidth=2)
    for x, y in zip(percentages, accuracies):
        ax.annotate(f"{y:.2f}%", (x, y), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=8)
    ax.set(title=f"{dataset}: test accuracy",
           xlabel="Share of training pool used (%)",
           ylabel="Test accuracy (%)", xticks=list(range(10, 101, 10)))
    ax.set_ylim(max(0, min(accuracies) - 3), 100.5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = output / "accuracy.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"Chart saved to {path}")


def run_experiment(dataset):
    texts, labels = load_data(dataset)
    output = ROOT / "results" / dataset
    output.mkdir(parents=True, exist_ok=True)
    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.1, stratify=labels, random_state=SEED
    )
    print(f"Training pool: {len(x_train)}; fixed test set: {len(x_test)}")
    # Stratified, nested subsets: every smaller sample is inside the larger one.
    subsets = {1.0: (x_train, y_train)}
    previous_fraction = 1.0
    for percent in range(90, 0, -10):
        fraction = percent / 100
        previous_x, previous_y = subsets[previous_fraction]
        subset_x, _, subset_y, _ = train_test_split(
            previous_x, previous_y, train_size=fraction / previous_fraction,
            stratify=previous_y, random_state=SEED,
        )
        subsets[fraction] = (subset_x, subset_y)
        previous_fraction = fraction

    with (output / "split.json").open("w", encoding="utf-8") as file:
        json.dump({"dataset": dataset, "seed": SEED, "test_fraction": 0.1,
                   "train_messages": len(x_train), "test_messages": len(x_test),
                   "test_spam": sum(y_test), "test_ham": len(y_test) - sum(y_test)}, file, indent=2)
    results = []
    for percent in range(10, 101, 10):
        fraction = percent / 100
        subset_x, subset_y = subsets[fraction]
        # Fit TF-IDF only on this training subset, never on test messages.
        model = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=3000, sublinear_tf=True)),
            ("network", MLPClassifier(
                hidden_layer_sizes=(32,), activation="relu", solver="adam",
                max_iter=100, random_state=SEED,
            )),
        ])
        print(f"\nTraining {fraction:.0%}: {len(subset_x)} messages...", flush=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(subset_x, subset_y)
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
        predicted = model.predict(x_test)
        row = {
            "training_percent": percent,
            "iterations": model.named_steps["network"].n_iter_,
            "converged": converged,
            "training_messages": len(subset_x),
            "accuracy": accuracy_score(y_test, predicted),
            "spam_precision": precision_score(y_test, predicted, zero_division=0),
            "spam_recall": recall_score(y_test, predicted, zero_division=0),
            "spam_f1": f1_score(y_test, predicted, zero_division=0),
        }
        results.append(row)
        print(" | ".join(f"{key}: {value:.4f}" if isinstance(value, float)
                         else f"{key}: {value}" for key, value in row.items()))

    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {output / 'metrics.csv'}")
    plot_accuracy(results, dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["uci", "super_sms", "both"], default="both")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    datasets = ["uci", "super_sms"] if args.dataset == "both" else [args.dataset]
    if not args.plot_only:
        from prepare_data import main as prepare
        prepare()
    for dataset in datasets:
        if args.plot_only:
            with (ROOT / "results" / dataset / "metrics.csv").open(encoding="utf-8-sig") as file:
                plot_accuracy(list(csv.DictReader(file)), dataset)
        else:
            run_experiment(dataset)
