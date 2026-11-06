"""Compare one neural network trained on ten amounts of SMS data."""
from pathlib import Path
import csv
import argparse
import json
import warnings
import hashlib
import joblib
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import confusion_matrix
from plots import plot_accuracy, plot_comparison, plot_model_comparison

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
SEED = 42


def load_data(dataset):
    path = ROOT / "data" / "clean" / f"{dataset}.csv"
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    return [r["text"] for r in rows], [int(r["label"] == "spam") for r in rows]


def run_experiment(dataset, classifier='mlp'):
    texts, labels = load_data(dataset)
    output = ROOT / "results" / dataset
    if classifier == 'logistic':
        output = output / 'logistic'
    output.mkdir(parents=True, exist_ok=True)
    model_folder = ROOT / 'models' / dataset
    model_folder.mkdir(parents=True, exist_ok=True)
    # Refuse incompatible caches after changes to data, architecture or dependencies.
    import sklearn
    import numpy
    import platform
    config = {'seed': SEED, 'test_size': 0.1, 'features': 3000,
              'hidden': [32], 'max_iter': 100, 'pipeline_version': 1,
              'sklearn': sklearn.__version__, 'numpy': numpy.__version__,
              'python': platform.python_version()}
    if classifier == 'logistic':
        config.pop('hidden')
        config.update(classifier='logistic', max_iter=1000, C=1.0, solver='lbfgs')
    data_bytes = (ROOT / 'data' / 'clean' / f'{dataset}.csv').read_bytes()
    signature = hashlib.sha256(data_bytes + json.dumps(config, sort_keys=True).encode()).hexdigest()
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
                   "data_sha256": hashlib.sha256(data_bytes).hexdigest(),
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
            ) if classifier == 'mlp' else LogisticRegression(
                C=1.0, solver='lbfgs', max_iter=1000, random_state=SEED)),
        ])
        print(f"\nModel {fraction:.0%}: {len(subset_x)} training messages...", flush=True)
        model_path = model_folder / f'{classifier}_{percent:03d}.joblib'
        if model_path.exists():
            # Load only locally generated, trusted joblib files.
            saved = joblib.load(model_path)
            if saved['signature'] != signature:
                raise RuntimeError(f'Cache does not match current data/config: {model_path}. '
                                   'Move the old model folder before training a new experiment.')
            model = saved['model']
            converged = saved['converged']
            print(f'Loaded saved model: {model_path}', flush=True)
        else:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(subset_x, subset_y)
            converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
            temporary = model_path.with_suffix('.tmp')
            joblib.dump({'model': model, 'signature': signature, 'config': config,
                         'converged': converged}, temporary)
            temporary.replace(model_path)
        predicted = model.predict(x_test)
        tn, fp, fn, tp = confusion_matrix(y_test, predicted, labels=[0, 1]).ravel()
        row = {
            "training_percent": percent,
            "iterations": int(numpy.asarray(model.named_steps["network"].n_iter_).max()),
            "converged": converged,
            "training_messages": len(subset_x),
            "accuracy": accuracy_score(y_test, predicted),
            "spam_precision": precision_score(y_test, predicted, zero_division=0),
            "spam_recall": recall_score(y_test, predicted, zero_division=0),
            "spam_f1": f1_score(y_test, predicted, zero_division=0),
            "true_negative": int(tn), "false_positive": int(fp),
            "false_negative": int(fn), "true_positive": int(tp),
        }
        results.append(row)
        # Keep completed points even if the next training is interrupted.
        with (output / 'metrics.csv').open('w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(results)
        print(" | ".join(f"{key}: {value:.4f}" if isinstance(value, float)
                         else f"{key}: {value}" for key, value in row.items()))

    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {output / 'metrics.csv'}")
    plot_accuracy(results, dataset, classifier)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["uci", "super_sms", "both"], default="both")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--train", action="store_true", help="Explicitly train missing models; reuse saved ones")
    parser.add_argument('--model', choices=['mlp', 'logistic', 'both'], default='mlp')
    args = parser.parse_args()
    if args.train and args.plot_only:
        parser.error('Choose either --train or --plot-only')
    datasets = ["uci", "super_sms"] if args.dataset == "both" else [args.dataset]
    classifiers = ['mlp', 'logistic'] if args.model == 'both' else [args.model]
    if args.train:
        from prepare_data import main as prepare
        prepare()
    for dataset in datasets:
        for classifier in classifiers:
            if not args.train:
                folder = ROOT / 'results' / dataset
                if classifier == 'logistic':
                    folder = folder / 'logistic'
                path = folder / 'metrics.csv'
                if not path.exists():
                    print(f'No saved results for {dataset}/{classifier}; skipped.')
                    continue
                with path.open(encoding='utf-8-sig') as file:
                    plot_accuracy(list(csv.DictReader(file)), dataset, classifier)
            else:
                run_experiment(dataset, classifier)
        plot_model_comparison(dataset)
    plot_comparison()

