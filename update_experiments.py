from pathlib import Path
p = Path(__file__).with_name('main.py')
s = p.read_text(encoding='utf-8-sig')
s = s.replace('import sys', 'import argparse\nimport json\nimport warnings\nfrom sklearn.exceptions import ConvergenceWarning')
a = s.index('def load_data():')
b = s.index('\ndef plot_accuracy', a)
s = s[:a] + '''def load_data(dataset):
    path = ROOT / "data" / "clean" / f"{dataset}.csv"
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    return [r["text"] for r in rows], [int(r["label"] == "spam") for r in rows]

''' + s[b:]
s = s.replace('def plot_accuracy(results):\n    output = ROOT / "results"', 'def plot_accuracy(results, dataset):\n    output = ROOT / "results" / dataset')
s = s.replace('title="SMS classification: test accuracy"', 'title=f"{dataset}: test accuracy"')
s = s.replace('def main():\n    texts, labels = load_data()', 'def run_experiment(dataset):\n    texts, labels = load_data(dataset)\n    output = ROOT / "results" / dataset\n    output.mkdir(parents=True, exist_ok=True)')
s = s.replace('    results = []', '''    with (output / "split.json").open("w", encoding="utf-8") as file:
        json.dump({"dataset": dataset, "seed": SEED, "test_fraction": 0.1,
                   "train_messages": len(x_train), "test_messages": len(x_test),
                   "test_spam": sum(y_test), "test_ham": len(y_test) - sum(y_test)}, file, indent=2)
    results = []''')
s = s.replace('        model.fit(subset_x, subset_y)', '''        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(subset_x, subset_y)
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)''')
s = s.replace('            "training_percent": percent,', '            "training_percent": percent,\n            "iterations": model.named_steps["network"].n_iter_,\n            "converged": converged,')
s = s.replace('    output = ROOT / "results"\n    output.mkdir(exist_ok=True)\n    with (output / "metrics.csv")', '    with (output / "metrics.csv")')
s = s.replace('    plot_accuracy(results)', '    plot_accuracy(results, dataset)')
a = s.index('if __name__ == "__main__":')
s = s[:a] + '''if __name__ == "__main__":
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
'''
p.write_text(s, encoding='utf-8')
