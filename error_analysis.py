"""Analyze saved SMS classifiers on the fixed test split. Never fit models."""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from plots import MODEL_NAMES, NAMES
from error_report import build_report, refresh_readme

ROOT = Path(__file__).resolve().parent
URL = re.compile(r'https?://|www\.', re.I)


def write_csv(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def rate(numerator, denominator):
    return numerator / denominator if denominator else None


def verify_saved_metrics(dataset, classifier, percent, counts):
    """Check aggregate predictions against the original experiment, without fitting."""
    folder = ROOT / 'results' / dataset
    if classifier != 'mlp':
        folder = folder / classifier
    path = folder / 'metrics.csv'
    if not path.exists():
        raise ValueError(f'{path}: missing baseline metrics for cross-version verification')
    with path.open(encoding='utf-8-sig', newline='') as file:
        row = next((r for r in csv.DictReader(file) if int(r['training_percent']) == percent), None)
    if row is None:
        raise ValueError(f'{path}: no baseline metrics for {percent}% training')
    tn, fp, fn, tp = counts
    observed = {'true_negative': tn, 'false_positive': fp, 'false_negative': fn, 'true_positive': tp,
                'accuracy': (tn + tp) / (tn + fp + fn + tp),
                'spam_precision': tp / (tp + fp) if tp + fp else 0,
                'spam_recall': tp / (tp + fn) if tp + fn else 0,
                'spam_f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0}
    checked = []
    for key, value in observed.items():
        if row.get(key, '') != '':
            if not np.isclose(float(row[key]), value, rtol=0, atol=1e-10):
                raise ValueError(f'{dataset}/{classifier}: predictions differ from saved {key}: '
                                 f'original={row[key]}, current={value}. '
                                 'Use the original environment; no training was performed.')
            checked.append(key)
    if not all(k in checked for k in ('accuracy', 'spam_precision', 'spam_recall', 'spam_f1')):
        raise ValueError(f'{path}: insufficient metrics to verify cross-version predictions')
    return checked


def analyze(dataset, percent):
    source = ROOT / 'data' / 'clean' / f'{dataset}.csv'
    if not source.exists():
        raise FileNotFoundError(f'Missing cleaned data: {source}. No preprocessing is run automatically.')
    data_bytes = source.read_bytes()
    with source.open(encoding='utf-8-sig', newline='') as f:
        data = list(csv.DictReader(f))
    texts = [r['text'] for r in data]
    labels = np.array([{'ham': 0, 'spam': 1}[r['label']] for r in data])
    # Same input order and stratification as main.py.
    _, indices = train_test_split(np.arange(len(texts)), test_size=0.1, stratify=labels, random_state=42)
    x_test = [texts[i] for i in indices]
    y_test = labels[indices]
    predictions, summaries, segments, records, source_signatures = {}, [], [], [], {}
    missing = []
    compatibility = {}
    for classifier in MODEL_NAMES:
        path = ROOT / 'models' / dataset / f'{classifier}_{percent:03d}.joblib'
        if not path.exists():
            missing.append(classifier)
            print(f'{dataset}/{classifier}: saved model missing; skipped. Metrics alone cannot reconstruct SMS errors.')
            continue
        # Only load trusted local files produced by main.py.
        saved = joblib.load(path)
        config = saved['config']
        signature = hashlib.sha256(data_bytes + json.dumps(config, sort_keys=True).encode()).hexdigest()
        if saved.get('signature') != signature:
            raise ValueError(f'{path}: model/data signature mismatch. Analysis stopped; no training performed.')
        if config.get('seed') != 42 or config.get('test_size') != 0.1:
            raise ValueError(f'{path}: incompatible test split')
        versions = {'sklearn': {'saved': config.get('sklearn'), 'current': sklearn.__version__},
                    'numpy': {'saved': config.get('numpy'), 'current': np.__version__}}
        version_mismatch = any(v['saved'] != v['current'] for v in versions.values())
        if version_mismatch:
            print(f'{dataset}/{classifier}: library versions differ: {versions}. '
                  'Checking predictions against saved metrics; no training.')
        model = saved['model']
        predicted = np.asarray(model.predict(x_test), dtype=int)
        predictions[classifier] = predicted
        source_signatures[classifier] = saved['signature']
        if hasattr(model, 'predict_proba'):
            positive_column = list(model.classes_).index(1)
            scores = model.predict_proba(x_test)[:, positive_column]
            score_kind = 'spam_probability'
        else:
            scores = model.decision_function(x_test)
            score_kind = 'svm_margin_not_probability'
        tn, fp, fn, tp = (int(v) for v in confusion_matrix(y_test, predicted, labels=[0, 1]).ravel())
        verified = verify_saved_metrics(dataset, classifier, percent, (tn, fp, fn, tp)) if version_mismatch else []
        compatibility[classifier] = {'versions': versions, 'version_mismatch': version_mismatch,
                                     'matching_baseline_metrics': verified}
        summaries.append({'model': classifier, 'test_messages': len(y_test),
                          'true_negative': tn, 'false_positive': fp, 'false_negative': fn, 'true_positive': tp,
                          'false_positive_rate': rate(fp, tn + fp), 'false_negative_rate': rate(fn, tp + fn)})
        for position, (index, text, actual, pred, score) in enumerate(zip(indices, x_test, y_test, predicted, scores)):
            records.append({'model': classifier, 'test_position': position,
                            'message_id': hashlib.sha256(text.encode('utf-8')).hexdigest(),
                            'clean_row': int(index), 'text': text,
                            'actual': 'spam' if actual else 'ham', 'predicted': 'spam' if pred else 'ham',
                            'error_type': ('false_positive' if pred else 'false_negative') if actual != pred else 'correct',
                            'score': float(score), 'score_kind': score_kind,
                            'characters': len(text), 'has_url': bool(URL.search(text))})
        masks = {'short': np.array([len(t) <= 40 for t in x_test]),
                 'medium': np.array([40 < len(t) <= 160 for t in x_test]),
                 'long': np.array([len(t) > 160 for t in x_test]),
                 'with_url': np.array([bool(URL.search(t)) for t in x_test])}
        masks['without_url'] = ~masks['with_url']
        for name, mask in masks.items():
            support = int(mask.sum())
            errors = int(np.sum((predicted != y_test) & mask))
            ham = int(np.sum((y_test == 0) & mask))
            spam = int(np.sum((y_test == 1) & mask))
            segment_fp = int(np.sum((predicted == 1) & (y_test == 0) & mask))
            segment_fn = int(np.sum((predicted == 0) & (y_test == 1) & mask))
            segments.append({'model': classifier, 'segment': name, 'messages': support,
                             'ham': ham, 'spam': spam, 'errors': errors, 'error_rate': rate(errors, support),
                             'false_positive': segment_fp, 'false_negative': segment_fn,
                             'false_positive_rate': rate(segment_fp, ham), 'false_negative_rate': rate(segment_fn, spam)})
    if not predictions:
        print(f'{dataset}: no saved models available; no report generated.')
        return
    output = ROOT / 'results' / dataset / 'error_analysis' / str(percent)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'predictions.csv', records, list(records[0]))
    wrong = [r for r in records if r['error_type'] != 'correct']
    write_csv(output / 'errors.csv', wrong, list(records[0]))
    write_csv(output / 'summary.csv', summaries, list(summaries[0]))
    write_csv(output / 'segments.csv', segments, list(segments[0]))
    shared = []
    for j, index in enumerate(indices):
        failed = [m for m, pred in predictions.items() if pred[j] != y_test[j]]
        if failed:
            shared.append({'message_id': hashlib.sha256(x_test[j].encode('utf-8')).hexdigest(),
                           'text': x_test[j], 'actual': 'spam' if y_test[j] else 'ham',
                           'wrong_models': ', '.join(failed), 'wrong_model_count': len(failed),
                           'available_model_count': len(predictions)})
    write_csv(output / 'shared_errors.csv', shared,
              ['message_id', 'text', 'actual', 'wrong_models', 'wrong_model_count', 'available_model_count'])
    manifest = {'dataset': dataset, 'training_percent': percent, 'test_messages': len(y_test),
                'test_ham': int(np.sum(y_test == 0)), 'test_spam': int(np.sum(y_test == 1)),
                'data_sha256': hashlib.sha256(data_bytes).hexdigest(),
                'test_indices_sha256': hashlib.sha256(json.dumps(indices.tolist()).encode()).hexdigest(),
                'model_signatures': source_signatures, 'missing_models': missing,
                'compatibility': compatibility,
                'fit_performed': False, 'segments': 'length <=40, 41-160, >160; URL regex https?:// or www.'}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze saved models; no training is performed.')
    parser.add_argument('--dataset', choices=['uci', 'super_sms', 'both'], default='both')
    parser.add_argument('--training-percent', type=int, nargs='+', choices=list(range(10, 101, 10)), default=[10, 100])
    args = parser.parse_args()
    for dataset in (NAMES if args.dataset == 'both' else [args.dataset]):
        outputs = {percent: analyze(dataset, percent) for percent in sorted(set(args.training_percent))}
        build_report(dataset, {p: path for p, path in outputs.items() if path is not None})
    refresh_readme()


