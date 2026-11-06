"""Generate Polish charts from saved results. No training or preprocessing."""
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, MaxNLocator

ROOT = Path(__file__).resolve().parent
NAMES = {'uci': 'UCI SMS Spam Collection', 'super_sms': 'Super SMS'}
MODEL_NAMES = {'mlp': 'MLP', 'logistic': 'Regresja logistyczna',
               'naive_bayes': 'Multinomial Naive Bayes', 'svm': 'Linear SVM'}
COLORS = ['#2563eb', '#0d9488', '#d97706', '#9333ea']
MARKERS = ['o', 's', '^', 'D']
METRICS = [
    ('accuracy', 'Poprawne odpowiedzi', 'Accuracy (%)'),
    ('spam_f1', 'Jakość wykrywania spamu', 'F1 dla spamu (%)'),
    ('spam_recall', 'Ile spamu wykrywa model?', 'Recall dla spamu (%)'),
    ('spam_precision', 'Jak często alarm jest poprawny?', 'Precision dla spamu (%)'),
]


def result_folder(dataset, classifier):
    base = ROOT / 'results' / dataset
    return base if classifier == 'mlp' else base / classifier


def style(ax):
    ax.grid(axis='y', alpha=0.18)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=12)


def save(fig, ax, path, title, ylabel, ticks, values, note, zoom=True):
    minimum = min(values)
    padding = max(0.5, (100 - minimum) * 0.05)
    lower = max(0, math.floor((minimum - padding) * 2) / 2) if zoom else 0
    ax.set_ylim(lower, 100)
    ax.set_xlim(min(ticks) - 3, max(ticks) + 3)
    ax.set_xticks(ticks, [f'{p}%' for p in ticks])
    ax.set_xlabel('Wykorzystana część puli treningowej (%)', fontsize=13, labelpad=12)
    ax.set_ylabel(ylabel, fontsize=13, labelpad=12)
    positions = MaxNLocator(nbins=6).tick_values(lower, 100)
    positions = [v for v in positions if lower <= v < 100]
    # Avoid a nearly overlapping last tick just below the fixed 100% endpoint.
    if positions and 100 - positions[-1] < (100 - lower) * 0.07:
        positions.pop()
    ax.set_yticks(positions + [100])
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=1))
    ax.set_title(title, fontsize=19, loc='left', pad=22)
    ax.legend(loc='best', fontsize=11, frameon=True, facecolor='white', framealpha=0.95)
    fig.subplots_adjust(left=0.12, right=0.96, top=0.80, bottom=0.25)
    fig.text(0.12, 0.12, f'Oś Y: {lower:g}–100%. Jeden trening na punkt; brak oceny zmienności.',
             fontsize=10, color='#475569')
    fig.text(0.12, 0.04, note, fontsize=9, color='#64748b')
    fig.savefig(path, dpi=180, facecolor='white')
    plt.close(fig)
    print(f'CREATED: {path.resolve()}')


def plot_accuracy(results, dataset, classifier='mlp'):
    """Keep the existing per-model chart filenames."""
    rows = sorted(results, key=lambda r: int(r['training_percent']))
    if not rows:
        return
    folder = result_folder(dataset, classifier)
    folder.mkdir(parents=True, exist_ok=True)
    x = [int(r['training_percent']) for r in rows]
    name = MODEL_NAMES[classifier]
    index = list(MODEL_NAMES).index(classifier)
    color = COLORS[index]
    for field, title, ylabel in METRICS:
        y = [float(r[field]) * 100 for r in rows]
        fig, ax = plt.subplots(figsize=(11, 6.5))
        style(ax)
        ax.plot(x, y, marker=MARKERS[index], linewidth=2.6, color=color, label=name)
        filename = 'accuracy.png' if field == 'accuracy' else f'{field}.png'
        save(fig, ax, folder / filename, f'{NAMES[dataset]} | {name}\n{title}', ylabel,
             x, y, 'Procent dotyczy puli treningowej (90% danych), nie całego zbioru.')

    fig, ax = plt.subplots(figsize=(11, 6.5))
    style(ax)
    values = []
    for (field, title, _), color in zip(METRICS[1:], COLORS):
        y = [float(r[field]) * 100 for r in rows]
        values.extend(y)
        ax.plot(x, y, marker='o', linewidth=2, label=field.replace('spam_', '').upper(), color=color)
    save(fig, ax, folder / 'spam_quality.png', f'{NAMES[dataset]} | {name}\nWykrywanie spamu',
         'Wartość miary (%)', x, values, 'F1: balans precision i recall. Test nie jest uzywany do uczenia.')

    fig, ax = plt.subplots(figsize=(11, 6.5))
    style(ax)
    errors = [(1 - float(r['accuracy'])) * 100 for r in rows]
    bars = ax.bar(x, errors, color=color, width=6)
    ax.bar_label(bars, labels=[f'{v:.2f}' for v in errors], padding=5)
    ax.set(xticks=x, xlabel='Wykorzystana część puli treningowej (%)',
           ylabel='Błędy na 100 testowych SMS', ylim=(0, max(1, max(errors) * 1.3)))
    ax.set_title(f'{NAMES[dataset]} | {name}\nCzęstość błędów — mniej znaczy lepiej', fontsize=18, loc='left')
    fig.tight_layout()
    fig.savefig(folder / 'errors.png', dpi=180)
    plt.close(fig)

    if all('false_positive' in r and 'false_negative' in r for r in rows):
        fig, ax = plt.subplots(figsize=(11, 6.5))
        style(ax)
        ax.bar([p - 1.7 for p in x], [int(r['false_positive']) for r in rows], width=3.4,
               color=COLORS[2], label='Fałszywy alarm: ham jako spam')
        ax.bar([p + 1.7 for p in x], [int(r['false_negative']) for r in rows], width=3.4,
               color=COLORS[0], label='Przeoczony spam: spam jako ham')
        ax.set(xticks=x, xlabel='Wykorzystana część puli treningowej (%)', ylabel='Liczba testowych SMS')
        ax.set_title(f'{NAMES[dataset]} | {name}\nRodzaje błędów', fontsize=18, loc='left')
        ax.legend()
        fig.tight_layout()
        fig.savefig(folder / 'error_types.png', dpi=180)
        plt.close(fig)


def plot_model_comparison(dataset):
    """Compare only compatible results within one dataset. Preserve old chart paths."""
    available = {}
    splits = {}
    for classifier in MODEL_NAMES:
        folder = result_folder(dataset, classifier)
        if not ((folder / 'metrics.csv').exists() and (folder / 'split.json').exists()):
            print(f'{dataset}/{classifier}: no saved results; omitted from chart.')
            continue
        with (folder / 'metrics.csv').open(encoding='utf-8-sig', newline='') as f:
            table = {int(r['training_percent']): r for r in csv.DictReader(f)}
        if table:
            available[classifier] = table
            splits[classifier] = json.loads((folder / 'split.json').read_text(encoding='utf-8'))
    if not available:
        print(f'{dataset}: no results to plot.')
        return
    reference = next(iter(splits.values()))
    keys = ['seed', 'test_fraction', 'train_messages', 'test_messages', 'test_spam', 'test_ham']
    for classifier, split in splits.items():
        if any(k not in reference or k not in split or reference[k] != split[k] for k in keys):
            raise ValueError(f'{dataset}/{classifier}: incompatible split metadata')
    hashes = {s['data_sha256'] for s in splits.values() if s.get('data_sha256')}
    if len(hashes) > 1:
        raise ValueError(f'{dataset}: dataset checksums differ')
    legacy = any(not s.get('data_sha256') for s in splits.values())
    sizes = {}
    for classifier, table in available.items():
        for percent, row in table.items():
            n = int(row['training_messages'])
            if percent in sizes and sizes[percent] != n:
                raise ValueError(f'{dataset}: training sample sizes differ at {percent}%')
            sizes[percent] = n
    ticks = sorted(sizes)
    names = ', '.join(MODEL_NAMES[c] for c in available)
    note = f'Test: {reference["test_messages"]} SMS. Dostępne modele: {len(available)}/4.\n'
    note += ('Starszy wynik bez sumy kontrolnej: identyczność danych nie jest w pełni potwierdzona.'
             if legacy else 'Sumy kontrolne danych i parametry podzialu są zgodne.')
    for field, title, ylabel in METRICS:
        fig, ax = plt.subplots(figsize=(11, 6.5))
        style(ax)
        values = []
        for classifier, color, marker in zip(MODEL_NAMES, COLORS, MARKERS):
            if classifier not in available:
                continue
            table = available[classifier]
            x = sorted(table)
            y = [float(table[p][field]) * 100 for p in x]
            values.extend(y)
            ax.plot(x, y, marker=marker, markersize=7, linewidth=2.4,
                    label=MODEL_NAMES[classifier], color=color)
        path = ROOT / 'results' / dataset / f'compare_{field}.png'
        save(fig, ax, path, f'{NAMES[dataset]}\n{title}', ylabel, ticks, values, note)
    print(f'{dataset}: {names}')


if __name__ == '__main__':
    for dataset in NAMES:
        for classifier in MODEL_NAMES:
            path = result_folder(dataset, classifier) / 'metrics.csv'
            if path.exists():
                with path.open(encoding='utf-8-sig', newline='') as f:
                    plot_accuracy(list(csv.DictReader(f)), dataset, classifier)
        plot_model_comparison(dataset)

