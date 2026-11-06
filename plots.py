"""Create presentation charts from saved metrics; never train models."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parent
NAMES = {'uci': 'UCI — исходный набор', 'super_sms': 'Super SMS — новый набор'}
COLORS = ['#2563eb', '#0d9488', '#d97706']


def style(ax):
    ax.grid(axis='y', alpha=0.18)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)


def save(fig, path, note):
    fig.text(0.08, 0.025, note, fontsize=9, color='#475569')
    fig.tight_layout(rect=(0, 0.10, 1, 0.98))
    fig.savefig(path, dpi=180, facecolor='white')
    plt.close(fig)
    print(f'Chart saved: {path}')


def plot_accuracy(results, dataset, classifier='mlp'):
    rows = sorted(results, key=lambda r: int(r['training_percent']))
    folder = ROOT / 'results' / dataset
    if classifier == 'logistic':
        folder = folder / 'logistic'
    model_name = 'MLP' if classifier == 'mlp' else 'Логистическая регрессия'
    folder.mkdir(parents=True, exist_ok=True)
    split_path = folder / 'split.json'
    split = json.loads(split_path.read_text(encoding='utf-8')) if split_path.exists() else {}
    test_n = split.get('test_messages', '?')
    x = list(range(len(rows)))
    ticks = [f"{int(r['training_percent'])}%\n{int(r['training_messages']):,} SMS".replace(',', ' ') for r in rows]
    accuracy = [float(r['accuracy']) * 100 for r in rows]
    note = (f'{model_name}. Общий тест: {test_n} SMS. Проценты по X — доля обучающей части (90% данных).\n'
            'Одна точка — один запуск; разброс результатов не оценивался.')
    # A full scale and a labelled zoom show both magnitude and small changes.
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'{NAMES[dataset]}: как объём обучения влияет на точность', fontsize=16)
    for ax in axes:
        style(ax)
        ax.plot(x, accuracy, color=COLORS[0], marker='o', linewidth=2)
        ax.set_xticks(x, ticks, fontsize=8)
        ax.set_xlabel('Объём данных, использованных для обучения')
        ax.set_ylabel('Правильно распознанные SMS / все тестовые SMS')
        ax.yaxis.set_major_formatter(PercentFormatter(100))
    axes[0].set(title='Полная шкала: доля правильных ответов', ylim=(0, 100))
    low = max(0, min(accuracy) - 1.5)
    axes[1].set(title='Увеличенный участок: небольшие изменения', ylim=(low, 100.5))
    for i, value in enumerate(accuracy):
        axes[1].annotate(f'{value:.2f}%', (i, value), xytext=(0, 8),
                         textcoords='offset points', ha='center', fontsize=8)
    if split.get('test_messages'):
        baseline = max(split['test_ham'], split['test_spam']) / split['test_messages'] * 100
        axes[0].axhline(baseline, color='#64748b', linestyle='--',
                       label=f'Всегда выбирать больший класс: {baseline:.1f}%')
        axes[0].legend(loc='lower right', fontsize=8)
    save(fig, folder / 'accuracy.png', note)

    fig, ax = plt.subplots(figsize=(13, 6))
    style(ax)
    for field, label, color in zip(
        ['spam_precision', 'spam_recall', 'spam_f1'],
        ['Precision: какая доля тревог верна', 'Recall: какая доля спама найдена',
         'F1: баланс precision и recall'], COLORS):
        ax.plot(x, [float(r[field]) * 100 for r in rows], marker='o',
                linewidth=2, label=label, color=color)
    ax.set_xticks(x, ticks, fontsize=9)
    ax.set(title=f'{NAMES[dataset]}: насколько хорошо распознаётся спам',
           xlabel='Объём данных для обучения', ylabel='Значение метрики (%)', ylim=(0, 105))
    ax.yaxis.set_major_formatter(PercentFormatter(100))
    ax.legend(loc='lower right')
    save(fig, folder / 'spam_quality.png', note)

    fig, ax = plt.subplots(figsize=(12, 6))
    style(ax)
    errors = [(1 - float(r['accuracy'])) * 100 for r in rows]
    bars = ax.bar(x, errors, color=COLORS[2], width=0.65)
    ax.bar_label(bars, labels=[f'{v:.2f}' for v in errors], padding=4, fontsize=9)
    ax.set_xticks(x, ticks, fontsize=9)
    ax.set(title=f'{NAMES[dataset]}: как часто модель ошибается',
           xlabel='Объём данных для обучения',
           ylabel='Ошибки на каждые 100 тестовых SMS — меньше лучше',
           ylim=(0, max(1, max(errors) * 1.3)))
    save(fig, folder / 'errors.png', note)

    if all('false_positive' in r and 'false_negative' in r for r in rows):
        fig, ax = plt.subplots(figsize=(13, 6))
        style(ax)
        ax.bar([i - 0.18 for i in x], [int(r['false_positive']) for r in rows],
               width=0.36, color=COLORS[2], label='Обычный SMS ошибочно назван спамом')
        ax.bar([i + 0.18 for i in x], [int(r['false_negative']) for r in rows],
               width=0.36, color=COLORS[0], label='Спам пропущен как обычный SMS')
        ax.set_xticks(x, ticks, fontsize=9)
        ax.set(title=f'{NAMES[dataset]}: два типа ошибок', xlabel='Объём данных для обучения',
               ylabel='Количество сообщений на общем тесте')
        ax.legend()
        save(fig, folder / 'error_types.png', note)


def plot_comparison():
    paths = {name: ROOT / 'results' / name / 'metrics.csv' for name in NAMES}
    if not all(p.exists() for p in paths.values()):
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for (name, path), color in zip(paths.items(), COLORS):
        with path.open(encoding='utf-8-sig', newline='') as f:
            rows = sorted(csv.DictReader(f), key=lambda r: int(r['training_messages']))
        for ax, field in zip(axes, ['accuracy', 'spam_f1']):
            ax.plot([int(r['training_messages']) for r in rows],
                    [float(r[field]) * 100 for r in rows], marker='o', color=color, label=NAMES[name])
    for ax, title in zip(axes, ['Точность всех ответов (accuracy)', 'Качество распознавания спама (F1)']):
        style(ax)
        ax.set(title=title, xlabel='Фактическое количество SMS для обучения',
               ylabel='Метрика на собственном тесте (%)', ylim=(0, 105))
        ax.yaxis.set_major_formatter(PercentFormatter(100))
        ax.legend()
    fig.suptitle('Два исследования: разный объём и разные данные', fontsize=16)
    save(fig, ROOT / 'results' / 'comparison.png',
         'Тесты и пропорции классов различаются. Разница качества не доказывает эффект только объёма данных.')


def plot_model_comparison(dataset):
    folder = ROOT / 'results' / dataset
    folders = [folder, folder / 'logistic']
    if not all((p / 'metrics.csv').exists() and (p / 'split.json').exists() for p in folders):
        print(f'Model comparison for {dataset}: both result tables are required.')
        return
    splits = [json.loads((p / 'split.json').read_text(encoding='utf-8')) for p in folders]
    keys = ['seed', 'test_fraction', 'train_messages', 'test_messages',
            'test_spam', 'test_ham']
    if any(k not in splits[0] or k not in splits[1] or splits[0][k] != splits[1][k] for k in keys):
        print(f'Model comparison for {dataset}: cannot verify identical data/split. '
              'Recorded split parameters differ or are missing; comparison skipped.')
        return
    hashes = [s.get('data_sha256') for s in splits]
    if all(hashes) and hashes[0] != hashes[1]:
        print(f'Model comparison for {dataset}: dataset checksums differ; skipped.')
        return
    legacy = not all(hashes)
    if legacy:
        print(f'{dataset}: legacy results; split metadata agrees, but old dataset checksum is missing.')
    tables = []
    for p in folders:
        with (p / 'metrics.csv').open(encoding='utf-8-sig', newline='') as f:
            tables.append({int(r['training_percent']): r for r in csv.DictReader(f)})
    common = sorted(set(tables[0]) & set(tables[1]))
    if not common:
        return
    if any(tables[0][p]['training_messages'] != tables[1][p]['training_messages'] for p in common):
        raise ValueError('Training subset sizes differ between models')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, field, title in zip(axes.flat,
                               ['accuracy', 'spam_precision', 'spam_recall', 'spam_f1'],
                               ['Точность всех ответов', 'Precision: доля верных тревог',
                                'Recall: доля найденного спама', 'F1 для спама']):
        style(ax)
        for table, name, color in zip(tables, ['Нейросеть MLP', 'Логистическая регрессия'], COLORS):
            ax.plot(common, [float(table[p][field]) * 100 for p in common],
                    marker='o', label=name, color=color, linewidth=2)
        ax.set(title=title, xlabel='Доля обучающей выборки (%)', ylabel='Метрика (%)',
               xticks=common, ylim=(0, 105))
        ax.yaxis.set_major_formatter(PercentFormatter(100))
        ax.legend(loc='lower right', fontsize=9)
    fig.suptitle(f'{NAMES[dataset]}: логистическая регрессия против нейросети', fontsize=16)
    save(fig, folder / 'model_comparison.png',
         (f'Параметры разделения совпадают; тест: {splits[0]["test_messages"]} SMS. '
          'Для старого MLP нет контрольной суммы: идентичность данных не подтверждена.\n'
          if legacy else f'Подписи данных и параметры разделения совпадают; тест: {splits[0]["test_messages"]} SMS.\n')
         + 'Показаны только объёмы, для которых готовы обе модели; без повторного обучения.')
