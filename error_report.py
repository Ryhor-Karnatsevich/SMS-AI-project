"""Compact error-analysis report: small vs full training within one study."""
import csv
import html
import json
from pathlib import Path

import numpy as np
from plots import MODEL_NAMES, NAMES, COLORS, style, plt

ROOT = Path(__file__).resolve().parent
SHORT = {'mlp': 'MLP', 'logistic': 'Regresja logistyczna',
         'naive_bayes': 'Naive Bayes', 'svm': 'Linear SVM'}


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as file:
        return list(csv.DictReader(file))


def save_chart(fig, path, note):
    # Keep legend, plot labels and footer in separate vertical regions.
    fig.subplots_adjust(left=0.25, right=0.95, top=0.78, bottom=0.15)
    fig.text(0.25, 0.035, note, fontsize=10, color='#475569')
    fig.savefig(path, dpi=180, facecolor='white')
    plt.close(fig)
    print(f'CREATED: {path.resolve()}')


def build_report(dataset, outputs):
    if not outputs:
        print(f'{dataset}: no saved models available; no overview generated.')
        return
    manifests = {p: json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
                 for p, folder in outputs.items()}
    reference = next(iter(manifests.values()))
    for manifest in manifests.values():
        if any(manifest[k] != reference[k] for k in
               ['dataset', 'data_sha256', 'test_indices_sha256', 'test_messages', 'test_ham', 'test_spam']):
            raise ValueError(f'{dataset}: snapshots do not share identical data and test rows')
    tables = {p: {r['model']: r for r in read_csv(folder / 'summary.csv')}
              for p, folder in outputs.items()}
    predictions = {p: read_csv(folder / 'predictions.csv') for p, folder in outputs.items()}
    # Stable message identity, not CSV row position, is used for error transitions.
    by_model = {p: {m: {r['message_id']: r for r in rows if r['model'] == m}
                    for m in MODEL_NAMES if m in tables[p]} for p, rows in predictions.items()}
    pairs = [m for m in MODEL_NAMES if m in tables.get(10, {}) and m in tables.get(100, {})]
    changes = []
    for model in pairs:
        small, full = by_model[10][model], by_model[100][model]
        if set(small) != set(full) or any(small[k]['actual'] != full[k]['actual'] for k in small):
            raise ValueError(f'{dataset}/{model}: inconsistent test SMS across training sizes')
        old = {k for k, r in small.items() if r['actual'] != r['predicted']}
        new = {k for k, r in full.items() if r['actual'] != r['predicted']}
        changes.append({'model': model, 'errors_10': len(old), 'errors_100': len(new),
                        'corrected': len(old - new), 'introduced': len(new - old),
                        'persistent': len(old & new)})
    complete = len(pairs) == len(MODEL_NAMES)
    output = ROOT / 'results' / dataset / 'error_analysis'
    output.mkdir(parents=True, exist_ok=True)
    if changes:
        with (output / 'changes.csv').open('w', encoding='utf-8-sig', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(changes[0]))
            writer.writeheader(); writer.writerows(changes)
    display = [(p, m, tables[p][m]) for m in MODEL_NAMES
               for p in sorted(tables) if m in tables[p]]
    fig, ax = plt.subplots(figsize=(11, max(5.5, 0.65 * len(display) + 2)))
    style(ax)
    ax.grid(axis='y', visible=False)
    ax.grid(axis='x', alpha=0.18)
    fp = [int(r['false_positive']) for _, _, r in display]
    fn = [int(r['false_negative']) for _, _, r in display]
    y = np.arange(len(display))
    ax.barh(y, fp, color=COLORS[2], label='Fałszywy alarm: ham → spam', height=0.62)
    ax.barh(y, fn, left=fp, color=COLORS[0], label='Przeoczony spam: spam → ham', height=0.62)
    total = [a + b for a, b in zip(fp, fn)]
    for i, (a, b) in enumerate(zip(fp, fn)):
        for start, value in [(0, a), (a, b)]:
            if value >= max(total) * 0.08 and value:
                ax.text(start + value / 2, i, str(value), ha='center', va='center', fontsize=11, color='white')
        ax.text(a + b + max(0.5, max(total) * 0.015), i, str(a + b), va='center', fontsize=11)
    ax.set_yticks(y, [f'{SHORT[m]} · {p}%' for p, m, _ in display], fontsize=12)
    ax.invert_yaxis()
    ax.set_xlim(0, max(1, max(total) * 1.14))
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.set_xlabel('Liczba błędnych odpowiedzi na tym samym teście', fontsize=12, labelpad=12)
    ax.set_title(f'{NAMES[dataset]}\nBłędy przy małej i pełnej próbce treningowej', fontsize=18, loc='left', pad=55)
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.01), ncol=1, frameon=False, fontsize=10)
    status = 'Pełne porównanie: 4/4 modeli.' if complete else f'Porównanie niepełne: {len(pairs)}/4 par 10%–100%.'
    note = f'Test: {reference["test_messages"]} SMS. Procent dotyczy puli treningowej.\n{status}'
    save_chart(fig, output / 'error_comparison.png', note)

    if changes:
        fig, ax = plt.subplots(figsize=(11, 6))
        style(ax)
        ax.grid(axis='y', visible=False)
        ax.grid(axis='x', alpha=0.18)
        y = np.arange(len(changes))
        corrected = [r['corrected'] for r in changes]
        introduced = [r['introduced'] for r in changes]
        for offsets, values, label, color in [
            (y - 0.17, corrected, 'Poprawione: błąd przy 10%, poprawnie przy 100%', COLORS[1]),
            (y + 0.17, introduced, 'Nowe: poprawnie przy 10%, błąd przy 100%', COLORS[2])]:
            bars = ax.barh(offsets, values, height=0.30, color=color, label=label)
            ax.bar_label(bars, padding=5, fontsize=12)
        ax.set_yticks(y, [SHORT[r['model']] for r in changes], fontsize=12)
        ax.invert_yaxis()
        ax.set_xlim(0, max(1, max(corrected + introduced) * 1.18))
        ax.xaxis.get_major_locator().set_params(integer=True)
        ax.set_xlabel('Liczba testowych SMS', fontsize=12, labelpad=12)
        ax.set_title(f'{NAMES[dataset]}\nCo zmienia wzrost treningu z 10% do 100%?', fontsize=18, loc='left', pad=55)
        ax.legend(loc='lower left', bbox_to_anchor=(0, 1.01), frameon=False, fontsize=10)
        save_chart(fig, output / 'error_changes.png', note)

    lines = [f'# Analiza błędów · {NAMES[dataset]}', '',
             f'Ten sam test: **{reference["test_messages"]} SMS** '
             f'({reference["test_ham"]} ham, {reference["test_spam"]} spam). '
             'Porównujemy modele przy 10% i 100% puli treningowej, bez zmiany progu decyzji.', '',
             '**Status: ' + status + '**', '']
    missing = [f'{MODEL_NAMES[m]} ({p}%)' for p in (10, 100) for m in MODEL_NAMES
               if m not in tables.get(p, {})]
    if missing:
        lines += ['Brak zapisanych modeli: ' + ', '.join(missing) + '. '
                  'Starych metryk nie używamy do odtwarzania błędnych SMS. Pełna analiza wymaga zapisania tych modeli.', '']
    lines += ['## Liczby i zmiana błędów', '',
              'FP: zwykły SMS uznany za spam. FN: przeoczony spam.', '',
              '| Model | FP: 10% → 100% | FN: 10% → 100% | Poprawione | Nowe |',
              '|---|---:|---:|---:|---:|']
    for row in changes:
        m = row['model']
        lines.append(f'| {MODEL_NAMES[m]} | {tables[10][m]["false_positive"]} → {tables[100][m]["false_positive"]} '
                     f'| {tables[10][m]["false_negative"]} → {tables[100][m]["false_negative"]} '
                     f'| {row["corrected"]} | {row["introduced"]} |')
    lines += ['', '![Rodzaje błędów](error_comparison.png)', '']
    if changes:
        lines += ['![Poprawione i nowe błędy](error_changes.png)', '', '## Wnioski liczbowe', '']
    for row in changes:
        difference = row['errors_100'] - row['errors_10']
        description = f'spadła o {-difference}' if difference < 0 else f'wzrosła o {difference}' if difference > 0 else 'nie zmieniła się'
        lines.append(f'- **{MODEL_NAMES[row["model"]]}:** liczba błędów {description} '
                     f'({row["errors_10"]} → {row["errors_100"]}). '
                     f'Poprawiono {row["corrected"]} SMS, wprowadzono {row["introduced"]} nowych pomyłek; '
                     f'{row["persistent"]} pozostało błędnych.')
    if changes:
        lines += ['', 'Poprawione i nowe błędy liczymy dla tych samych wiadomości, nie z różnicy samych sum. '
                  'Więcej danych może równocześnie usuwać część pomyłek i wprowadzać inne.']

    reviewed = json.loads((ROOT / 'reviewed_errors.json').read_text(encoding='utf-8-sig')).get(dataset, [])
    lines += ['', '## Krótki przegląd treści', '',
              'Przykłady wybrano jakościowo z wcześniejszego raportu. Nie są losową ani reprezentatywną próbą. '
              'Komentarze są interpretacją treści, nie dowodem przyczyny decyzji modelu.', '']
    for example in reviewed:
        key = example['message_id']
        found = next((lookup[key] for model_rows in by_model.values() for lookup in model_rows.values() if key in lookup), None)
        if found is None:
            continue
        lines += [f'### {example["title"]}', '',
                  '<blockquote><p>' + html.escape(found['text']) + '</p></blockquote>', '',
                  f'Etykieta zbioru: **{found["actual"]}**.']
        for p in sorted(by_model):
            failed = [MODEL_NAMES[m] for m, lookup in by_model[p].items()
                      if key in lookup and lookup[key]['actual'] != lookup[key]['predicted']]
            lines.append(f'Błąd przy {p}%: ' + (', '.join(failed) if failed else 'brak w dostępnych modelach') + '.')
        lines += ['', example['comment'], '']
    lines += ['## Granice wniosków', '',
              'To opis błędów na teście po końcowej ocenie. Nie zmieniamy etykiet, progu ani parametrów na podstawie '
              'tych samych przykładów. Nie jest to pomiar poprawy po dostrojeniu modelu. Jeden podział danych '
              'nie wystarcza do wykazania istotnej przewagi algorytmu. Przy różnych wersjach bibliotek '
              'zgodność miar zbiorczych nie dowodzi identyczności każdej predykcji; szczegóły kontroli '
              'są w manifestach dla 10% i 100%.', '',
              'Pełne predykcje, błędne SMS, wspólne pomyłki i segmenty długości/URL pozostają w podfolderach '
              'odpowiednich rozmiarów treningu.']
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (output / 'status.json').write_text(json.dumps({'complete': complete,
        'compared_models': pairs, 'missing_models': missing, 'training_percentages': sorted(outputs),
        'fit_performed': False}, indent=2), encoding='utf-8')
    print(f'REPORT: {output / "report.md"}')
