"""Reproducible cleaning of two separate SMS corpora; standard library only."""
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def normalize(text):
    text = unicodedata.normalize('NFKC', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Cf')
    return ' '.join(text.split())


def exact_key(text):
    return normalize(text).casefold()


def template_key(text):
    text = exact_key(text)
    text = re.sub(r'https?://\S+|www\.\S+', ' URL ', text)
    text = re.sub(r'\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b', ' EMAIL ', text)
    text = re.sub(r'\d+', ' NUM ', text)
    return ' '.join(re.findall(r'\w+', text))


def read_uci():
    path = ROOT / 'data' / 'SMSSpamCollection'
    if not path.exists():
        path = ROOT / 'SMSSpamCollection'
    rows = []
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        parts = line.split('\t', 1)
        rows.append((parts[0], parts[1] if len(parts) == 2 else ''))
    return rows, path


def clean(rows, name, blocked=None, aggressive=False):
    counts = Counter(input_rows=len(rows))
    groups = defaultdict(list)
    rejected = []
    def reject(text, label, reason):
        counts[reason] += 1
        rejected.append({'label': label, 'text': text, 'reason': reason})
    for label, raw in rows:
        text = normalize(raw)
        if label not in ('ham', 'spam'):
            reject(text, label, 'invalid_label'); continue
        if not text or not re.search(r'[A-Za-z]', text):
            reject(text, label, 'empty_or_no_latin_letters'); continue
        if aggressive and (len(text) < 5 or '\ufffd' in text or
                           any(unicodedata.category(c) == 'Cc' for c in text)):
            reject(text, label, 'damaged_or_too_short'); continue
        key = template_key(text) if aggressive else exact_key(text)
        if blocked and (exact_key(text) in blocked[0] or template_key(text) in blocked[1]):
            reject(text, label, 'overlap_with_uci'); continue
        groups[key].append((label, text))
    kept = []
    for group in groups.values():
        if len({label for label, _ in group}) > 1:
            for label, text in group:
                reject(text, label, 'conflicting_labels')
        else:
            label, text = group[0]
            kept.append({'label': label, 'text': text})
            for label, text in group[1:]:
                reject(text, label, 'duplicate_template' if aggressive else 'duplicate_text')
    folder = ROOT / 'data' / 'clean'
    folder.mkdir(parents=True, exist_ok=True)
    for suffix, records, fields in [('', kept, ['label', 'text']),
                                   ('_rejected', rejected, ['label', 'text', 'reason'])]:
        with (folder / f'{name}{suffix}.csv').open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader(); writer.writerows(records)
    counts['kept_rows'] = len(kept)
    assert counts['input_rows'] == len(kept) + len(rejected)
    return kept, dict(counts)


def main():
    uci_rows, uci_path = read_uci()
    uci, uci_report = clean(uci_rows, 'uci')
    # Block ALL original UCI texts, including those rejected by cleaning.
    blocked = ({exact_key(t) for _, t in uci_rows},
               {template_key(t) for _, t in uci_rows})
    source = ROOT / 'data' / 'super_sms' / 'super_sms_dataset.csv'
    # Authors read their corpus with latin-1. Preserve the original download.
    with source.open(encoding='latin-1', newline='') as f:
        raw = list(csv.DictReader(f))
    rows = [({'0': 'ham', '1': 'spam'}.get(r.get('Labels', '').strip(), 'invalid'),
             r.get('SMSes') or '') for r in raw]
    super_rows, super_report = clean(rows, 'super_sms', blocked, aggressive=True)
    assert not ({template_key(r['text']) for r in super_rows} & blocked[1])
    report = {'source': 'https://github.com/smspamresearch/spstudy',
              'encoding': 'latin-1, following author notebook; no automatic language detection',
              'label_mapping': {'0': 'ham', '1': 'spam'},
              'policy': 'Super: NFKC, invisible-character removal, invalid/empty/damaged filtering; '
                        'remove UCI exact/template overlaps; deduplicate URL/email/number/punctuation templates; '
                        'drop all members of conflicting-label groups. Keep original text features.',
              'uci': uci_report, 'super_sms': super_report,
              'class_counts': {name: dict(Counter(r['label'] for r in rows))
                               for name, rows in [('uci', uci), ('super_sms', super_rows)]},
              'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [uci_path, source]}}
    (ROOT / 'data' / 'clean' / 'cleaning_report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
