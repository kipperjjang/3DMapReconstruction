"""Append immutable experiment records and render the accuracy study comparison."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'outputs' / 'accuracy_study_v2'


def main():
    manifest = json.loads((STUDY / 'study.json').read_text())
    ledger = STUDY / 'experiments.jsonl'
    existing = {r['name']: r for r in (json.loads(line) for line in ledger.read_text().splitlines())} if ledger.exists() else {}
    rows = []
    for entry in manifest['experiments']:
        path = ROOT / entry['directory'] / 'report.json'
        if not path.exists():
            continue
        report = json.loads(path.read_text())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        record = dict(entry, report_sha256=digest, configuration=report['configuration'],
            validation=report['best_validation'], test=report['held_out_test'])
        if entry['name'] in existing:
            if existing[entry['name']]['report_sha256'] != digest:
                raise ValueError('An already recorded report was changed: ' + str(path))
        else:
            with ledger.open('a') as stream:
                stream.write(json.dumps(record) + '\n')
        model = report['best_validation']['metrics']['model']
        test = report['held_out_test']
        final = STUDY / 'selected_test.json'
        if final.exists():
            final_report = json.loads(final.read_text())
            if final_report['selected_name'] == entry['name']:
                test = final_report['held_out_test']
        rows.append({'experiment': entry['name'], 'updates': report['configuration']['updates'],
            'noise_m': report['configuration']['position_noise_m'],
            'initial_lr': report['configuration'].get('initial_lr', .01) if report['configuration']['updates'] else 'BN statistics only',
            'best_step': report['best_validation']['step'],
            'validation_precision': model['precision'], 'validation_recall': model['recall'], 'validation_f1': model['f1'],
            'test_precision': test['model']['precision'] if test else '',
            'test_recall': test['model']['recall'] if test else '',
            'test_f1': test['model']['f1'] if test else '', 'directory': entry['directory']})
    if not rows:
        return
    with (STUDY / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (STUDY / 'comparison.json').write_text(json.dumps(rows, indent=2) + '\n')
    lines = ['# Accuracy study comparison', '',
        'Same dataset/split/grid/architecture and predictive alpha=0.5. Selection uses validation F1 only.',
        'Candidate test evaluation is deferred; only the selected candidate is evaluated on the existing benchmark.',
        'This test scene was used in the previous experiment, so it is not a newly unseen benchmark.', '',
        '| Experiment | Updates | Noise (m) | Initial LR | Best step | Validation F1 | Test F1 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        test = '{:.6f}'.format(r['test_f1']) if r['test_f1'] != '' else 'deferred'
        lines.append('| {experiment} | {updates} | {noise_m} | {initial_lr} | {best_step} | {validation_f1:.6f} | '.format(**r) + test + ' |')
    lines += ['', 'Every run directory contains original metrics, checkpoints, configuration and (for new runs) a frozen source snapshot.',
              'experiments.jsonl is an append-only ledger; changed recorded reports are rejected.']
    (STUDY / 'COMPARISON.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
