"""Freeze validation-only selection, then register its separate test evaluation."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'outputs' / 'accuracy_study_v2'


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register-evaluation', type=Path)
    args = parser.parse_args()
    selection_path = STUDY / 'selection.json'
    if args.register_evaluation is None:
        manifest = json.loads((STUDY / 'study.json').read_text())
        candidates = []
        for entry in manifest['experiments']:
            report_path = ROOT / entry['directory'] / 'report.json'
            report = json.loads(report_path.read_text())
            if report['status'] != 'completed':
                raise ValueError('All declared candidate experiments must complete before selection')
            candidates.append(dict(entry, validation_f1=report['best_validation']['metrics']['model']['f1'],
                step=report['best_validation']['step'], report_sha256=digest(report_path)))
        best = max(candidates, key=lambda item: item['validation_f1'])
        checkpoint = ROOT / best['directory'] / 'best.pt'
        selection = {'criterion': 'full validation micro voxel F1 only', 'selected_name': best['name'],
            'checkpoint': str(checkpoint.relative_to(ROOT)), 'checkpoint_sha256': digest(checkpoint),
            'validation_f1': best['validation_f1'], 'step': best['step'], 'candidates': candidates}
        with selection_path.open('x') as stream:
            stream.write(json.dumps(selection, indent=2) + '\n')
        print(json.dumps(selection, indent=2))
    else:
        selection = json.loads(selection_path.read_text())
        report = json.loads(args.register_evaluation.read_text())
        if report['configuration']['initial_checkpoint_sha256'] != selection['checkpoint_sha256']:
            raise ValueError('Evaluated weights differ from the frozen validation selection')
        if report['held_out_test'] is None or report['mode'] != 'evaluation':
            raise ValueError('Expected completed frozen-checkpoint test evaluation')
        result = dict(selection, evaluation_report=str(args.register_evaluation.resolve()),
            held_out_test=report['held_out_test'], repeated_validation=report['validation'])
        with (STUDY / 'selected_test.json').open('x') as stream:
            stream.write(json.dumps(result, indent=2) + '\n')
        print('Recorded frozen selection test F1:', result['held_out_test']['model']['f1'])
    original = ROOT / 'outputs' / 'training_run_v1'
    hashes = json.loads((STUDY / 'baseline_artifact_hashes.json').read_text())
    for name, expected in hashes.items():
        if digest(original / name) != expected:
            raise ValueError('Original baseline artifact changed: ' + name)
    print('All original baseline artifact hashes unchanged')


if __name__ == '__main__':
    main()
