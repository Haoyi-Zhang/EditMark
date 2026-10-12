#!/usr/bin/env python3
"""Bounded, model-free evidence inventory and same-information arithmetic checks.

Reads retained data as text. Never executes programs, edits, models or detectors.
Fixture ablations remain explicitly synthetic. Direct arithmetic is independently
written here; it is not an independent empirical experiment or a novelty claim.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import tracemalloc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from editmark_audit.aggregates import analyze
from editmark_audit.analysis import summarize
from editmark_audit.certificate import certify_analysis
from editmark_audit.constants import CERTIFICATE_BUNDLE_SCHEMA
from editmark_audit.io import EvidenceError, load_rows, loads, write_new_json
from editmark_audit.records import Pair
from editmark_audit.verify import verify_bundle


def rational(value):
    return None if value is None else [value.numerator, value.denominator]


def direct_counts(document):
    """Ordinary exact calculation, without certificate/provenance guarantees."""
    counts = document['counts']
    n, b, a = (counts[k] for k in ('n', 'before_positive', 'after_positive'))
    if any(type(x) is not int or x < 0 for x in (n, b, a)) or max(b, a) > n:
        raise EvidenceError('Direct calculation requires exact same-cohort counts')
    transitions = document.get('transitions')
    lo, hi = max(0, a + b - n), min(a, b)
    if transitions is not None:
        t = [transitions[k] for k in ('n00', 'n01', 'n10', 'n11')]
        if (any(type(x) is not int or x < 0 for x in t) or sum(t) != n
                or t[2] + t[3] != b or t[1] + t[3] != a):
            raise EvidenceError('Contradictory transition counts')
        lo = hi = t[3]
    return {
        'change': rational(Fraction(a - b, n)) if n else None,
        'survival_bounds': [rational(Fraction(lo, b)), rational(Fraction(hi, b))]
        if b else [None, None],
    }


def direct_gate(pair):
    """Independent strict eligibility specification; does not call analyzer gates."""
    if pair.supported is not True:
        return 'unsupported' if pair.supported is False else 'support_unknown'
    if pair.changed is not True:
        return 'unchanged' if pair.changed is False else 'change_unknown'
    if not pair.original.source_sha256 or not pair.edited.source_sha256:
        return 'source_identity_missing'
    for side in ('original', 'edited'):
        v = getattr(pair, side).validation
        if v.available is not True:
            return side + '_validation_unavailable'
        if v.passed is not True:
            return side + ('_checks_failed' if v.passed is False else '_checks_unknown')
    b, a = pair.original, pair.edited
    if not b.validation.contract_id or not a.validation.contract_id:
        return 'check_contract_unknown'
    if b.validation.contract_id != a.validation.contract_id:
        return 'check_contract_changed'
    rules = [(p.detection.rule_id, p.detection.threshold, p.detection.comparator)
             for p in (b, a)]
    if any(not r[0] or r[1] is None or not r[2] for r in rules):
        return 'decision_rule_unknown'
    if rules[0] != rules[1]:
        return 'decision_rule_changed'
    return None


def direct_pairs(pairs):
    target, joint, exclusions = [], Counter(), Counter()
    for p in pairs:
        reason = direct_gate(p)
        if reason is None:
            target.append(p)
            for side in ('original', 'edited'):
                d = getattr(p, side).detection
                if d.available is not True or d.detected is None:
                    reason = side + '_decision_unavailable'
                    break
        if reason is not None:
            exclusions[reason] += 1
        else:
            joint[f'n{int(p.original.detection.detected)}{int(p.edited.detection.detected)}'] += 1
    counts = {k: joint[k] for k in ('n00', 'n01', 'n10', 'n11')}
    n = sum(counts.values())
    choices = []
    for p in target:
        sides = [getattr(p, s).detection for s in ('original', 'edited')]
        bits = [(d.detected,) if d.available is True and d.detected is not None
                else (False, True) for d in sides]
        choices.append(tuple(int(a) - int(b) for b, a in itertools.product(*bits)))
    # Bounded exhaustive completion, not the analyzer's interval formula.
    completion_count = 1
    for c in choices:
        completion_count *= len(c)
    if completion_count > 4096:
        raise EvidenceError('Fixture completion enumeration exceeds 4096 worlds')
    sums = [sum(world) for world in itertools.product(*choices)] if target else []
    change = Fraction(counts['n01'] - counts['n10'], n) if n else None
    before_positive = counts['n10'] + counts['n11']
    return {
        'attempted_pairs': len(pairs), 'target_pairs': len(target), 'complete_pairs': n,
        'transitions': counts, 'first_exclusion_counts': dict(sorted(exclusions.items())),
        'change': rational(change),
        'survival': rational(Fraction(counts['n11'], before_positive)) if before_positive else None,
        'target_change_bounds': [rational(Fraction(min(sums), len(target))),
                                 rational(Fraction(max(sums), len(target)))]
        if target else [None, None],
        'completion_worlds': completion_count if target else 0,
    }


def fixture_checks(root):
    results = []
    for filename in ('synthetic_pairs.jsonl', 'synthetic_count_basis.jsonl'):
        original = load_rows(root / 'examples' / filename)
        variants = ['retained'] if filename == 'synthetic_pairs.jsonl' else [
            'retained', 'original_decision_missing', 'edited_decision_missing',
            'both_decisions_missing', 'all_decisions_missing', 'noop',
            'validator_drift', 'rule_drift',
        ]
        for variant in variants:
            rows = copy.deepcopy(original)
            if 'missing' in variant:
                sides = ('original', 'edited') if variant.startswith(('both', 'all')) else (
                    'original' if variant.startswith('original') else 'edited',)
                for row in rows if variant.startswith('all') else rows[:1]:
                    for side in sides:
                        row[side]['detection'].update(available=False, detected=None, score=None)
            elif variant == 'noop':
                rows[0]['edited']['source'] = rows[0]['original']['source']
                rows[0]['changed'] = False
            elif variant == 'validator_drift':
                rows[0]['edited']['validation']['contract_id'] = 'owned-other-checks'
            elif variant == 'rule_drift':
                rows[0]['edited']['detection']['rule_id'] = 'owned-other-rule'
            pairs = [Pair.parse(row) for row in rows]
            expected = direct_pairs(pairs)
            actual = summarize(pairs)
            interval = actual['missing_decision_change_bounds']
            comparisons = {
                'attempted_pairs': actual['attempted_pairs'],
                'target_pairs': actual['eligible_target_pairs'],
                'complete_pairs': actual['admitted_pairs'],
                'transitions': {k: actual[k] for k in expected['transitions']},
                'first_exclusion_counts': actual['first_exclusion_counts'],
                'change': [actual['net_detection_change_exact'][k] for k in
                           ('numerator', 'denominator')]
                if actual['net_detection_change_exact'] else None,
                'survival': rational(Fraction(actual['n11'], actual['survival_denominator']))
                if actual['survival_denominator'] else None,
                'target_change_bounds': [
                    rational(Fraction(interval[k], interval['denominator']))
                    for k in ('lower_numerator', 'upper_numerator')]
                if interval['denominator'] else [None, None],
            }
            if any(expected[k] != v for k, v in comparisons.items()):
                raise AssertionError(f'Direct pair mismatch: {filename}/{variant}')
            expected_survival = float(Fraction(*expected['survival'])) if expected['survival'] else None
            if actual['survival'] != expected_survival:
                raise AssertionError(f'Direct survival mismatch: {filename}/{variant}')
            results.append({'file': 'examples/' + filename, 'variant': variant,
                            'evidence_kind': 'synthetic_test', 'agrees': True, **expected})
    document = load_rows(root / 'examples/synthetic_cohort_counts.json')[0]
    expected = direct_counts(document)
    actual = analyze(document)
    observed = {'change': [actual['net_detection_change'][k] for k in ('numerator', 'denominator')],
                'survival_bounds': [[actual['survival_bounds'][s][k] for k in
                                     ('numerator', 'denominator')] for s in ('lower', 'upper')]}
    if expected != observed:
        raise AssertionError('Direct marginal arithmetic mismatch')
    return {'pair_cases': results, 'marginals': expected, 'evidence_kind': 'synthetic_test',
            'warning': 'NOT AN EMPIRICAL WATERMARK RESULT'}


def inventory(root):
    """Read every retained JSON/JSONL/CSV; count candidate forms, not inferred events."""
    files, pair_kinds, legacy_rows, ambiguous, errors = [], Counter(), 0, [], []
    total_bytes = 0
    format_issues = []
    for path in sorted(root.rglob('*')):
        rel = path.relative_to(root)
        if (not path.is_file() or path.suffix.lower() not in ('.json', '.jsonl', '.csv')
                or any(s in ('.git', '__pycache__', 'repair_diagnostic') for s in rel.parts)):
            continue
        size = path.stat().st_size
        total_bytes += size
        if size > 64 * 1024**2 or total_bytes > 128 * 1024**2:
            raise EvidenceError('Inventory exceeds bounded 64 MiB/file or 128 MiB total')
        item = {'path': rel.as_posix(), 'bytes': size, 'objects': 0, 'normalized_pairs': 0}
        try:
            if path.suffix.lower() == '.csv':
                with path.open(encoding='utf-8-sig', newline='') as f:
                    reader = csv.DictReader(f)
                    item['columns'] = reader.fieldnames
                    item['rows'] = sum(1 for _ in reader)
                    if {'example_id', 'watermarked_detected', 'attacked_detected'} <= set(reader.fieldnames or []):
                        legacy_rows += item['rows']
                        ambiguous.append(rel.as_posix())
            else:
                try:
                    documents = load_rows(path)
                except EvidenceError as exc:
                    # Inspect an archived invalid-JSON aggregate without making its
                    # non-finite literal a measured zero, a finite score or a count.
                    # The authoritative analyzer still rejects these literals.
                    if path.suffix.lower() != '.json' or 'Non-finite JSON constant:' not in str(exc):
                        raise
                    constants = []
                    def retain_constant(token):
                        constants.append(token)
                        return {'recorded_nonfinite_literal': token}
                    def unique_object(items):
                        if len({key for key, _ in items}) != len(items):
                            raise EvidenceError('Duplicate JSON key in archived aggregate')
                        return dict(items)
                    value = json.loads(path.read_text(encoding='utf-8-sig'),
                                       parse_constant=retain_constant, object_pairs_hook=unique_object,
                                       parse_float=loads)
                    # The second strict pass retains duplicate/non-finite-output
                    # rejection and underflow validation where applicable.
                    value = loads(json.dumps(value, allow_nan=False))
                    documents = value if isinstance(value, list) else [value]
                    format_issues.append({'path': rel.as_posix(), 'recorded_literals': constants,
                                          'handling': 'Tagged for structural inspection only; original file unchanged, numerical outcome unresolved.'})
                stack = list(documents)
                while stack:
                    value = stack.pop()
                    if isinstance(value, dict):
                        item['objects'] += 1
                        if '$schema' in value:
                            item['schema_definition'] = True
                            continue  # A schema's properties are not event records.
                        if value.get('schema_version') == 'editmark-pair':
                            p = Pair.parse(value)
                            pair_kinds[p.evidence_kind] += 1
                            item['normalized_pairs'] += 1
                        if {'example_id', 'watermarked_detected', 'attacked_detected'} <= value.keys():
                            legacy_rows += 1
                            ambiguous.append(rel.as_posix())
                        if {'original', 'edited'} <= value.keys() and value.get('schema_version') != 'editmark-pair':
                            ambiguous.append(rel.as_posix())
                        stack.extend(v for v in value.values() if isinstance(v, (list, dict)))
                    elif isinstance(value, list):
                        stack.extend(value)
        except (EvidenceError, OSError, UnicodeError, ValueError) as exc:
            errors.append({'path': rel.as_posix(), 'error': str(exc)})
        files.append(item)
    corpus = []
    for path in sorted((root / 'data/release/sources').glob('*.normalized.jsonl')):
        rows = load_rows(path)
        corpus.append({'file': path.relative_to(root).as_posix(), 'released': len(rows),
                       'configured_canonical': sum(r.get('reference_kind') == 'canonical' for r in rows),
                       'configured_go': sum(r.get('reference_kind') == 'canonical' and r['language'] == 'go' for r in rows)})
    complete = not errors and not ambiguous and pair_kinds['saved_observation'] == 0
    return {'files': files, 'file_count': len(files), 'bytes_read': total_bytes,
            'normalized_pair_kinds': dict(pair_kinds), 'legacy_pair_rows': legacy_rows,
            'unresolved_candidate_files': sorted(set(ambiguous)), 'read_errors': errors,
            'archived_format_issues': format_issues,
            'released_task_inventory': corpus,
            'paired_effect_status': 'not_identifiable_from_retained_data' if complete else 'requires_record_review',
            'paired_change': None, 'edited_fpr': None, 'auroc_change': None,
            'scope': 'Candidate artifact JSON/JSONL/CSV only; source records are tasks, not detector events.'}


def rounded_components(root):
    path = root / 'results/tables/suite_all_models_methods/suite_all_models_methods_method_master_leaderboard.json'
    rows = load_rows(path)
    results = []
    for row in rows:
        for edit, cell in row['score_coverage']['attack_breakdown'].items():
            terms = [Fraction(str(cell[k])) for k in (
                'attack_retention', 'attack_attacked_detected_semantic_rate',
                'attack_attacked_pass_preservation') if cell[k] is not None]
            computed = sum(terms, Fraction()) / len(terms) if terms else None
            saved = Fraction(str(cell['attack_robustness'])) if cell['attack_robustness'] is not None else None
            residual = abs(computed - saved) if computed is not None and saved is not None else None
            results.append({'method': row['method'], 'edit': edit,
                            'direct_composite': rational(computed), 'saved_rounded_composite': rational(saved),
                            'absolute_rounding_residual': rational(residual),
                            'paired_change': None, 'warning': 'Rounded composite arithmetic only; not paired detector loss.'})
    return results


def costs(root, repetitions):
    path = root / 'examples/synthetic_cohort_counts.json'
    def direct():
        return json.dumps(direct_counts(loads(path.read_text(encoding='utf-8-sig'))), sort_keys=True).encode()
    def contract():
        data = path.read_bytes()
        document = loads(data.decode('utf-8-sig'))
        analyzed = analyze(document)
        bundle = {'schema_version': CERTIFICATE_BUNDLE_SCHEMA,
                  'input_file': {'name': path.name, 'sha256': hashlib.sha256(data).hexdigest()},
                  'requested_claim': 'positive_decision_change',
                  'certificates': [certify_analysis(analyzed, 'positive_decision_change')]}
        verified = verify_bundle(bundle)
        return json.dumps({'analysis': analyzed, 'bundle': bundle, 'verification': verified}, sort_keys=True).encode()
    result = {}
    for name, function in (('ordinary_direct', direct), ('contract_and_verification', contract)):
        function()  # warm imports/file cache outside timing
        samples = []
        for _ in range(repetitions):
            start = time.perf_counter_ns()
            payload = function()
            samples.append(time.perf_counter_ns() - start)
        tracemalloc.start()
        function()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        result[name] = {'repetitions': repetitions, 'median_ns': statistics.median(samples),
                        'min_ns': min(samples), 'max_ns': max(samples),
                        'peak_python_allocation_bytes_separate_run': peak, 'serialized_output_bytes': len(payload)}
    end_to_end = {}
    parent = root / 'results/repair_diagnostic'
    parent.mkdir(parents=True, exist_ok=True)
    for worker in ('ordinary_direct', 'contract_and_verification'):
        times, sizes = [], []
        for _ in range(7):
            with tempfile.TemporaryDirectory(prefix='cost-', dir=parent) as td:
                start = time.perf_counter_ns()
                process = subprocess.run(
                    [sys.executable, '-B', str(Path(__file__).resolve()),
                     '--cost-worker', worker, '--out', td],
                    cwd=root, capture_output=True, text=True, check=True,
                )
                times.append(time.perf_counter_ns() - start)
                sizes.append(sum(p.stat().st_size for p in Path(td).iterdir() if p.is_file()))
        end_to_end[worker] = {'repetitions': 7, 'median_ns': statistics.median(times),
                             'min_ns': min(times), 'max_ns': max(times),
                             'published_output_bytes': sorted(set(sizes))}
    return {'measurements': result, 'subprocess_durable_workflow': end_to_end,
            'included': 'Warm in-process input read, strict JSON decode, arithmetic and JSON serialization; contract additionally validates identity/partition, hashes, certifies and re-derives.',
            'excluded': 'Interpreter startup/imports, inventory scan, process/RSS/native allocation, durable publication/fsync/link, model/detector/task execution.',
            'durable_workflow_scope': 'Separately measured fresh Python process, imports, input reads/digests, calculations, JSON serialization and exclusive fsync/hard-link publication. Contract uses count, certificate and basis-verifier CLI functions in one process and publishes three files; direct publishes one. Parent temporary-directory creation/deletion and result-file stat calls are outside timing. Not three separate CLI processes.',
            'comparability': 'Same counts/information and estimands, different guarantees/output sizes; ordinary arithmetic is not a certificate verifier.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--repetitions', type=int, default=101)
    parser.add_argument('--cost-worker', choices=('ordinary_direct', 'contract_and_verification'))
    args = parser.parse_args(argv)
    if not 1 <= args.repetitions <= 1000:
        parser.error('repetitions must be in [1,1000]')
    output = args.out.resolve()
    if not output.is_relative_to(ROOT / 'results/repair_diagnostic'):
        parser.error('output must be within candidate artifact/results/repair_diagnostic')
    if args.cost_worker:
        path = ROOT / 'examples/synthetic_cohort_counts.json'
        if args.cost_worker == 'ordinary_direct':
            data = path.read_bytes()
            write_new_json(output / 'direct.json', {
                'input_sha256': hashlib.sha256(data).hexdigest(),
                'evidence_kind': 'synthetic_test',
                'calculation': direct_counts(loads(data.decode('utf-8-sig'))),
                'warning': 'Arithmetic only; no certificate/provenance guarantees.',
            })
        else:
            from editmark_audit import aggregates, certificate, verify
            counted, certified, verified = (output / x for x in ('counts.json', 'certificate.json', 'verification.json'))
            for function, arguments in (
                (aggregates.main, [str(path), '--out', str(counted)]),
                (certificate.main, [str(counted), '--claim', 'positive_decision_change', '--out', str(certified)]),
                (verify.main, [str(certified), '--basis', str(counted), '--out', str(verified)]),
            ):
                if function(arguments) != 0:
                    raise EvidenceError('Cost worker failed; no timing comparison is valid')
        return 0
    result = {'environment': {'python': sys.version, 'executable': sys.executable,
                              'platform': platform.platform()},
              'inventory': inventory(ROOT), 'same_information_checks': fixture_checks(ROOT),
              'rounded_component_checks': rounded_components(ROOT), 'costs': costs(ROOT, args.repetitions),
              'scientific_boundary': 'No new real paired observations, detector performance, external validation or research novelty established.'}
    write_new_json(output, result)
    print(json.dumps({
        'environment': result['environment'],
        'inventory': {k: v for k, v in result['inventory'].items() if k != 'files'},
        'same_information_checks': result['same_information_checks'],
        'costs': result['costs'],
        'rounded_component_cells': len(result['rounded_component_checks']),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
