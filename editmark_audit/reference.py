"""Bounded independent arithmetic check, not a replacement for the full verifier.

Standard library only; imports neither analyzer, certifier, parser nor bounds.
Enumerates feasible 2x2 tables for count certificates with n<=10000. Optional
raw-basis mode supports complete source-linked, score-null pairs only and recounts
source groups. Other evidence paths must use the full verifier.
No source/detector execution, missing-data inference or origin authentication.
"""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

STRATUM = ('experiment_id', 'method', 'model', 'source_group', 'language', 'edit', 'variant_id')
DENIALS = {
    'edited_false_positive_rate': 'missing_negative_cohort',
    'auroc_change': 'missing_score_distributions_and_controls',
    'post_edit_provenance_probability': 'missing_prevalence_and_calibration',
}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load(text):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON field')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('non-finite JSON constant: ' + value)
    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


def integer(value):
    require(type(value) is int and value >= 0, 'exact nonnegative count required')
    return value


def identifier(value):
    require(type(value) is str and bool(value.strip()) and value == value.strip(), 'nonempty canonical identifier')
    return value


def digest_text(value):
    require(type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value),
            'lowercase SHA-256 required')
    return value


def normalized_rule(rule):
    require(type(rule) is dict and set(rule) == {'rule_id', 'threshold', 'comparator'}, 'rule fields')
    value = rule['threshold']
    require(type(value) in (int, float), 'numeric threshold required')
    threshold = float(value)
    require(math.isfinite(threshold) and (type(value) is not int or threshold == value), 'finite exact threshold')
    require(rule['comparator'] in ('ge', 'gt', 'le', 'lt'), 'rule comparator')
    return dict(rule_id=identifier(rule['rule_id']), threshold=threshold, comparator=rule['comparator'])


def fraction_matches(observed, expected):
    if expected is None:
        return observed is None
    return (type(observed) is dict and set(observed) == {'numerator', 'denominator', 'decimal'}
        and type(observed['numerator']) is int and type(observed['denominator']) is int
        and observed['denominator'] > 0
        and (observed['numerator'], observed['denominator']) == (expected.numerator, expected.denominator)
        and type(observed['decimal']) is float and observed['decimal'] == float(expected))


def check_certificate(item):
    require(item['schema_version'] == 'editmark-claim-certificate', 'unsupported certificate schema')
    core = {key: value for key, value in item.items() if key != 'certificate_sha256'}
    require(item['certificate_sha256'] == sha(canonical(core).encode()), 'certificate hash')
    require(item['population'] == 'watermarked_positive' and
            item['analysis_contract'] == 'same-rule-changed-pairs', 'unsupported population/contract')
    require(item['membership_verified'] is False and item['claim_text_verified'] is False,
            'unverified origin/free prose cannot be promoted')
    digest_text(item['cohort_sha256'])
    require(type(item['stratum']) is dict and set(item['stratum']) == set(STRATUM), 'stratum fields')
    for value in item['stratum'].values():
        identifier(value)
    require(item['evidence_kind'] in ('saved_observation', 'synthetic_test'), 'evidence kind')
    counts = item['counts']
    require(set(counts) == {'n', 'before_positive', 'after_positive'}, 'count fields')
    n, b, a = (integer(counts[key]) for key in ('n', 'before_positive', 'after_positive'))
    require(max(a, b) <= n <= 10000, 'outside bounded arithmetic domain')
    excluded = item['first_exclusion_counts']
    require(type(excluded) is dict and integer(item['attempted_pairs']) ==
            n + sum(integer(v) for v in excluded.values()), 'exclusion partition')
    normalized_rule(item['decision_rule'])
    # Represent possible joint facts explicitly, not via production bound helpers.
    tables = []
    for j in range(min(a, b) + 1):
        table = dict(n00=n-a-b+j, n01=a-j, n10=b-j, n11=j)
        if all(v >= 0 for v in table.values()):
            tables.append(table)
    supplied = item.get('transitions')
    if supplied is not None:
        require(type(supplied) is dict and set(supplied) == {'n00', 'n01', 'n10', 'n11'}, 'joint fields')
        for value in supplied.values():
            integer(value)
        require(supplied in tables, 'joint table contradicts marginals')
        tables = [supplied]
    require(bool(tables), 'no feasible table')
    claim = item['requested_claim']
    expected = None
    bounds = None
    if claim == 'positive_decision_change':
        if n:
            values = {Fraction(t['n01']-t['n10'], n) for t in tables}
            require(len(values) == 1, 'net change not constant')
            expected = values.pop()
            decision, status = 'admissible', 'identified'
            statement = ('On the named jointly valid, changed, same-rule positive cohort, '
                         f'the after-minus-before positive-decision change is {float(expected):.6g}.')
        else:
            decision, status = 'not_admissible', 'undefined'
            statement = 'The supplied disclosure does not identify a nonempty same-cohort change.'
    elif claim == 'detection_survival':
        if not b:
            decision, status = 'not_admissible', 'undefined_no_initial_detections'
            statement = 'Detection survival is undefined because the cohort has no initial positives.'
        else:
            values = [Fraction(t['n11'], b) for t in tables]
            lo, hi = min(values), max(values)
            if lo == hi:
                expected = lo
                decision, status = 'admissible', 'identified'
                statement = f'Detection survival is identified on the named cohort as {float(lo):.6g}.'
            else:
                decision, status, bounds = 'partially_admissible', 'sharp_bounds', (lo, hi)
                statement = ('The marginals identify only a sharp finite-cohort interval for detection survival: '
                             f'[{float(lo):.6g}, {float(hi):.6g}].')
    else:
        require(claim in DENIALS, 'unsupported claim')
        decision, status = 'not_admissible', DENIALS[claim]
        # Denial wording is checked by the full verifier, not this arithmetic path.
        statement = None
    require((item['decision'], item['identification']) == (decision, status), 'claim status')
    require(fraction_matches(item['estimate'], expected), 'estimate arithmetic')
    if bounds:
        require(set(item.get('bounds', {})) == {'lower', 'upper'} and
                all(fraction_matches(item['bounds'][key], value)
                    for key, value in zip(('lower', 'upper'), bounds)), 'bound arithmetic')
    else:
        require('bounds' not in item, 'unexpected bounds')
    if statement is not None:
        require(item['supported_statement'] == statement, 'numeric statement')
    return dict(claim=claim, feasible_tables=len(tables), decision=decision, identification=status)


def check_bundle(bundle):
    require(type(bundle) is dict and set(bundle) == {'schema_version', 'input_file', 'requested_claim', 'certificates'},
            'bundle fields')
    require(bundle['schema_version'] in ('editmark-claim-certificate-bundle', 'editmark-claim-certificate-bundle-v2'),
            'unsupported bundle schema')
    require(type(bundle['input_file']) is dict and set(bundle['input_file']) == {'name', 'sha256'}, 'input-file fields')
    identifier(bundle['input_file']['name'])
    digest_text(bundle['input_file']['sha256'])
    require(type(bundle['certificates']) is list and bool(bundle['certificates']), 'nonempty certificates required')
    claim = bundle['requested_claim']
    require(claim in ('positive_decision_change', 'detection_survival', *DENIALS), 'unsupported bundle claim')
    seen = {}
    results = []
    kinds = set()
    for cert in bundle['certificates']:
        require(cert['requested_claim'] == claim, 'bundle/certificate claim mismatch')
        result = check_certificate(cert)
        kinds.add(cert['evidence_kind'])
        identity = (cert['cohort_sha256'], tuple(cert['stratum'][k] for k in STRATUM))
        facts = canonical({k:cert[k] for k in ('evidence_kind', 'population', 'analysis_contract',
            'counts', 'transitions', 'decision_rule', 'attempted_pairs', 'first_exclusion_counts')})
        require(identity not in seen or seen[identity] == facts, 'conflicting cohort/stratum facts')
        if identity not in seen:
            results.append(result)
        seen[identity] = facts
    require(len(kinds) == 1, 'mixed evidence kinds')
    return results


def check_complete_basis(bundle, basis):
    """Independently recount the deliberately narrow complete-pair evidence path."""
    check_bundle(bundle)
    raw = Path(basis).read_bytes()
    require(sha(raw) == bundle['input_file']['sha256'], 'basis digest')
    groups = defaultdict(list)
    seen = {}
    originals = {}
    for line in raw.decode('utf-8-sig').splitlines():
        if not line.strip():
            continue
        row = load(line)
        require(row['schema_version'] == 'editmark-pair' and
                row['population'] == 'watermarked_positive', 'unsupported raw pair')
        key = tuple(row[k] for k in STRATUM) + (row['sample_id'], row['replicate_id'])
        for value in key:
            identifier(value)
        encoded = canonical(row)
        if key in seen:
            require(seen[key] == encoded, 'conflicting duplicate raw pair')
            continue
        seen[key] = encoded
        old, new = row['original'], row['edited']
        original_key = tuple(row[k] for k in STRATUM[:5]) + (row['sample_id'], row['replicate_id'])
        original_facts = canonical([row['evidence_kind'], old, row.get('cluster_id')])
        require(original_key not in originals or originals[original_key] == original_facts,
                'conflicting repeated original')
        originals[original_key] = original_facts
        require(row['supported'] is True and row['changed'] is True, 'basis requires complete eligible pairs')
        for side in (old, new):
            require(sha(side['source'].encode()) == side['source_sha256'], 'raw source digest')
            validation, detection = side['validation'], side['detection']
            require(validation['available'] is True and validation['passed'] is True,
                    'basis functional assertions unavailable')
            identifier(validation['contract_id'])
            require(detection['available'] is True and type(detection['detected']) is bool,
                    'basis requires complete Boolean decisions')
            require(detection['score'] is None, 'reference basis supports score-null Booleans only')
        require(old['source_sha256'] != new['source_sha256'] and
                old['validation']['contract_id'] == new['validation']['contract_id'], 'change/contract mismatch')
        old_rule = normalized_rule({k:old['detection'][k] for k in ('rule_id', 'threshold', 'comparator')})
        new_rule = normalized_rule({k:new['detection'][k] for k in ('rule_id', 'threshold', 'comparator')})
        require(old_rule == new_rule, 'raw rule drift')
        groups[tuple(row[k] for k in STRATUM)].append(row)
    require(set(groups) == {tuple(c['stratum'][k] for k in STRATUM) for c in bundle['certificates']},
            'basis stratum coverage')
    dependence = []
    for key, rows in sorted(groups.items()):
        transitions = Counter('n'+str(int(r['original']['detection']['detected']))+
                              str(int(r['edited']['detection']['detected'])) for r in rows)
        table = {k: transitions[k] for k in ('n00', 'n01', 'n10', 'n11')}
        counts = dict(n=len(rows), before_positive=table['n10']+table['n11'],
                      after_positive=table['n01']+table['n11'])
        # Independently encode the published membership tuple contract. Shared
        # serialization conventions are required; no production parser is used.
        payload = []
        for row in sorted(rows, key=lambda r:(r['evidence_kind'], *[r[k] for k in STRATUM[:5]],
                          r['sample_id'], r['replicate_id'], r['edit'], r['variant_id'])):
            old, new = row['original'], row['edited']
            pair_key = (row['evidence_kind'], *[row[k] for k in STRATUM[:5]],
                        row['sample_id'], row['replicate_id'], row['edit'], row['variant_id'])
            rule = normalized_rule({k:old['detection'][k] for k in ('rule_id', 'threshold', 'comparator')})
            payload.append((pair_key, old['source_sha256'], new['source_sha256'],
                old['validation']['contract_id'], (rule['rule_id'], rule['threshold'], rule['comparator']),
                old['detection']['detected'], new['detection']['detected']))
        cohort_hash = sha(json.dumps(payload, separators=(',', ':'), ensure_ascii=True).encode())
        for cert in bundle['certificates']:
            if tuple(cert['stratum'][k] for k in STRATUM) != key:
                continue
            require(canonical(cert['counts']) == canonical(counts), 'basis count mismatch')
            require(cert['cohort_sha256'] == cohort_hash, 'basis cohort fingerprint mismatch')
            require(all(r['evidence_kind'] == cert['evidence_kind'] for r in rows), 'basis evidence kind')
            require(cert.get('transitions') is None or canonical(cert['transitions']) == canonical(table),
                    'basis joint mismatch')
            require(cert['attempted_pairs'] == len(rows) and not cert['first_exclusion_counts'],
                    'complete basis partition mismatch')
            require(all(normalized_rule({k:r['original']['detection'][k] for k in
                    ('rule_id', 'threshold', 'comparator')}) == normalized_rule(cert['decision_rule']) for r in rows),
                    'basis rule mismatch')
        dependence.append(dict(stratum=dict(zip(STRATUM, key)), configuration_pairs=len(rows),
            distinct_original_source_texts=len({r['original']['source_sha256'] for r in rows}),
            declared_clusters=len({r['cluster_id'] for r in rows if type(r.get('cluster_id')) is str and r['cluster_id']}),
            unclustered_configurations=sum(not (type(r.get('cluster_id')) is str and r['cluster_id']) for r in rows),
            independent_program_count_established=False))
    require(Path(basis).read_bytes() == raw, 'basis changed during reference check')
    return dependence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    parser.add_argument('--basis', type=Path)
    args = parser.parse_args()
    bundle = load(args.bundle.read_text(encoding='utf-8-sig'))
    results = check_bundle(bundle)
    dependence = check_complete_basis(bundle, args.basis) if args.basis else None
    print(json.dumps(dict(scope='bounded independent arithmetic/status and optional complete-pair recount; not full schema/prose/origin verification',
                          checks=results, dependence=dependence), sort_keys=True))


if __name__ == '__main__':
    main()
