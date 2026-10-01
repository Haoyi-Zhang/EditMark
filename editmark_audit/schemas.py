"""Local structural schemas for evidence documents.

The Python parsers remain authoritative for cross-field consistency, exact
integer semantics, duplicate identities, and arithmetic partitions.
"""
from __future__ import annotations
from .aggregates import SCHEMA as COUNT_SCHEMA, STRATA, EXCLUSIONS
from .constants import CERTIFICATE_BUNDLE_SCHEMA, CERTIFICATE_SCHEMA, STRICT_CONTRACT
from .records import SCHEMA as PAIR_SCHEMA, KINDS, COMPARATORS


def object_schema(properties, required=(), nullable=False):
    return {
        'type': ['object', 'null'] if nullable else 'object',
        'properties': properties,
        'required': list(required),
        'additionalProperties': False,
    }


def pair_schema():
    identity = {'type': 'string', 'minLength': 1}
    optional_text = {'type': ['string', 'null']}
    flag = {'type': ['boolean', 'null']}
    number = {'type': ['number', 'null']}
    validation = object_schema(
        {'available': flag, 'passed': flag, 'contract_id': optional_text}, nullable=True)
    detection = object_schema(
        {
            'available': flag,
            'detected': flag,
            'score': number,
            'threshold': number,
            'comparator': {'enum': [None, '', *COMPARATORS]},
            'rule_id': optional_text,
        },
        nullable=True,
    )
    program = object_schema(
        {
            'source': {'type': 'string'},
            'source_sha256': {
                'anyOf': [
                    {'type': 'null'}, {'const': ''},
                    {'type': 'string', 'pattern': '^[a-fA-F0-9]{64}$'},
                ]
            },
            'validation': validation,
            'detection': detection,
        },
        nullable=True,
    )
    ids = (
        'evidence_kind', 'experiment_id', 'method', 'model', 'source_group',
        'language', 'sample_id', 'replicate_id', 'edit', 'variant_id',
    )
    properties = {key: identity.copy() for key in ids}
    properties.update(
        schema_version={'const': PAIR_SCHEMA},
        population={'const': 'watermarked_positive'},
        evidence_kind={'enum': list(KINDS)},
        cluster_id=optional_text,
        supported=flag,
        changed=flag,
        original=program,
        edited=program,
    )
    result = object_schema(properties, ('schema_version', 'population', *ids))
    return {
        '$schema': 'https://json-schema.org/draft/2020-12/schema',
        'title': 'EditMark normalized positive pair',
        '$comment': (
            'Run Pair.parse after structural validation. Unknown observations remain null. '
            'Runtime checks enforce finite exact inputs, source hashes, duplicate identity, '
            'and cross-field consistency.'
        ),
        **result,
    }


def count_schema():
    integer = {'type': 'integer', 'minimum': 0}
    ids = {key: {'type': 'string', 'minLength': 1} for key in STRATA}
    transitions = object_schema(
        {key: integer for key in ('n00', 'n01', 'n10', 'n11')},
        ('n00', 'n01', 'n10', 'n11'),
        nullable=True,
    )
    properties = {
        'schema_version': {'const': COUNT_SCHEMA},
        'population': {'const': 'watermarked_positive'},
        'evidence_kind': {'enum': list(KINDS)},
        'analysis_contract': {'const': STRICT_CONTRACT},
        'cohort_sha256': {'type': 'string', 'pattern': '^[a-fA-F0-9]{64}$'},
        'stratum': object_schema(ids, STRATA),
        'decision_rule': object_schema(
            {
                'rule_id': {'type': 'string', 'minLength': 1},
                'threshold': {'type': 'number'},
                'comparator': {'enum': list(COMPARATORS)},
            },
            ('rule_id', 'threshold', 'comparator'),
        ),
        'counts': object_schema(
            {key: integer for key in ('n', 'before_positive', 'after_positive')},
            ('n', 'before_positive', 'after_positive'),
        ),
        'transitions': transitions,
        'attempted_pairs': integer,
        'first_exclusion_counts': object_schema({key: integer for key in EXCLUSIONS}),
    }
    return {
        '$schema': 'https://json-schema.org/draft/2020-12/schema',
        'title': 'EditMark exact fixed-cohort counts',
        '$comment': (
            'Runtime checks enforce non-Boolean integer counts, finite rules, exact partitions, '
            'and marginal/joint compatibility. Cohort membership is a supplied assertion.'
        ),
        **object_schema(properties, [key for key in properties if key != 'transitions']),
    }


def certificate_schema():
    fraction = object_schema(
        {
            'numerator': {'type': 'integer'},
            'denominator': {'type': 'integer', 'minimum': 1},
            'decimal': {'type': 'number'},
        },
        ('numerator', 'denominator', 'decimal'),
        nullable=True,
    )
    return {
        '$schema': 'https://json-schema.org/draft/2020-12/schema',
        'title': 'EditMark claim certificate',
        **object_schema(
            {
                'schema_version': {'const': CERTIFICATE_SCHEMA},
                'claim_id': {'type': 'string'},
                'claim_text': {'type': 'string'},
                'requested_claim': {'type': 'string'},
                'decision': {'enum': ['admissible', 'partially_admissible', 'not_admissible']},
                'identification': {'type': 'string'},
                'population': {'type': 'string'},
                'analysis_contract': {'type': 'string'},
                'cohort_sha256': {'type': 'string', 'pattern': '^[a-f0-9]{64}$'},
                'stratum': {'type': 'object'},
                'decision_rule': {'type': 'object'},
                'counts': {'type': 'object'},
                'transitions': {
                    'type': ['object', 'null'],
                    'properties': {
                        key: {'type': 'integer', 'minimum': 0}
                        for key in ('n00', 'n01', 'n10', 'n11')
                    },
                    'additionalProperties': False,
                },
                'attempted_pairs': {'type': 'integer', 'minimum': 0},
                'first_exclusion_counts': {'type': 'object'},
                'analysis_status': {'type': 'string'},
                'membership_verified': {'type': 'boolean'},
                'claim_text_verified': {'const': False},
                'requirements': {'type': 'object'},
                'prohibited_interpretations': {'type': 'array', 'items': {'type': 'string'}},
                'estimand': {'type': 'string'},
                'estimate': fraction,
                'bounds': {'type': 'object'},
                'supported_statement': {'type': 'string'},
                'evidence_kind': {'type': ['string', 'null']},
                'synthetic_warning': {'type': ['string', 'null']},
                'certificate_sha256': {'type': 'string', 'pattern': '^[a-f0-9]{64}$'},
            },
            (
                'schema_version', 'requested_claim', 'decision', 'identification',
                'population', 'analysis_contract', 'cohort_sha256', 'stratum',
                'decision_rule', 'counts', 'transitions', 'attempted_pairs',
                'first_exclusion_counts', 'analysis_status', 'membership_verified',
                'claim_text_verified', 'requirements', 'prohibited_interpretations', 'supported_statement',
                'certificate_sha256',
            ),
        ),
    }



def certificate_bundle_schema():
    return {
        '$schema': 'https://json-schema.org/draft/2020-12/schema',
        'title': 'EditMark claim-certificate bundle',
        **object_schema(
            {
                'schema_version': {'const': CERTIFICATE_BUNDLE_SCHEMA},
                'input_file': object_schema(
                    {
                        'name': {'type': 'string', 'minLength': 1},
                        'sha256': {'type': 'string', 'pattern': '^[a-f0-9]{64}$'},
                    },
                    ('name', 'sha256'),
                ),
                'requested_claim': {'type': 'string', 'minLength': 1},
                'certificates': {
                    'type': 'array',
                    'minItems': 1,
                    'items': certificate_schema(),
                },
            },
            ('schema_version', 'input_file', 'requested_claim', 'certificates'),
        ),
    }
