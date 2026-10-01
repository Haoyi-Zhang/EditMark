"""Shared identifiers for the evidence contract.

These strings identify document types, not experimental releases.  They are
kept in one module so the parser, schemas, command-line tools, and tests cannot
drift silently.
"""

PAIR_SCHEMA = "editmark-pair"
COUNT_SCHEMA = "editmark-cohort-counts"
PAIR_REPORT_SCHEMA = "editmark-pair-report"
COUNT_ANALYSIS_SCHEMA = "editmark-count-analysis"
COUNT_REPORT_SCHEMA = "editmark-count-report"
CERTIFICATE_SCHEMA = "editmark-claim-certificate"
CERTIFICATE_BUNDLE_SCHEMA = "editmark-claim-certificate-bundle"
STRICT_CONTRACT = "same-rule-changed-pairs"
