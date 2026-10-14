"""Predeclared owner-authored Python carrier fixture, not a watermark method.

All eight three-bit states are included. The neutral review annotation does not
modify the carrier, and the detector/threshold are fixed before measurement.
No public task, upstream watermark, attack helper or external target is loaded.
"""
import ast
import hashlib
import json
import re

DOMAIN = (-2, -1, 0, 1, 2)
BODY = "def step(x):\n    return x + 1\n"
ANNOTATION = "# Reviewed finite-domain arithmetic fixture.\n"
CONTRACT_SPEC = {"entrypoint": "step", "inputs": list(DOMAIN),
                 "expected": [x + 1 for x in DOMAIN], "scope": "finite integer domain only"}
CONTRACT_ID = "owned-xplusone:" + hashlib.sha256(
    json.dumps(CONTRACT_SPEC, sort_keys=True).encode()).hexdigest()
DETECTOR_SPEC = {"carrier": "three-bit comment", "score": "number of one bits",
                 "threshold": 2, "comparator": "ge", "key": None}


def source_for(bits, edited=False):
    if type(bits) is not int or bits not in range(8):
        raise ValueError("Only the eight predeclared owned carrier states are permitted")
    return "# owned-finite-carrier: " + format(bits, "03b") + "\n" + BODY + (
        ANNOTATION if edited else "")


def measured_validation(source):
    """Execute only an exact member of this module's finite approved source set."""
    approved = {source_for(bits, edited) for bits in range(8) for edited in (False, True)}
    if source not in approved:
        raise ValueError("Refuse unknown source; this is not an arbitrary-code runner")
    if ast.dump(ast.parse(source)) != ast.dump(ast.parse(BODY)):
        raise ValueError("Owned executable body changed")
    namespace = {}
    exec(compile(source, "<owned-finite-carrier>", "exec"), {"__builtins__": {}}, namespace)
    calls = [{"input": x, "expected": x + 1, "actual": namespace["step"](x)} for x in DOMAIN]
    return all(call["actual"] == call["expected"] for call in calls), calls


def measured_detection(source):
    match = re.fullmatch(r"# owned-finite-carrier: ([01]{3})", source.splitlines()[0])
    if not match:
        raise ValueError("Missing the declared owned carrier")
    score = match.group(1).count("1")
    return score, score >= DETECTOR_SPEC["threshold"]
