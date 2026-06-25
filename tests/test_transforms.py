from __future__ import annotations

from posteditbench.transforms.registry import build_transform_bundle
from posteditbench.utils import normalize_whitespace, strip_comments


def test_strip_comments_transform():
    source = """
# headline
def demo(value):
    return value  # trailing
""".strip()
    transform = build_transform_bundle("strip_comments")
    mutated = transform.apply(source)

    assert "#" not in mutated
    assert "headline" not in mutated
    assert mutated == normalize_whitespace(strip_comments(source))


def test_canonicalize_text_transform():
    source = """
// heading
def Demo(Value):
    return Value
""".strip()
    transform = build_transform_bundle("canonicalize_text")
    mutated = transform.apply(source)

    assert mutated == "def demo(value):\n    return value"


def test_strip_comments_fails_closed_on_malformed_python():
    source = """
# headline
def demo(value):
    if value:
        return value
      return 0  # trailing
""".strip()

    assert strip_comments(source, language="python") == source
