from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from pathlib import Path
from typing import Callable

from ..benchmarks import load_benchmark_corpus
from ..models import BenchmarkExample


@dataclass(frozen=True, slots=True)
class TemplateSpec:
    name: str
    prompt: str
    builder: Callable[[str], str]
    tests: tuple[str, ...]


def _factorial(name: str) -> str:
    return f"""
def {name}(n):
    if n < 2:
        return 1
    result = 1
    for value in range(2, n + 1):
        result *= value
    return result
""".strip()


def _clamp(name: str) -> str:
    return f"""
def {name}(value, lower, upper):
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value
""".strip()


def _sum_positive(name: str) -> str:
    return f"""
def {name}(items):
    total = 0
    for item in items:
        if item > 0:
            total += item
    return total
""".strip()


def _normalize_scores(name: str) -> str:
    return f"""
def {name}(scores):
    if not scores:
        return []
    total = float(sum(scores))
    if total == 0:
        return [0.0 for _ in scores]
    return [score / total for score in scores]
""".strip()


def _is_prime(name: str) -> str:
    return f"""
def {name}(number):
    if number < 2:
        return False
    for divisor in range(2, int(number ** 0.5) + 1):
        if number % divisor == 0:
            return False
    return True
""".strip()


def _max_of_two(name: str) -> str:
    return f"""
def {name}(left, right):
    if left >= right:
        return left
    return right
""".strip()


TEMPLATES: tuple[TemplateSpec, ...] = (
    TemplateSpec(
        name="factorial",
        prompt="Write a function that returns the factorial of a non-negative integer.",
        builder=_factorial,
        tests=("assert factorial(5) == 120", "assert factorial(0) == 1"),
    ),
    TemplateSpec(
        name="clamp",
        prompt="Write a function that clamps a value between lower and upper bounds.",
        builder=_clamp,
        tests=("assert clamp(3, 1, 5) == 3", "assert clamp(-1, 0, 4) == 0"),
    ),
    TemplateSpec(
        name="sum_positive",
        prompt="Write a function that sums only the positive numbers from a list.",
        builder=_sum_positive,
        tests=("assert sum_positive([1, -2, 3]) == 4", "assert sum_positive([]) == 0"),
    ),
    TemplateSpec(
        name="normalize_scores",
        prompt="Write a function that normalizes scores so they sum to 1.",
        builder=_normalize_scores,
        tests=("assert normalize_scores([1, 1]) == [0.5, 0.5]",),
    ),
    TemplateSpec(
        name="is_prime",
        prompt="Write a function that decides whether an integer is prime.",
        builder=_is_prime,
        tests=("assert is_prime(7) is True", "assert is_prime(8) is False"),
    ),
    TemplateSpec(
        name="max_of_two",
        prompt="Write a function that returns the larger of two values.",
        builder=_max_of_two,
        tests=("assert max_of_two(3, 4) == 4", "assert max_of_two(10, 2) == 10"),
    ),
)


def _rewrite_tests(tests: tuple[str, ...], template_name: str, function_name: str) -> tuple[str, ...]:
    return tuple(test.replace(template_name, function_name) for test in tests)


def _normalize_languages(language: str | Sequence[str] | None) -> list[str]:
    if language is None:
        return []
    if isinstance(language, str):
        return [language] if language else []
    return [str(item) for item in language if str(item)]


def _synthesize_from_templates(
    count: int,
    *,
    seed: int,
    language: str | Sequence[str] | None,
    prompt_prefix: str,
) -> list[BenchmarkExample]:
    languages = _normalize_languages(language) or ["python"]
    examples: list[BenchmarkExample] = []
    variant_pool = ["base", "robust", "adversarial"]
    for index in range(count):
        template = TEMPLATES[index % len(TEMPLATES)]
        function_name = f"{template.name}_{index}"
        prompt = f"{prompt_prefix}{template.prompt}"
        example_language = languages[(seed + index) % len(languages)]
        reference_tests = _rewrite_tests(template.tests, template.name, function_name)
        examples.append(
            BenchmarkExample(
                example_id=f"example-{index:03d}",
                language=example_language,
                prompt=prompt,
                reference_solution=template.builder(function_name),
                reference_tests=reference_tests,
                execution_tests=reference_tests if example_language.lower() == "python" else (),
                metadata={
                    "template": template.name,
                    "index": index,
                    "seed": seed,
                    "variant": variant_pool[(seed + index) % len(variant_pool)],
                },
            )
        )
    return examples


def generate_corpus(
    count: int | None,
    *,
    seed: int = 7,
    language: str | Sequence[str] | None = None,
    include_reference_kinds: Sequence[str] | None = None,
    prompt_prefix: str = "",
    benchmark_path: str | Path | None = None,
) -> list[BenchmarkExample]:
    if benchmark_path is not None:
        return load_benchmark_corpus(
            benchmark_path,
            count=count,
            seed=seed,
            languages=language,
            include_reference_kinds=include_reference_kinds,
            prompt_prefix=prompt_prefix,
        )
    return _synthesize_from_templates(count or 8, seed=seed, language=language, prompt_prefix=prompt_prefix)
