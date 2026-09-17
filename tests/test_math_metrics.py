"""Scientific grading regressions, including false-positive prevention."""

import json
import signal
import time

import pytest

pytest.importorskip("math_verify", reason="math graders run in the isolated .venv-math-eval environment")

from baseline_common import math_metrics as metrics


@pytest.mark.parametrize("prediction,reference", [
    (r"Reasoning contains 99. Final answer: 1,200", "compute...\n#### 1200"),
    (r"\boxed{\$1,234.50}", "#### 1234.5"),
    (r"\boxed{-0.5}", "#### -0.50"),
    (r"\boxed{\frac{3}{4}}", "#### 0.75"),
    ("3/4", "0.75"),
    (r"\boxed{\sqrt{4}}", "2"),
    ("#### -1,000", "-1000"),
])
def test_gsm8k_exact_numeric_equivalence(prediction, reference):
    result = metrics.math_score(prediction, reference, "gsm8k")
    assert result["correct"]
    assert result["parse_error"] is None


@pytest.mark.parametrize("prediction,reference", [
    (r"\boxed{0.333333}", "#### 0.3333333"),
    (r"\boxed{2}", "#### -2"),
    (r"\boxed{100}", "#### 1000"),
    (r"\boxed{1,2}", "#### 12"),
    (r"\boxed{0/0}", "#### 0"),
    (r"\boxed{\infty}", "#### 1"),
    (r"\boxed{x=2}", "#### 2"),
])
def test_gsm8k_wrong_or_non_scalar_answers(prediction, reference):
    assert not metrics.math_score(prediction, reference, "gsm8k")["correct"]


@pytest.mark.parametrize("prediction,reference", [
    (r"\boxed{\frac{2}{4}}", r"Solution: $\boxed{\frac{1}{2}}$."),
    (r"\boxed{\sqrt{8}/2}", r"\sqrt{2}"),
    (r"\boxed{(x+1)^2}", r"x^2+2x+1"),
    (r"\boxed{\{3,1,2\}}", r"\{1,2,3\}"),
    (r"\boxed{(-\infty,2]}", r"(-\infty,2]"),
    (r"\boxed{(1,2,3)}", r"(1,2,3)"),
    (r"\boxed{i^2}", "-1"),
    (r"\boxed{\frac{1}{3}}", "0.333333"),
    (r"\boxed{\frac12}", r"\frac{1}{2}"),
    (r"\boxed{\sqrt2}", r"\sqrt{2}"),
    (r"\boxed{\frac{12}{5525}}", r"\frac{12}{5,\!525}"),
    (r"\boxed{\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}}",
     r"\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}"),
    (r"\boxed{42_7}", "30"),
    (r"\boxed{0.0011_{2}}", r"\frac{3}{16}"),
    (r"\boxed{50\%}", "0.5"),
    (r"\boxed{70,110}", r"\{70,110\}"),
])
def test_math_symbolic_equivalence(prediction, reference):
    result = metrics.math_score(prediction, reference, "math")
    assert result["correct"], result


@pytest.mark.parametrize("prediction,reference", [
    (r"\boxed{x+1}", "y+1"),
    (r"\boxed{x}", "X"),
    (r"\boxed{(2,1,3)}", "(1,2,3)"),
    (r"\boxed{\{1,2\}}", r"\{1,2,3\}"),
    (r"\boxed{1}", "1,2"),
    (r"\boxed{(-\infty,2)}", r"(-\infty,2]"),
    (r"\boxed{\frac{1}{}}", "1"),
])
def test_math_wrong_or_partial_answers(prediction, reference):
    result = metrics.math_score(prediction, reference, "math")
    assert not result["correct"], result


@pytest.mark.parametrize("prediction", [
    "We obtain 42 at an intermediate step, but cannot determine the answer.",
    "We obtain 42 at an intermediate step. So 41.",
    r"The useful quantity is $42$.",
    r"\boxed{41}",
    "\\boxed{42}\nFinal answer: 41",
    r"\boxed{41} or \boxed{42}",
    r"\boxed{42} and \boxed{41}",
    r"\boxed{41,42}",
    r"\boxed{42} then the final answer is \boxed{41",
    "",
])
def test_does_not_reward_correct_number_inside_rationale(prediction):
    result = metrics.math_score(prediction, "42", "math")
    assert not result["correct"], result


def test_bad_reference_is_fatal_not_scored_as_wrong():
    with pytest.raises(ValueError):
        metrics.math_score(r"\boxed{42}", r"solution \boxed{", "math")


def test_no_untrusted_python_evaluation(tmp_path):
    target = tmp_path / "must_not_exist"
    payload = "__import__('pathlib').Path('" + str(target) + "').write_text('bad')"
    result = metrics.math_score(r"\boxed{" + payload + "}", "42", "math")
    assert not result["correct"]
    assert result["parse_error"] == "unsafe_answer_syntax"
    assert not target.exists()


def test_parser_failure_and_extraction_failure_are_distinct():
    unanchored = metrics.math_score("I cannot solve this", "42", "math")
    malformed = metrics.math_score(r"\boxed{\frac{42}{}}", "42", "math")
    assert unanchored["extraction_failed"]
    assert not malformed["extraction_failed"]
    assert malformed["parse_error"]


def test_symbolic_comparison_has_a_real_deadline(monkeypatch):
    original = metrics._backend()
    metrics.validate_reference("8", "math")
    def slow_verify(*args, **kwargs):
        time.sleep(0.5)
        return True
    monkeypatch.setattr(metrics, "_backend", lambda: (*original[:3], slow_verify, *original[4:]))
    real_deadline = metrics._deadline
    monkeypatch.setattr(metrics, "_deadline", lambda: real_deadline(0.03))
    started = time.monotonic()
    result = metrics.math_score(r"\boxed{8}", "8", "math")
    assert time.monotonic() - started < 0.4
    assert not result["correct"]
    assert result["parse_error"] == "verification_timeout"


def test_deadline_restores_previous_signal_handler():
    previous = signal.getsignal(signal.SIGALRM)
    with metrics._deadline(0.1):
        pass
    assert signal.getsignal(signal.SIGALRM) == previous
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


def test_metric_manifest_pins_rules_and_dependencies():
    manifest = metrics.metric_definition("math")
    assert json.loads(json.dumps(manifest)) == manifest
    assert manifest["dependencies"] == metrics.DEPENDENCIES
    assert manifest["invalid_references"] == "fatal preflight error"
    assert not manifest["unanchored_rationale_fallback"]


@pytest.mark.parametrize("prediction,reference", [
    (r"\boxed{Friday}", r"\text{Friday}"),
    (r"\boxed{C}", r"\text{(C)}"),
    (r"\boxed{(\text{E})}", r"(\text{E})"),
    (r"\boxed{G, E, B}", r"\text{B, E, G}"),
    (r"\boxed{18}", r"18\text{ ways.}"),
    (r"\boxed{15}", r"15\mbox{ cm}^2"),
    (r"\boxed{\{12,13\}}", r"12\text{ and }13"),
    ("**Final answer:** 42", "42"),
])
def test_literal_and_presentation_rules(prediction, reference):
    result = metrics.math_score(prediction, reference, "math")
    assert result["correct"], result


@pytest.mark.parametrize("prediction,reference", [
    (r"\boxed{yadirF}", r"\text{Friday}"),
    (r"\boxed{Friday, maybe Saturday}", r"\text{Friday}"),
    (r"\boxed{B}", r"\text{B, E, G}"),
    (r"\boxed{50}", r"50\%"),
    (r"\boxed{t}", "1"),
    (r"\boxed{42}", "42_7"),
    (r"\boxed{999_2}", "999"),
])
def test_literal_units_and_percent_do_not_change_mathematical_meaning(prediction, reference):
    assert not metrics.math_score(prediction, reference, "math")["correct"]


@pytest.mark.parametrize("answer", [
    r"9^{9^9}", r"9^9^9", r"(9^9)^9", r"9^{9}^{9}",
    r"9^{9999*9999}", r"9^{9999+9999}", r"9^{9999\cdot9999}",
    r"9^{\frac{9999}{0.00001}}", r"9^\frac{9999}{0.00001}",
    r"9^{\sqrt{999999999999}}", r"9^{9999!}", r"(9^9)!",
    r"1000000000!", r"(9999*9999)!", r"(2004!)!", r"2004!!",
    r"(2000+9999)!", r"2^{10001}", r"2^{1e10000}",
    r"999999999999999999999999999999999999999999999999999999999999^10000",
    r"\frac{999999999999}{1}!", r"\sqrt{999999999999}!",
    r"\binom{1000000000}{500000000}", r"\binom{9^9}{5}",
    r"\binom{10000}{5000}!",
])
def test_expensive_expressions_never_reach_symbolic_backend(answer, monkeypatch):
    def forbidden_backend():
        pytest.fail("An unsafe operand reached the symbolic backend")
    metrics._parse_answer.cache_clear()
    monkeypatch.setattr(metrics, "_backend", forbidden_backend)
    with pytest.raises(metrics.AnswerParseError):
        metrics._parse_answer(answer, "math")


@pytest.mark.parametrize("answer", [
    r"\frac{1}{2004!}", r"2^{2007}", r"11^{96}", r"x^2+2x+1",
    r"(x+1)^2", r"2^{2+3}", r"2^{\frac{1}{2}}", r"2^\frac12",
    r"2^{n+1}", r"2^{2n}", r"(2000+4)!", r"60^\circ",
    r"\binom{20}{10}", r"\frac{4}{2}!",
])
def test_guard_allows_bounded_ordinary_mathematics(answer):
    # Exercise only the guard: a 2004! value is not allocated by this test.
    metrics._guard_answer(answer)


def test_complexity_error_is_recorded_as_incorrect_prediction(monkeypatch):
    metrics._parse_answer("42", "math")
    original = metrics._backend()
    def forbidden_parser(*args, **kwargs):
        pytest.fail("Dangerous prediction reached ANTLR")
    monkeypatch.setattr(metrics, "_backend", lambda: (forbidden_parser, *original[1:]))
    result = metrics.math_score(r"\boxed{9^{9^9}}", "42", "math")
    assert not result["correct"]
    assert result["parse_error"] == "nested_power_or_factorial"
    assert not result["extraction_failed"]
