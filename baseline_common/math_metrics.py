"""Final-answer correctness for the fixed MetaMathQA campaign.

The ANTLR LaTeX parser is called on *only* an explicitly extracted final
answer, never on a rationale.  Its grammar consumes the entire expression;
we deliberately do not use Math-Verify's free-text extraction/fallbacks or
SymPy's eval-based ``parse_expr``.  See docs/math_grading.md for the protocol.
"""

from __future__ import annotations

from contextlib import contextmanager
from fractions import Fraction
from functools import lru_cache
import importlib.metadata
import hashlib
import re
import signal
import threading
import time


GRADER_VERSION = "metamath-final-v2"
TIMEOUT_SECONDS = 5.0
MAX_ANSWER_CHARS = 4096
MAX_COMPLETION_CHARS = 131072
MAX_POWER_EXPONENT = 10000
MAX_FACTORIAL_ARGUMENT = 10000
MAX_ARITHMETIC_BITS = 256
MAX_POWER_BITS = 100000
DEPENDENCIES = {
    "math-verify": "0.9.0", "latex2sympy2_extended": "1.11.0",
    "antlr4-python3-runtime": "4.13.2", "sympy": "1.13.1",
}
_BOX = re.compile(r"\\(?:boxed|fbox)\s*\{")
_FINAL = re.compile(r"(?im)(?:\*\*)?\b(?:the\s+)?final\s+answer(?:\*\*)?\s*(?:is\b|[:=])\s*(?:\*\*)?\s*")
_HASH = re.compile(r"(?m)^\s*####\s*")
_NUMBER = r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_SCALAR = re.compile(rf"{_NUMBER}(?:\s*/\s*{_NUMBER})?\Z")
# These are gold-annotation interpretations, frozen by complete solution
# hashes before any model output exists. Source row numbers are zero-based.
REFERENCE_OVERRIDES = {
    "a580deb03cc5bffb4fbc38e2310cace79fd202a62e95c3cde8997e7dcc149079":
        {"row": 333, "answer": "-3", "alternatives": ["-3", "3"],
         "reason": "gold explicitly accepts both signs, despite the derivation giving -3; preserve annotated acceptance"},
    "ade85c49c87508e1d285f60e9d88c018ea28a4d8d42f27840ffd63f5e6e2ecd0":
        {"row": 414, "answer": r"\{-2,1\}", "reason": "question asks all solutions; gold has two separate boxes"},
    "17dae234dd0a00d2ecf17bc0106c1548b98b2b1ff6b448592211b3d8160fd3a1":
        {"row": 2703, "answer": r"\{3,\frac{2}{5}\}", "reason": "question asks all solutions; gold has two separate boxes"},
    "91702eef2d1f016f99f4a684d90a595eba49d7c0f78c1806654f7d02d4cd7881":
        {"row": 2716, "answer": "(10,0)", "alternatives": ["(10,0)", "(0,0)"],
         "reason": "question explicitly requests only one focus and permits either point"},
    "7848474323ef8f28fb8d4c1eec20681db9f915b316c000f78901caeb4632cfd7":
        {"row": 3628, "answer": "x+11", "reason": "two boxes are equivalent polynomial forms"},
    "924fa52d28956659b60f90b8953ad264f3db66b5c76d5f766a414ac0606c3d1a":
        {"row": 4258, "answer": "12x-34", "reason": "two boxes are equivalent polynomial forms"},
    "136a93ace73513de76f1d33b6f618640187184d55d91c9eeb04e5f4a53d2558e":
        {"row": 4860, "answer": r"\{\frac{\pi}{4},\frac{5\pi}{4}\}",
         "reason": "question asks all solutions; gold has two separate boxes"},
    "75f3ec17279d32db62e0544c71f42e37cb19b7e19b8fc70f5bf662aefb52a1a6":
        {"row": 2152, "answer": r"\{5-10i,11,-3+6i\}", "reason": "question asks all three possible parallelogram vertices"},
    "d69fdc848eff7184e8a33ec4a74dfdcdb10082df76caf241ae39da8d58a640dc":
        {"row": 2549, "answer": r"\{-\frac{3}{2},-\frac{3}{4}\}", "reason": "question asks both solutions, boxed separately"},
    "49795cf0fe2f5835a84a9156c68e89eb1a072ca316fa968167dbf8314d70145b":
        {"row": 2656, "answer": r"\{-10879,10879\}", "reason": "question asks both values of b, boxed separately"},
    "bc9d573809020b4fa3107d06229d59137aa954c03b08515fa3079747cf7cd81f":
        {"row": 2912, "answer": r"(\frac{3}{2},-\frac{5}{2})",
         "alternatives": [r"(\frac{3}{2},-\frac{5}{2})", r"(-\frac{5}{2},-\frac{5}{2})"],
         "reason": "question explicitly asks for either one of two vertices"},
    "b3fe46fe3f7c2672d7454489d2e941a87da7e99180dd2afa00b8178d80083ac7":
        {"row": 2936, "answer": "(3,-5)", "alternatives": ["(3,-5)", "(0,0)"],
         "reason": "gold explicitly accepts either of two points on the odd function"},
    "c899a3a8ddaa8afb591e8587112b529b8581e2329fbdb3d8f9cc558229fdc186":
        {"row": 4060, "answer": "6",
         "reason": "box denotes divisor-count operator; gold explicitly leaves final answer 6 unboxed"},
    "6f6e4bed7d6edd64ab43e4d9388bb92e53c68f38ea6c18d68606e174cd832d12":
        {"row": 4559, "answer": r"\{-1,2\}", "reason": "question asks both possible values of a"},
    "5d24c516df211881fea26778f937d01dc2dbe8d646fd9182e37825e3a53e8a3b":
        {"row": 4562, "answer": r"\{30,45,105\}", "reason": "question asks all three angles in degrees, boxed separately"},
    "edf899f558a5289a42f4772082fbe6b0301774b591aeeae7e4f4655cc4c225a7":
        {"row": 4791, "answer": r"\{70,110\}",
         "reason": "question asks both solutions 70 and 110; unbraced gold comma is not a thousands separator"},
}
_UNIT_WORDS = {
    "student", "students", "pound", "pounds", "teacher", "teachers", "dollar", "dollars",
    "cm", "feet", "children tickets", "ft", "ways", "ways.", "meal", "meals", "degrees",
    "inch", "inches", "unit", "units", "edge", "edges", "square units", "square inches",
    "m", "km", "integer", "integers", "digit", "digits", "minute", "minutes", "cent", "cents",
    "square feet", "gm", "multiple", "multiples", "mph", "point", "points", "hour", "hours",
    "meter", "meters", "square meters",
}


class AnswerParseError(ValueError):
    """An answer is absent, ambiguous, malformed or unsupported."""


class _GradeTimeout(BaseException):
    # Upstream broad ``except Exception`` must not swallow our deadline.
    pass


def _benchmark(benchmark):
    if benchmark not in {"gsm8k", "math"}:
        raise ValueError("Math benchmark must be 'gsm8k' or 'math'")
    return benchmark


@lru_cache(maxsize=1)
def _backend():
    try:
        versions = {name: importlib.metadata.version(name) for name in DEPENDENCIES}
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("Math grading requires the pinned .venv-math-eval environment") from exc
    if versions != DEPENDENCIES:
        raise RuntimeError(f"Math grader dependency versions differ: {versions}; expected {DEPENDENCIES}")
    from latex2sympy2_extended import latex2sympy
    from latex2sympy2_extended.latex2sympy2 import ConversionConfig
    from latex2sympy2_extended.math_normalization import NormalizationConfig
    from math_verify import verify
    from math_verify import grader as backend_grader
    from math_verify.grader import should_treat_as_complex
    import sympy
    # The public verifier warns once when its own alarm is disabled. Our
    # enclosing deadline supplies exactly the protection that warning asks
    # the caller to provide, and also reports timeouts without swallowing them.
    backend_grader.TIMEOUT_WARNING_SHOWN = True
    return latex2sympy, ConversionConfig, NormalizationConfig, verify, should_treat_as_complex, sympy


@contextmanager
def _deadline(seconds=TIMEOUT_SECONDS):
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("Math grading must run in a process's main thread for bounded symbolic computation")
    def expired(signum, frame):
        raise _GradeTimeout()
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    started = time.monotonic()
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, min(seconds, previous_timer[0]) if previous_timer[0] else seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0]:
            signal.setitimer(signal.ITIMER_REAL, max(1e-6, previous_timer[0] - (time.monotonic() - started)), previous_timer[1])


def _unwrap(text):
    text = text.strip().replace("−", "-").replace("\u2212", "-")
    for left, right in (("$$", "$$"), ("$", "$"), (r"\(", r"\)"), (r"\[", r"\]")):
        if text.startswith(left) and text.endswith(right) and len(text) > len(left) + len(right):
            text = text[len(left):-len(right)].strip()
            break
    return text


def _boxes(text):
    boxes = []
    for match in _BOX.finditer(text):
        if boxes and match.start() < boxes[-1][1]:
            continue
        start, depth = match.end(), 1
        end = start
        while end < len(text) and depth:
            char = text[end]
            # \{ and \} are literal set delimiters, not LaTeX argument braces.
            escaped = end > 0 and text[end - 1] == "\\"
            if not escaped:
                depth += (char == "{") - (char == "}")
            end += 1
        if depth:
            boxes.append((match.start(), len(text), None))
        else:
            boxes.append((match.start(), end, text[start:end - 1].strip()))
    return boxes


def extract_final_answer(text, *, reference=False, benchmark="math"):
    """Return final-answer text, rejecting a missing anchor in model prose.

    MATH gold may also be an already extracted answer; GSM8K gold may be a
    numeric answer or its native worked solution ending in ``#### number``.
    """
    _benchmark(benchmark)
    if not isinstance(text, str) or not text.strip():
        raise AnswerParseError("empty_answer")
    if len(text) > MAX_COMPLETION_CHARS:
        raise AnswerParseError("completion_too_long")
    boxes = _boxes(text)
    anchors = [(match.start(), match.end(), "final") for match in _FINAL.finditer(text)]
    anchors.extend((match.start(), match.end(), "hash") for match in _HASH.finditer(text))
    last_anchor = max(anchors, default=(-1, -1, None))
    if boxes and (reference or boxes[-1][0] >= last_anchor[0]):
        start, end, answer = boxes[-1]
        if answer is None:
            raise AnswerParseError("unclosed_final_box")
        # Adjacent separate boxes can be a multi-answer set. They must not
        # silently reduce to the last element (which inflates correctness).
        if len(boxes) > 1:
            separator = text[boxes[-2][1]:start].strip().strip("$").strip()
            if re.fullmatch(r"(?:,|and|or|,\s*(?:and|or))", separator, flags=re.I):
                raise AnswerParseError("multiple_adjacent_final_boxes_use_one_box")
    elif last_anchor[0] >= 0:
        answer = text[last_anchor[1]:].strip()
        # An explicit final-answer line must contain only its mathematical
        # answer; all following nonempty lines are rejected, not searched.
        answer = answer.rstrip(".").strip()
    else:
        answer = _unwrap(text)
        if not reference and not _SCALAR.fullmatch(answer):
            raise AnswerParseError("missing_final_answer_anchor")
    answer = _unwrap(answer)
    if not answer:
        raise AnswerParseError("empty_final_answer")
    if len(answer) > MAX_ANSWER_CHARS:
        raise AnswerParseError("final_answer_too_long")
    return answer


def _guard_answer(answer):
    # No Python names, quotes, backticks, or TeX external actions enter even
    # the grammar parser. This is also a bound against pathological input.
    if "__" in answer or "`" in answer or re.search(r"\\(?:input|include|write|openout|read|catcode|csname)\b", answer):
        raise AnswerParseError("unsafe_answer_syntax")
    if re.search(r"\b(?:import|exec|eval|lambda|subprocess)\b", answer):
        raise AnswerParseError("unsafe_answer_syntax")
    if re.search(r"\d{257,}", answer):
        raise AnswerParseError("numeric_literal_too_long")
    for exponent in re.findall(r"(?:[eE]|\^\s*\{?)\s*([+-]?\d+)", answer):
        if abs(int(exponent)) > MAX_POWER_EXPONENT:
            raise AnswerParseError("exponent_too_large")
    depth = 0
    for index, char in enumerate(answer):
        if index and answer[index - 1] == "\\":
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        if depth < 0 or depth > 64:
            raise AnswerParseError("invalid_brace_depth")
    if depth:
        raise AnswerParseError("unbalanced_answer_braces")
    _guard_expensive_operations(_expand_short_tex_arguments(answer))


_COMPLEXITY_TOKENS = re.compile(r"\\[A-Za-z]+|\\.|\d+(?:\.\d*)?(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?|[^\s]")


class _BoundedArithmetic:
    """Read small scalar operands without a symbolic parser or large powers.

    This deliberately supports only numbers, single symbols, fractions, and
    basic arithmetic. Unknown symbolic values propagate as None; every known
    intermediate is bounded. Powers and factorials are never evaluated here.
    """

    def __init__(self, tokens, limit, error):
        self.tokens, self.limit, self.error, self.index = tokens, limit, error, 0

    def checked(self, value):
        if value is not None and (abs(value) > self.limit or
                max(value.numerator.bit_length(), value.denominator.bit_length()) > MAX_ARITHMETIC_BITS):
            raise AnswerParseError(self.error)
        return value

    def atom(self):
        if self.index == len(self.tokens):
            raise AnswerParseError("unsupported_bounded_operand")
        token = self.tokens[self.index]
        self.index += 1
        if token in {"+", "-"}:
            value = self.atom()
            return -value if token == "-" and value is not None else value
        if token in {"(", "{", "["}:
            value = self.expression()
            if self.index == len(self.tokens) or self.tokens[self.index] != {"(": ")", "{": "}", "[": "]"}[token]:
                raise AnswerParseError("unsupported_bounded_operand")
            self.index += 1
            return value
        if token in {r"\frac", r"\dfrac", r"\tfrac", r"\cfrac"}:
            return self.combine(self.atom(), self.atom(), "/")
        if re.fullmatch(r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", token):
            # Existing lexical limits bound the string before Fraction runs.
            return self.checked(Fraction(token))
        if re.fullmatch(r"[A-Za-z]", token) or token in {r"\pi", r"\circ"}:
            return None
        raise AnswerParseError("unsupported_bounded_operand")

    def combine(self, left, right, operator):
        if operator == "/" and right == 0:
            raise AnswerParseError("undefined_bounded_operand")
        if left is None or right is None:
            return None
        if operator == "+":
            value = left + right
        elif operator == "-":
            value = left - right
        elif operator == "*":
            value = left * right
        else:
            value = left / right
        return self.checked(value)

    def term(self):
        value = self.atom()
        while self.index < len(self.tokens):
            token = self.tokens[self.index]
            if token in {"*", "/", r"\times", r"\cdot", r"\div"}:
                self.index += 1
                operator = "/" if token in {"/", r"\div"} else "*"
            elif token in {"+", "-", ")", "}", "]"}:
                break
            else:
                operator = "*"  # Ordinary implicit multiplication, e.g. 2n.
            value = self.combine(value, self.atom(), operator)
        return value

    def expression(self):
        value = self.term()
        while self.index < len(self.tokens) and self.tokens[self.index] in {"+", "-"}:
            operator = self.tokens[self.index]
            self.index += 1
            value = self.combine(value, self.term(), operator)
        return value

    def read(self):
        value = self.expression()
        if self.index != len(self.tokens):
            raise AnswerParseError("unsupported_bounded_operand")
        return value


def _guard_expensive_operations(answer):
    """Reject explosive syntax before ANTLR/SymPy can allocate its result.

    Signal deadlines cannot reliably interrupt a large C-level allocation.
    Group matching and the tiny bounded arithmetic grammar are consequently
    performed before importing or invoking the symbolic backend.
    """
    tokens = [token for token in _COMPLEXITY_TOKENS.findall(answer)
              if token not in {r"\left", r"\right", r"\!", r"\,", r"\;", r"\:", "\\ "}]
    if "^" not in tokens and "!" not in tokens and r"\binom" not in tokens:
        return
    if len(tokens) > 2048:
        raise AnswerParseError("too_many_answer_tokens")
    pairs, stack = {}, []
    for index, token in enumerate(tokens):
        if token in {"(", "{", "[", r"\{"}:
            stack.append(index)
            if len(stack) > 64:
                raise AnswerParseError("invalid_group_depth")
        elif token in {")", "}", "]", r"\}"}:
            if not stack:
                raise AnswerParseError("unbalanced_answer_groups")
            start = stack.pop()
            pairs[start], pairs[index] = index, start
    if stack:
        raise AnswerParseError("unbalanced_answer_groups")

    def forward(index):
        if index >= len(tokens):
            raise AnswerParseError("missing_bounded_operand")
        token = tokens[index]
        if token in {"+", "-"}:
            return forward(index + 1)
        if token in {"(", "{", "[", r"\{"}:
            return pairs[index] + 1
        if token in {r"\frac", r"\dfrac", r"\tfrac", r"\cfrac"}:
            return forward(forward(index + 1))
        return index + 1

    def backward(index):
        if index < 0:
            raise AnswerParseError("missing_bounded_operand")
        if tokens[index] in {")", "}", "]", r"\}"}:
            start = pairs[index]
            if start and tokens[start - 1] in {"}", ")"}:
                first = pairs[start - 1]
                if first and tokens[first - 1] in {r"\frac", r"\dfrac", r"\tfrac", r"\cfrac", r"\binom"}:
                    return first - 1
            if start and tokens[start - 1].startswith("\\") and tokens[start - 1] not in {r"\times", r"\cdot", r"\div"}:
                return start - 1
            return start
        if tokens[index] == "!":
            return backward(index - 1)
        return index

    for index, token in enumerate(tokens):
        if token == "^":
            end, start = forward(index + 1), backward(index - 1)
            exponent, base = tokens[index + 1:end], tokens[start:index]
            # Include unbraced chained powers, and grouped powers on either
            # side. Even small towers are outside this bounded protocol.
            if any(op in exponent + base for op in ("^", "!", r"\binom")) or (start and tokens[start - 1] == "^"):
                raise AnswerParseError("nested_power_or_factorial")
            value = _BoundedArithmetic(exponent, MAX_POWER_EXPONENT, "exponent_too_large").read()
            if value is not None and len(base) == 1 and re.fullmatch(r"\d+", base[0]):
                # No power is computed: bit length times exponent is a safe
                # conservative upper bound for the resulting integer size.
                if int(base[0]).bit_length() * abs(value) > MAX_POWER_BITS:
                    raise AnswerParseError("power_result_too_large")
        elif token == "!":
            start = backward(index - 1)
            argument = tokens[start:index]
            if any(op in argument for op in ("^", "!", r"\binom")) or (start and tokens[start - 1] == "^"):
                raise AnswerParseError("nested_power_or_factorial")
            value = _BoundedArithmetic(argument, MAX_FACTORIAL_ARGUMENT, "factorial_argument_too_large").read()
            if value is None or value < 0 or value.denominator != 1:
                raise AnswerParseError("unsupported_factorial_argument")
        elif token == r"\binom":
            middle = forward(index + 1)
            end = forward(middle)
            for argument in (tokens[index + 1:middle], tokens[middle:end]):
                if any(op in argument for op in ("^", "!", r"\binom")):
                    raise AnswerParseError("nested_power_or_factorial")
                value = _BoundedArithmetic(argument, MAX_FACTORIAL_ARGUMENT, "binomial_argument_too_large").read()
                if value is None or value < 0 or value.denominator != 1:
                    raise AnswerParseError("unsupported_binomial_argument")


def _literal_normalize(answer):
    """Normalize a complete textual answer, never a substring of prose."""
    answer = _unwrap(answer)
    if re.fullmatch(r"\(\s*\\(?:text|mbox|textrm)\s*\{[^{}]+\}\s*\)", answer):
        answer = answer[1:-1].strip()
    match = re.fullmatch(r"\\(?:text|mbox|textrm)\s*\{([^{}]+)\}", answer)
    if match:
        answer = match.group(1).strip()
    answer = " ".join(answer.split()).casefold()
    if re.fullmatch(r"\([a-z]\)", answer):
        answer = answer[1:-1]
    if re.fullmatch(r"[a-z](?:\s*,\s*[a-z])+", answer):
        answer = ",".join(sorted(part.strip() for part in answer.split(",")))
    if not re.fullmatch(r"[\w\s:.,()]+", answer):
        raise AnswerParseError("invalid_literal_answer")
    return answer


def _normalize_math_presentation(answer):
    # Only an explicit LaTeX text suffix from a finite unit vocabulary may
    # be removed. This avoids upstream units=True stripping symbol names.
    unit = re.search(r"\\(?:text|mbox)\s*\{([^{}]+)\}\s*(?:\^\s*(?:\{[23]\}|[23]))?\s*$", answer)
    if unit and " ".join(unit.group(1).split()).casefold() in _UNIT_WORDS and answer[:unit.start()].strip():
        answer = answer[:unit.start()].rstrip()
        answer = re.sub(r"(?:\\[,;! ]\s*)+$", "", answer)
    answer = re.sub(r"\\(?:text|mbox)\s*\{\s*and\s*\}", ",", answer)
    answer = re.sub(r",\s*,", ",", answer)
    answer = re.sub(r"(?<!\\)\\[,;!: ]", "", answer)
    answer = answer.rstrip().removesuffix(".")
    # A comma inside a single numeric LaTeX argument is a thousands marker.
    answer = re.sub(r"(?<!\\)\{([+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?)\}",
                    lambda match: "{" + match.group(1).replace(",", "") + "}", answer)
    return _expand_short_tex_arguments(answer)


def _tex_argument(text, index):
    while index < len(text) and text[index].isspace():
        index += 1
    if index == len(text):
        raise AnswerParseError("missing_tex_argument")
    if text[index] == "{":
        depth, end = 1, index + 1
        while end < len(text) and depth:
            if end == 0 or text[end - 1] != "\\":
                depth += (text[end] == "{") - (text[end] == "}")
            end += 1
        if depth or not text[index + 1:end - 1].strip():
            raise AnswerParseError("empty_or_unclosed_tex_argument")
        return text[index + 1:end - 1], end
    token = re.match(r"\\[A-Za-z]+|[0-9A-Za-z]", text[index:])
    if token is None:
        raise AnswerParseError("invalid_tex_argument")
    return token.group(), index + len(token.group())


def _expand_short_tex_arguments(answer):
    r"""Expand valid one-token TeX arguments; never fill missing operands.

    ``\frac12`` and ``\sqrt2`` are legal TeX. ANTLR expects braces, so
    canonicalize the exact one-token arguments rather than repairing math.
    """
    chunks, index = [], 0
    pattern = re.compile(r"\\(?:(?:[dtc]?frac)|sqrt)(?![A-Za-z])")
    while match := pattern.search(answer, index):
        chunks.append(answer[index:match.start()])
        command = match.group()
        end = match.end()
        if command == r"\sqrt" and answer[end:].lstrip().startswith("["):
            # Already explicit indexed roots are handled by the grammar.
            chunks.append(command)
            index = end
            continue
        arguments = []
        for _ in range(1 if command == r"\sqrt" else 2):
            argument, end = _tex_argument(answer, end)
            arguments.append("{" + _expand_short_tex_arguments(argument) + "}")
        chunks.append((r"\sqrt" if command == r"\sqrt" else r"\frac") + "".join(arguments))
        index = end
    chunks.append(answer[index:])
    return "".join(chunks)


def _base_number(answer, sympy):
    match = re.fullmatch(r"([+-]?)([0-9A-Z]+)(?:\.([0-9A-Z]+))?_(?:\{(\d+)\}|(\d+))", answer)
    if not match:
        return None
    sign, integer, fractional, braced_base, base = match.groups()
    base = int(braced_base or base)
    if not 2 <= base <= 36:
        raise AnswerParseError("invalid_numeric_base")
    digits = integer + (fractional or "")
    if any(int(char, 36) >= base for char in digits):
        raise AnswerParseError("digit_outside_numeric_base")
    numerator = int(digits, base) * (-1 if sign == "-" else 1)
    return sympy.Rational(numerator, base ** len(fractional or ""))


def _numeric_string(answer):
    # Currency is presentation; '%' is a mathematical factor and is not
    # stripped. Commas are accepted only in proper three-digit groups.
    answer = _unwrap(answer).replace(r"\$", "$")
    answer = re.sub(r"^([+-]?)\s*[$£€]\s*", r"\1", answer)
    answer = re.sub(r"\s*(?:dollars?|euros?|pounds?)\s*$", "", answer, flags=re.I)
    if not _SCALAR.fullmatch(answer):
        return None
    parts = answer.replace(",", "").split("/")
    try:
        result = Fraction(parts[0].strip())
        if len(parts) == 2:
            result /= Fraction(parts[1].strip())
        return result
    except (ValueError, ZeroDivisionError, OverflowError) as exc:
        raise AnswerParseError("invalid_numeric_answer") from exc


@lru_cache(maxsize=16384)
def _parse_answer(answer, benchmark):
    _guard_answer(answer)
    latex2sympy, ConversionConfig, NormalizationConfig, _, should_treat_as_complex, sympy = _backend()
    try:
        with _deadline():
            if benchmark == "gsm8k":
                numeric = _numeric_string(answer)
                if numeric is not None:
                    return sympy.Rational(numeric.numerator, numeric.denominator)
            if benchmark == "math":
                literal = re.fullmatch(r"\(?\s*\\(?:text|mbox|textrm)\s*\{([^{}]+)\}\s*\)?", answer)
                if literal and not _SCALAR.fullmatch(literal.group(1).strip()):
                    return _literal_normalize(literal.group(1))
                answer = _normalize_math_presentation(answer)
                based = _base_number(answer, sympy)
                if based is not None:
                    return based
            expression = latex2sympy(
                answer, is_real=not should_treat_as_complex(answer),
                convert_degrees=False,
                normalization_config=NormalizationConfig(
                    basic_latex=True, units=False, malformed_operators=False,
                    nits=False, boxed="none", equations=False,
                ),
                conversion_config=ConversionConfig(lowercase_symbols=False),
            )
            # Upstream's integer comparison strips the percentage factor;
            # evaluate only that literal factor to preserve its actual scale.
            expression = expression.xreplace({sympy.UnevaluatedExpr(sympy.Rational(1, 100)): sympy.Rational(1, 100)})
            if should_treat_as_complex(answer):
                expression = expression.xreplace({symbol: sympy.I for symbol in expression.free_symbols if symbol.name == "i"})
            if benchmark == "gsm8k":
                expression = sympy.simplify(expression)
                if not (getattr(expression, "is_number", False) and expression.is_real is True
                        and expression.is_finite is True):
                    raise AnswerParseError("gsm8k_answer_not_finite_numeric_scalar")
            if expression.has(sympy.nan, sympy.zoo):
                raise AnswerParseError("undefined_answer")
            return expression
    except _GradeTimeout as exc:
        raise AnswerParseError("parsing_timeout") from exc
    except AnswerParseError:
        raise
    except Exception as exc:
        raise AnswerParseError("invalid_math_expression") from exc


@lru_cache(maxsize=16384)
def validate_reference(reference, benchmark):
    """Validate a full gold solution/final answer; return its extracted text.

    Invalid gold is a dataset error and raises; it is never silently counted
    as an incorrect prediction. Call this for every gold before generation.
    """
    answer, alternatives, _ = _reference_info(reference, benchmark)
    for value in alternatives or [answer]:
        _parse_answer(value, benchmark)
    return answer


def _reference_info(reference, benchmark):
    override_hash = hashlib.sha256(reference.encode("utf-8")).hexdigest() if isinstance(reference, str) else ""
    override = REFERENCE_OVERRIDES.get(override_hash) if benchmark == "math" else None
    if override:
        return override["answer"], override.get("alternatives"), override_hash
    return extract_final_answer(reference, reference=True, benchmark=benchmark), None, None


def math_score(prediction, reference, benchmark):
    """Score one deterministic generation; invalid predictions score false."""
    _benchmark(benchmark)
    reference_answer = validate_reference(reference, benchmark)
    _, alternatives, override_hash = _reference_info(reference, benchmark)
    result = {"correct": False, "predicted_answer": None, "reference_answer": reference_answer,
              "parse_error": None, "extraction_failed": False, "reference_override": override_hash}
    if alternatives:
        result["reference_alternatives"] = alternatives
    try:
        answer = extract_final_answer(prediction, benchmark=benchmark)
    except AnswerParseError as exc:
        return {**result, "parse_error": str(exc), "extraction_failed": True}
    result["predicted_answer"] = answer
    try:
        gold = _parse_answer(reference_answer, benchmark)
        if isinstance(gold, str):
            result["correct"] = gold == _literal_normalize(answer)
            return result
        # Resolve the only intrinsically ambiguous scalar/set notation using
        # the gold's structural type, never by searching for matching values.
        # E.g. gold set {70,110}: bare '70,110' denotes two answers, not 70110.
        sympy = _backend()[-1]
        if isinstance(gold, sympy.FiniteSet) and "," in answer and _SCALAR.fullmatch(answer):
            answer = r"\{" + answer + r"\}"
        pred = _parse_answer(answer, benchmark)
        _, _, _, verify, _, sympy = _backend()
        with _deadline():
            # GSM8K uses exact numeric equivalence, with no rounding or
            # relative-tolerance credit. MATH uses the pinned verifier.
            if benchmark == "gsm8k":
                correct = sympy.simplify(gold - pred) == 0
            else:
                correct = any(verify(candidate, pred, strict=True, float_rounding=6,
                                 numeric_precision=15, allow_set_relation_comp=False,
                                 timeout_seconds=None, raise_on_error=True)
                              for candidate in ([_parse_answer(value, benchmark) for value in alternatives]
                                                if alternatives else [gold]))
        result["correct"] = bool(correct)
    except _GradeTimeout:
        result["parse_error"] = "verification_timeout"
    except AnswerParseError as exc:
        result["parse_error"] = str(exc)
    except Exception:
        result["parse_error"] = "verification_error"
    return result


def metric_definition(benchmark):
    _benchmark(benchmark)
    _backend()
    return {
        "name": "math_accuracy", "benchmark": benchmark, "statistic": "pass@1",
        "implementation": GRADER_VERSION, "dependencies": dict(DEPENDENCIES),
        "aggregation": "100 * correct / all_examples", "range": [0, 100],
        "higher_is_better": True, "generations_per_example": 1,
        "prediction_extraction": "last explicit boxed/fbox, #### or final-answer anchor; otherwise entire plain numeric answer only",
        "invalid_predictions": "incorrect; extraction and parse failures recorded separately",
        "invalid_references": "fatal preflight error",
        "equivalence": "exact numeric" if benchmark == "gsm8k" else "Math-Verify strict symbolic; float_rounding=6, numeric_precision=15",
        "symbol_case_sensitive": True, "max_answer_chars": MAX_ANSWER_CHARS,
        "parsing_timeout_seconds": TIMEOUT_SECONDS, "verification_timeout_seconds": TIMEOUT_SECONDS,
        "preparse_complexity_guard": {
            "nested_or_chained_powers_and_factorials": "rejected before symbolic parsing",
            "numeric_exponent_limit": MAX_POWER_EXPONENT,
            "factorial_argument_limit": MAX_FACTORIAL_ARGUMENT,
            "bounded_operand_arithmetic_bits": MAX_ARITHMETIC_BITS,
            "literal_integer_power_result_bits": MAX_POWER_BITS,
            "operand_grammar": "bounded basic arithmetic, fractions, and single symbols; factorial arguments must be nonnegative integers",
        },
        "latex_malformed_operator_repair": False, "unanchored_rationale_fallback": False,
        "latex_valid_single_token_arguments": "expanded to braces without adding missing operands",
        "numeric_base_notation": "exact numeric value; base suffix never discarded",
        "percentages": "explicit factor 1/100; no scale-insensitive matching",
        "imaginary_unit": "i is sqrt(-1)",
        "literal_answers": "whole-answer text normalization; never products of letters",
        "explicit_latex_unit_suffixes": sorted(_UNIT_WORDS),
        "gold_annotation_overrides": REFERENCE_OVERRIDES if benchmark == "math" else {},
    }
