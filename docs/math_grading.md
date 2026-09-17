# Math answer grading

The fixed MetaMathQA campaign evaluates the same saved student independently
on all 1,319 GSM8K test questions and all 5,000 competition-MATH test questions.
The reported metric is final-answer accuracy, `100 * correct / examples`.
With one greedy generation per question this is pass@1. Invalid predictions
remain in the denominator; invalid references stop preflight. ROUGE-L is not
used to measure mathematical correctness.

`baseline_common/math_metrics.py` supplies `math_score(prediction, reference,
benchmark)`, `validate_reference(reference, benchmark)`, and
`metric_definition(benchmark)` for benchmark names `gsm8k` and `math`.
Prediction records distinguish `extraction_failed` from later `parse_error`.
Every reference is validated before model loading. Parsed references are
cached for repeated comparisons; no generated-answer results are cached by
this module.

## Final-answer extraction

The evaluation prompt asks for the final answer in one `\boxed{...}`. The
grader chooses the final explicit boxed/fbox, `####`, or `Final answer: / is`
anchor. Nested LaTeX braces are balanced. A later malformed box is a failure;
the grader does not fall back to an earlier correct box. Adjacent separate
boxes are rejected as ambiguous; multi-part answers should use one box with
the complete tuple, set, interval, or expression. An unanchored prediction is
accepted only when the entire completion is a numeric scalar. Numbers or
formulas elsewhere in a rationale are never searched for a correct match.

GSM8K references may contain their original worked solution followed by
`####`, or an extracted numeric value. MATH references may contain their
original solution with the final boxed answer, or an extracted final answer.
For gold solutions, a final box takes precedence over later explanatory
sentences such as “The final answer is positive, because …”. Prediction
extraction still honors a later explicit final-answer correction.

## Numeric and symbolic equivalence

GSM8K uses exact numeric equivalence, including properly grouped thousands
commas, a leading currency sign, signs, decimals, fractions, and numeric
LaTeX. It rejects non-scalar, non-real, undefined, and infinite answers. There
is no numeric tolerance for GSM8K.

MATH uses the [Hugging Face Math-Verify verifier](https://github.com/huggingface/Math-Verify)
with `strict=True`, `float_rounding=6`, `numeric_precision=15`, and
`allow_set_relation_comp=False`. Its symbolic comparison covers fractions,
radicals, equations, expressions, sets, intervals and tuples. Symbol case is
preserved. Sets, tuples, and interval boundary openness are significant.
Floating-point answers receive the pinned verifier's six-decimal rounding
behavior; this is a grading convention, not an exact algebra theorem prover.

The wrapper preserves the numerical meaning of explicit percentages (`50%`
is `1/2`) and positional base notation (`42_7` is `30`, never `42`), and treats
the mathematical symbol `i` as the imaginary unit. These rules address
upstream representations that otherwise drop a base suffix or compare an
integer percentage without its scale. A complete textual answer such as a
weekday, a person's name, or a choice label uses whole-answer normalization,
not multiplication of its individual letters. A finite list of explicit
LaTeX unit suffixes is removed; its exact contents appear in metric provenance.

Only extracted answers go through the
[latex2sympy2_extended ANTLR grammar](https://github.com/huggingface/latex2sympy2_extended),
which requires the entire input to parse. We avoid Math-Verify's general
free-text expression extractor, numeric fallback and last-equation repair.
No model text enters Python `eval`, `sympify(string)`, or `parse_expr`. The
LaTeX parser receives no `variable_values` (its optional eval-based path).
Malformed-operator repair and arbitrary unit stripping are disabled.
Valid TeX single-token arguments are expanded for the parser: `\frac12`
becomes `\frac{1}{2}` and `\sqrt2` becomes `\sqrt{2}`. This never fills an empty
or missing operand. Matrix row separators, tuple order, and full set contents
are retained. For a gold finite set, a bare comma-separated prediction is
interpreted as a complete set; this avoids reading the two answers `70,110`
as the single integer `70110`.

Each symbolic parse and verification has a five-second POSIX signal deadline;
answer length, brace depth, and numeric literal length are bounded. Timeout
exceptions cannot be swallowed by the verifier's exception handlers. Run
grading in the main thread of a Python process. Timeouts count as incorrect
predictions and are recorded. The source itself is designed for Linux, matching
the training server.

Before the symbolic backend runs, a separate structural guard rejects nested
or chained powers, factorial nesting, and factorial or binomial expressions
inside powers. For example, `9^{9^9}`, `(9^9)^9`, `(2004!)!`, and
`9^{9999*9999}` never reach ANTLR or Math-Verify. This matters because a signal
deadline alone cannot reliably interrupt a large allocation inside a native
library. The guard uses a small arithmetic grammar with bounded rational
intermediates; it never calculates a power or factorial. Absolute exponent
values and factorial/binomial arguments are limited to 10,000, rational
intermediates to 256 bits, and literal integer power results to a conservative
100,000-bit estimate. Ordinary arithmetic and fractions are supported in
exponents; single symbolic variables are allowed. Factorial/binomial arguments
must resolve to nonnegative integers. Unsupported operand syntax and even small
nested powers are conservatively rejected and remain incorrect predictions in
the denominator. These fixed rules preserve every benchmark gold answer,
including the MATH answer `1/(2004!)`, and are recorded in metric provenance as
grader version `metamath-final-v2`.

## Reproducibility and relationship to original graders

The evaluation environment pins `math-verify==0.9.0`,
`latex2sympy2_extended==1.11.0`, `antlr4-python3-runtime==4.13.2`, and
`sympy==1.13.1`. The grader checks these versions and writes them with the
extraction/equivalence rules into every metric definition. Dependencies are
isolated from the environment used by live training jobs.

The [original GSM8K implementation](https://github.com/openai/grade-school-math/blob/master/grade_school_math/dataset.py)
extracts `####` and compares strings after removing commas. This campaign
preserves its final-answer principle while accepting equivalent numeric
representations and the shared boxed output convention. The
[original MATH equivalence helper](https://github.com/hendrycks/math/blob/main/modeling/math_equivalence.py)
uses normalized string equality. This campaign instead uses pinned symbolic
equivalence. Consequently these results must be described as a controlled
MetaMathQA comparison with the documented grader, not an exact reproduction
of each baseline paper's extraction and evaluation implementation.

## Gold annotation interpretations fixed before evaluation

All 5,000 original MATH questions remain in the test set. These interpretations
were determined from the source questions and solutions before generating any
model predictions. They are keyed by SHA256 of the complete original solution
UTF-8 text, so a changed source fails to match the exception automatically.
This differs from a bare last-box grader; it preserves required multiple answers
and the source's explicitly accepted alternatives. Every result records any override hash,
and the complete rules are embedded in metric provenance. Row numbers below are
zero-based in `/nas/Datasets/hendrycks_competition_math/data/test/0000.parquet`.

| Source row | Accepted answer(s) | Reason | Complete solution SHA256 |
|---|---|---|---|
| 333 | `-3 OR 3` | gold explicitly accepts both signs, despite the derivation giving -3; preserve annotated acceptance | `a580deb03cc5bffb4fbc38e2310cace79fd202a62e95c3cde8997e7dcc149079` |
| 414 | `\{-2,1\}` | question asks all solutions; gold has two separate boxes | `ade85c49c87508e1d285f60e9d88c018ea28a4d8d42f27840ffd63f5e6e2ecd0` |
| 2152 | `\{5-10i,11,-3+6i\}` | question asks all three possible parallelogram vertices | `75f3ec17279d32db62e0544c71f42e37cb19b7e19b8fc70f5bf662aefb52a1a6` |
| 2549 | `\{-\frac{3}{2},-\frac{3}{4}\}` | question asks both solutions, boxed separately | `d69fdc848eff7184e8a33ec4a74dfdcdb10082df76caf241ae39da8d58a640dc` |
| 2656 | `\{-10879,10879\}` | question asks both values of b, boxed separately | `49795cf0fe2f5835a84a9156c68e89eb1a072ca316fa968167dbf8314d70145b` |
| 2703 | `\{3,\frac{2}{5}\}` | question asks all solutions; gold has two separate boxes | `17dae234dd0a00d2ecf17bc0106c1548b98b2b1ff6b448592211b3d8160fd3a1` |
| 2716 | `(10,0) OR (0,0)` | question explicitly requests only one focus and permits either point | `91702eef2d1f016f99f4a684d90a595eba49d7c0f78c1806654f7d02d4cd7881` |
| 2912 | `(\frac{3}{2},-\frac{5}{2}) OR (-\frac{5}{2},-\frac{5}{2})` | question explicitly asks for either one of two vertices | `bc9d573809020b4fa3107d06229d59137aa954c03b08515fa3079747cf7cd81f` |
| 2936 | `(3,-5) OR (0,0)` | gold explicitly accepts either of two points on the odd function | `b3fe46fe3f7c2672d7454489d2e941a87da7e99180dd2afa00b8178d80083ac7` |
| 3628 | `x+11` | two boxes are equivalent polynomial forms | `7848474323ef8f28fb8d4c1eec20681db9f915b316c000f78901caeb4632cfd7` |
| 4060 | `6` | box denotes divisor-count operator; gold explicitly leaves final answer 6 unboxed | `c899a3a8ddaa8afb591e8587112b529b8581e2329fbdb3d8f9cc558229fdc186` |
| 4258 | `12x-34` | two boxes are equivalent polynomial forms | `924fa52d28956659b60f90b8953ad264f3db66b5c76d5f766a414ac0606c3d1a` |
| 4559 | `\{-1,2\}` | question asks both possible values of a | `6f6e4bed7d6edd64ab43e4d9388bb92e53c68f38ea6c18d68606e174cd832d12` |
| 4562 | `\{30,45,105\}` | question asks all three angles in degrees, boxed separately | `5d24c516df211881fea26778f937d01dc2dbe8d646fd9182e37825e3a53e8a3b` |
| 4791 | `\{70,110\}` | question asks both solutions 70 and 110; unbraced gold comma is not a thousands separator | `edf899f558a5289a42f4772082fbe6b0301774b591aeeae7e4f4655cc4c225a7` |
| 4860 | `\{\frac{\pi}{4},\frac{5\pi}{4}\}` | question asks all solutions; gold has two separate boxes | `136a93ace73513de76f1d33b6f618640187184d55d91c9eeb04e5f4a53d2558e` |

## Validation

`.venv-math-eval/bin/python -m pytest -q tests/test_math_metrics.py` passes
108 regression cases, including deliberately incorrect, ambiguous, malformed,
and executable-looking strings, plus a real enforced timeout. Twenty-five
dangerous expressions use a replacement backend that fails the test if called;
the tests never ask ANTLR or Math-Verify to evaluate those expressions. The direct
test output is in [math-grading-tests.txt](evidence/math-grading-tests.txt).

The [complete gold audit](evidence/math-grading-audit.json) validates all
1,319 GSM8K and 5,000 MATH references from the final
`metamathqa_50k_v1/tests/` files. Their original solution text is identical to
the source parquet rows in the same order. All 6,319 extracted correct answers
score correctly when explicitly boxed; all 6,319 deliberately unanchored
rationales containing those same correct answers are rejected. No row is
omitted. Sixteen MATH references use the documented gold interpretations.
Every original question also matches its source row in order, and all annotated
alternative answers pass. The evidence binds source data, canonical test files, grader source, and
dependency versions by hashes or exact pins. This checks the grading code;
it contains no trained-model evaluation results.

Reproduce this audit without loading a model or using a GPU:

```bash
CUDA_VISIBLE_DEVICES='' .venv-math-eval/bin/python docs/evidence/math-grading-audit.py
```
