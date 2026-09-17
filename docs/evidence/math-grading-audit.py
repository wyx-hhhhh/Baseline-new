"""Re-run the frozen campaign's CPU-only grading and source-reference audit."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pyarrow.parquet as parquet

from baseline_common import math_metrics as metrics


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    started = time.monotonic()
    output = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "grader_source_sha256": sha256(ROOT / "baseline_common/math_metrics.py"),
        "audit_source_sha256": sha256(__file__),
        "datasets": [], "failures": [],
        "scope": "CPU grading audit only; generated no model responses and launched no training",
    }
    sources = [
        ("gsm8k", "/nas/Datasets/GSM8K/main/test-00000-of-00001.parquet", "question", "answer", 1319),
        ("math", "/nas/Datasets/hendrycks_competition_math/data/test/0000.parquet", "problem", "solution", 5000),
    ]
    for benchmark, source, question_key, reference_key, count in sources:
        canonical = Path("/nas/Users/wyx/Baseline/data/metamathqa_50k_v1/tests") / f"{benchmark}.jsonl"
        originals = parquet.read_table(source).to_pylist()
        records = [json.loads(line) for line in canonical.read_text().splitlines()]
        assert len(originals) == len(records) == count
        assert [row[reference_key] for row in originals] == [row["response"] for row in records]
        assert [row[question_key] for row in originals] == [row["prompt"] for row in records]
        passed, rejected, overrides, alternative_passed = 0, 0, [], 0
        for index, row in enumerate(records):
            reference = row["response"]
            answer = metrics.validate_reference(reference, benchmark)
            score = metrics.math_score(r"\boxed{" + answer + "}", reference, benchmark)
            if score["correct"] and score["parse_error"] is None:
                passed += 1
            else:
                output["failures"].append({"benchmark": benchmark, "row": index, "selfscore": score})
            unanchored = "An intermediate quantity is " + answer + "; I cannot determine a final result."
            negative = metrics.math_score(unanchored, reference, benchmark)
            if not negative["correct"] and negative["extraction_failed"]:
                rejected += 1
            else:
                output["failures"].append({"benchmark": benchmark, "row": index, "unanchored": negative})
            if score["reference_override"]:
                digest = score["reference_override"]
                interpretation = metrics.REFERENCE_OVERRIDES[digest]
                assert digest == hashlib.sha256(reference.encode("utf-8")).hexdigest()
                assert interpretation["row"] == index
                overrides.append({"row": index, "complete_solution_sha256": digest})
                for alternative in score.get("reference_alternatives", []):
                    alternative_score = metrics.math_score(r"\boxed{" + alternative + "}", reference, benchmark)
                    if alternative_score["correct"]:
                        alternative_passed += 1
                    else:
                        output["failures"].append({"benchmark": benchmark, "row": index, "alternative": alternative_score})
        assert len(overrides) == (16 if benchmark == "math" else 0)
        if benchmark == "math":
            assert {entry["complete_solution_sha256"] for entry in overrides} == set(metrics.REFERENCE_OVERRIDES)
        output["datasets"].append({
            "benchmark": benchmark, "source": source, "source_sha256": sha256(source),
            "canonical_file": str(canonical), "canonical_sha256": sha256(canonical),
            "rows": count, "source_references_identical_in_order": True,
            "source_questions_identical_in_order": True, "rows_omitted": 0,
            "gold_validation_and_selfscore_passed": passed,
            "unanchored_correct_rationale_rejected": rejected,
            "annotation_overrides": len(overrides), "annotation_override_records": overrides,
            "additional_accepted_alternatives_verified": alternative_passed,
            "metric": metrics.metric_definition(benchmark),
        })
        print(f"{benchmark}: {passed}/{count} gold scores; {rejected}/{count} rationale-only rejections; {len(overrides)} fixed interpretations", flush=True)
    output["elapsed_seconds"] = time.monotonic() - started
    destination = ROOT / "docs/evidence/math-grading-audit.json"
    destination.write_text(json.dumps(output, indent=2) + "\n")
    assert not output["failures"], output["failures"]
    print(f"PASS: {destination}", flush=True)


if __name__ == "__main__":
    main()
