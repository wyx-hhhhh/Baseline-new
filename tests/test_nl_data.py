import json
from pathlib import Path

import pytest

from baseline_common.nl_data import (BENCHMARKS, SOURCE_FILES, load_nl_records,
                                     prepare_nl_pool, validate_nl_pool)


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def source(tmp_path, benchmark="dolly", rows=None):
    root = tmp_path / "sources"
    for index, relative in enumerate(SOURCE_FILES[benchmark]):
        write(root / relative, rows or [{"instruction": f"Question {index}", "input": "context",
                                        "output": ["one", "two"] if benchmark == "super_natural" else "answer",
                                        "prompt": "DO NOT DOUBLE TEMPLATE THIS"}])
    return root


def test_multireference_exact_duplicates_and_all_provenance(tmp_path):
    rows = [{"instruction": "Question", "input": "Context", "output": ["first", "best answer"], "topic": "task"}]
    root = source(tmp_path, "super_natural", rows)
    manifest = prepare_nl_pool(root, tmp_path / "pool", ["super_natural"])
    entry = manifest["benchmarks"]["super_natural"]
    assert (entry["source_records"], entry["exact_duplicates_merged"], entry["records"]) == (4, 3, 1)
    row, = load_nl_records(tmp_path / "pool/super_natural.jsonl")
    assert row["prompt"] == "Question\n\nContext"
    assert row["references"] == ["first", "best answer"]
    assert row["response"] == "first"
    assert row["topic"] == "task"
    assert len(row["source_rows"]) == 4
    assert validate_nl_pool(tmp_path / "pool") == manifest


def test_prompt_variants_and_distinct_reference_examples_not_merged(tmp_path):
    root = source(tmp_path, rows=[{"instruction": "Question", "output": "A"},
                                  {"instruction": "Question", "output": "B"},
                                  {"instruction": "question", "output": "A"}])
    manifest = prepare_nl_pool(root, tmp_path / "pool", ["dolly"])
    assert manifest["benchmarks"]["dolly"]["records"] == 3


def test_union_train_and_pair_exclusions_normalize_prompts(tmp_path):
    root = source(tmp_path, rows=[{"instruction": "ＦＯＯ  bar", "output": "A"},
                                  {"instruction": "Other", "output": "B"},
                                  {"instruction": "Holdout", "output": "C"}])
    train, pairs = tmp_path / "train.jsonl", tmp_path / "pairs.jsonl"
    write(train, [{"id": "a", "prompt": "foo\nbar", "response": "something"}])
    write(pairs, [{"id": "b", "prompt": "OTHER", "chosen_ids": [1], "rejected_ids": [2]}])
    manifest = prepare_nl_pool(root, tmp_path / "pool", ["dolly"], [train, pairs, train])
    entry = manifest["benchmarks"]["dolly"]
    assert (entry["training_overlaps_removed"], entry["records"]) == (2, 1)
    assert len(manifest["exclusion_files"]) == 2
    assert all(row["matched_files"] for row in entry["excluded_records"])
    assert load_nl_records(tmp_path / "pool/dolly.jsonl")[0]["prompt"] == "Holdout"


def test_only_explicit_validation_files_are_read(tmp_path):
    root = source(tmp_path)
    write(root / "DollyEval/raw.jsonl", [{"bad": "training file"}])
    write(root / "DollyEval/valid.copy.jsonl", [{"bad": "duplicate copy"}])
    manifest = prepare_nl_pool(root, tmp_path / "pool", ["dolly"])
    assert len(manifest["source_files"]) == 1


def test_missing_required_bin_fails_without_partial_output(tmp_path):
    root = source(tmp_path, "super_natural")
    (root / SOURCE_FILES["super_natural"][2]).unlink()
    with pytest.raises(FileNotFoundError, match="Required local"):
        prepare_nl_pool(root, tmp_path / "pool", ["super_natural"])
    assert not (tmp_path / "pool").exists()


@pytest.mark.parametrize("output", [[], ["good", ""], ["good", 123], None, {"answer": "no"}])
def test_invalid_reference_is_rejected(tmp_path, output):
    root = source(tmp_path, rows=[{"instruction": "Question", "output": output}])
    with pytest.raises(ValueError):
        prepare_nl_pool(root, tmp_path / "pool", ["dolly"])
    assert not (tmp_path / "pool").exists()


def test_resume_is_immutable_and_rejects_changed_request(tmp_path):
    root = source(tmp_path)
    pool = tmp_path / "pool"
    manifest = prepare_nl_pool(root, pool, ["dolly"])
    before = {path.name: path.read_bytes() for path in pool.iterdir()}
    assert prepare_nl_pool(root, pool, ["dolly"]) == manifest
    assert before == {path.name: path.read_bytes() for path in pool.iterdir()}
    with pytest.raises(FileExistsError):
        prepare_nl_pool(root, pool, ["dolly"], resume=False)
    with pytest.raises(ValueError, match="request changed"):
        prepare_nl_pool(root, pool, ["selfinst"])


@pytest.mark.parametrize("changed", ["source", "exclusion", "canonical"])
def test_changed_input_or_artifact_rejected_on_resume(tmp_path, changed):
    root = source(tmp_path)
    exclusion = tmp_path / "train.jsonl"
    write(exclusion, [{"prompt": "not a benchmark question"}])
    pool = tmp_path / "pool"
    prepare_nl_pool(root, pool, ["dolly"], [exclusion])
    target = {"source": root / SOURCE_FILES["dolly"][0], "exclusion": exclusion,
              "canonical": pool / "dolly.jsonl"}[changed]
    target.write_text(target.read_text() + "\n")
    with pytest.raises(ValueError, match="changed|mismatch"):
        prepare_nl_pool(root, pool, ["dolly"], [exclusion])


def test_incomplete_output_not_overwritten(tmp_path):
    root = source(tmp_path)
    (tmp_path / "pool").mkdir()
    with pytest.raises(ValueError, match="Incomplete"):
        prepare_nl_pool(root, tmp_path / "pool", ["dolly"])


def test_all_training_overlap_rejected(tmp_path):
    root = source(tmp_path)
    exclusion = tmp_path / "train.jsonl"
    write(exclusion, [{"prompt": "Question 0\n\ncontext"}])
    with pytest.raises(ValueError, match="No held-out"):
        prepare_nl_pool(root, tmp_path / "pool", ["dolly"], [exclusion])


@pytest.mark.parametrize("benchmarks", [[], ["dolly", "dolly"], ["unknown"]])
def test_invalid_selection_rejected(tmp_path, benchmarks):
    with pytest.raises(ValueError, match="benchmarks must"):
        prepare_nl_pool(tmp_path, tmp_path / "pool", benchmarks)


@pytest.mark.parametrize("mutation", ["string_refs", "wrong_first", "duplicate_id", "invalid_source"])
def test_canonical_validation(tmp_path, mutation):
    root = source(tmp_path)
    prepare_nl_pool(root, tmp_path / "pool", ["dolly"])
    path = tmp_path / "pool/dolly.jsonl"
    rows = load_nl_records(path)
    if mutation == "string_refs":
        rows[0]["references"] = "answer"
    elif mutation == "wrong_first":
        rows[0]["response"] = "changed"
    elif mutation == "duplicate_id":
        rows.append(rows[0].copy())
    else:
        rows[0]["source_rows"][0]["line"] = 0
    write(path, rows)
    with pytest.raises(ValueError):
        load_nl_records(path)
