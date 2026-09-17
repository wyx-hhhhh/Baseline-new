import json

import pytest

from baseline_common.data import assert_disjoint, load_records, prompt_key
from baseline_common.math_data import prepare_math_pool, validate_math_pool


def write_source(path, rows):
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def fixture_sources(tmp_path, extras=()):
    rows = [{"query": f"variant {family} {variant}", "response": f"solution {variant}",
             "original_question": f"original {family}"}
            for family in range(20) for variant in range(3)]
    meta = write_source(tmp_path / "meta.json", rows + list(extras))
    gsm = write_source(tmp_path / "gsm.json", [
        {"question": "held out GSM", "answer": "Reasoning\n#### 12"},
        {"question": "second GSM", "answer": "Reasoning\n#### 13"}])
    math = write_source(tmp_path / "math.json", [
        {"problem": "held out MATH", "solution": r"Reasoning \boxed{\frac{1}{2}}",
         "type": "Algebra", "level": "Level 3"}])
    return meta, gsm, math


def prepare(sources, output, **kwargs):
    return prepare_math_pool(*sources, output, train_size=17, validation_size=7,
                             expected_gsm8k_count=2, expected_math_count=1, **kwargs)


def test_exact_quota_family_isolation_full_tests_and_manifest(tmp_path):
    sources = fixture_sources(tmp_path)
    output = tmp_path / "pool"
    manifest = prepare(sources, output)
    train, validation = load_records(output / "train.jsonl"), load_records(output / "validation.jsonl")
    assert (len(train), len(validation)) == (17, 7)
    assert_disjoint(train, validation)
    assert {r["original_question"] for r in train}.isdisjoint(r["original_question"] for r in validation)
    assert manifest["selection_counts"]["validation_reserved_family_records"] == 9
    assert manifest["selection_counts"]["unused_eligible_records"] == 36
    assert manifest["splits"]["train"]["selected_ids"] == [r["id"] for r in train]
    assert manifest["source_records"] == 60
    assert manifest["benchmark_source_records"] == {"gsm8k": 2, "math": 1}
    assert load_records(output / "tests/gsm8k.jsonl")[0]["response"] == "Reasoning\n#### 12"
    math_row = load_records(output / "tests/math.jsonl")[0]
    assert math_row["response"] == r"Reasoning \boxed{\frac{1}{2}}"
    assert math_row["type"] == "Algebra"
    assert math_row["level"] == "Level 3"
    assert validate_math_pool(output) == manifest


def test_selection_is_stable_under_source_reordering(tmp_path):
    sources = fixture_sources(tmp_path)
    first = prepare(sources, tmp_path / "first")
    meta = json.loads(sources[0].read_text())
    sources[0].write_text(json.dumps(list(reversed(meta))))
    second = prepare(sources, tmp_path / "second")
    for name in ("train", "validation", "gsm8k", "math"):
        assert first["splits"][name] == second["splits"][name]
    third = prepare(sources, tmp_path / "third", seed=43)
    assert first["splits"]["train"]["selected_ids"] != third["splits"]["train"]["selected_ids"]


def test_training_rows_sample_broadly_across_nonvalidation_families(tmp_path):
    sources = fixture_sources(tmp_path)
    write_source(sources[0], [
        {"query": f"variant {family} {variant}", "response": f"solution {variant}",
         "original_question": f"original {family}"}
        for family in range(100) for variant in range(10)])
    output = tmp_path / "pool"
    manifest = prepare_math_pool(*sources, output, train_size=100, validation_size=20,
                                 expected_gsm8k_count=2, expected_math_count=1)
    train = load_records(output / "train.jsonl")
    validation = load_records(output / "validation.jsonl")
    assert_disjoint(train, validation)
    # Filling whole training families would use only ten families for 100 rows.
    # A global deterministic row sample covers many more eligible originals.
    assert manifest["selection_counts"]["train_source_groups"] >= 45
    assert manifest["selection_counts"]["validation_source_groups"] == 2
    assert manifest["selection_policy_version"] == 2


def test_query_original_and_transitive_test_overlap_excluded_before_sampling(tmp_path):
    extras = [
        {"query": "  HELD OUT   GSM ", "response": "a", "original_question": "bridge"},
        {"query": "sibling", "response": "b", "original_question": "bridge"},
        # This links across query/original-question columns, transitively.
        {"query": "clean looking", "response": "c", "original_question": "sibling"},
        {"query": "math variant", "response": "d", "original_question": "held out MATH"},
        {"query": "", "response": "empty"},
    ]
    extras.append(dict(extras[1]))
    sources = fixture_sources(tmp_path, extras)
    manifest = prepare(sources, tmp_path / "pool")
    counts = manifest["filter_counts"]
    assert counts["source_records"] == 66
    assert counts["empty_text_records_removed"] == 1
    assert counts["exact_duplicates_removed"] == 1
    assert counts["evaluation_direct_overlaps_removed"] == 2
    assert counts["evaluation_family_overlaps_removed"] == 2
    assert counts["evaluation_overlaps_removed"] == 4
    assert counts["eligible_records"] == 60
    output = load_records(tmp_path / "pool/train.jsonl") + load_records(tmp_path / "pool/validation.jsonl")
    assert not {prompt_key(row["prompt"]) for row in output} & {prompt_key(row["query"]) for row in extras}


def test_duplicate_metadata_merges_family_links_and_stable_identity(tmp_path):
    extras = [
        {"query": "same variant", "response": "same solution", "original_question": "original 1"},
        {"query": "same variant", "response": "same solution", "original_question": "original 2"},
    ]
    sources = fixture_sources(tmp_path, extras)
    manifest = prepare_math_pool(*sources, tmp_path / "pool", train_size=46, validation_size=15,
                                 expected_gsm8k_count=2, expected_math_count=1)
    assert manifest["filter_counts"]["exact_duplicates_removed"] == 1
    assert manifest["filter_counts"]["eligible_source_groups"] == 19
    train = load_records(tmp_path / "pool/train.jsonl")
    validation = load_records(tmp_path / "pool/validation.jsonl")
    family1 = {r["source_group"] for r in train + validation if r.get("original_question") == "original 1"}
    family2 = {r["source_group"] for r in train + validation if r.get("original_question") == "original 2"}
    assert family1 == family2
    assert_disjoint(train, validation)


def test_resume_rechecks_pool_identity_arguments_and_sources(tmp_path):
    sources = fixture_sources(tmp_path)
    output = tmp_path / "pool"
    original = prepare(sources, output)
    before = {p: p.stat().st_mtime_ns for p in output.rglob("*") if p.is_file()}
    assert prepare(sources, output, resume=True) == original
    assert before == {p: p.stat().st_mtime_ns for p in before}
    with pytest.raises(FileExistsError, match="already exists"):
        prepare(sources, output)
    with pytest.raises(ValueError, match="request differs"):
        prepare(sources, output, resume=True, seed=1)
    sources[0].write_text(sources[0].read_text() + "\n")
    with pytest.raises(ValueError, match="source hashes changed"):
        prepare(sources, output, resume=True)


def test_resume_refuses_modified_artifacts_and_unknown_partial_outputs(tmp_path):
    sources = fixture_sources(tmp_path)
    output = tmp_path / "pool"
    prepare(sources, output)
    with (output / "train.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="artifact mismatch"):
        prepare(sources, output, resume=True)
    partial = tmp_path / "partial"
    partial.mkdir()
    unknown = partial / "do-not-touch.txt"
    unknown.write_text("sentinel")
    with pytest.raises(ValueError, match="Incomplete or unrecognized"):
        prepare(sources, partial, resume=True)
    assert unknown.read_text() == "sentinel"
    assert list(partial.iterdir()) == [unknown]


def test_resume_rejects_an_older_selection_policy(tmp_path):
    sources = fixture_sources(tmp_path)
    output = tmp_path / "pool"
    prepare(sources, output)
    path = output / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest.pop("selection_policy_version")
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="selection policy differs"):
        prepare(sources, output, resume=True)


def test_short_or_empty_benchmark_is_never_published(tmp_path):
    sources = fixture_sources(tmp_path)
    sources[1].write_text(json.dumps([{"question": "one", "answer": "#### 1"}]))
    with pytest.raises(ValueError, match="exactly 2"):
        prepare(sources, tmp_path / "pool")
    assert not (tmp_path / "pool").exists()
    sources[1].write_text(json.dumps([{"question": "one", "answer": ""},
                                    {"question": "two", "answer": "#### 2"}]))
    with pytest.raises(ValueError, match="Empty text"):
        prepare(sources, tmp_path / "pool")
    assert not (tmp_path / "pool").exists()


def test_insufficient_records_or_one_family_fail_without_publication(tmp_path):
    sources = fixture_sources(tmp_path)
    write_source(sources[0], [{"query": f"q{i}", "response": "a", "original_question": "same"}
                             for i in range(50)])
    with pytest.raises(ValueError, match="Insufficient independent"):
        prepare(sources, tmp_path / "one-group")
    write_source(sources[0], [{"query": f"q{i}", "response": "a"} for i in range(3)])
    with pytest.raises(ValueError, match="Insufficient independent"):
        prepare(sources, tmp_path / "small")
    assert not (tmp_path / "one-group").exists()
    assert not (tmp_path / "small").exists()


def test_benchmark_test_source_cannot_be_used_as_training(tmp_path):
    sources = fixture_sources(tmp_path)
    test_source = tmp_path / "test.json"
    test_source.write_text(sources[0].read_text())
    with pytest.raises(ValueError, match="Refusing evaluation/test"):
        prepare((test_source, sources[1], sources[2]), tmp_path / "pool")


def test_reads_saved_arrow_and_only_benchmark_parquet_test_split(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    sources = fixture_sources(tmp_path)
    meta_rows = json.loads(sources[0].read_text())
    meta = tmp_path / "arrow"
    meta.mkdir()
    (meta / "state.json").write_text(json.dumps({"_data_files": [{"filename": "data.arrow"}]}))
    (meta / "dataset_info.json").write_text("{}")
    table = pa.Table.from_pylist(meta_rows)
    with pa.OSFile(str(meta / "data.arrow"), "wb") as stream:
        with pa.ipc.new_stream(stream, table.schema) as writer:
            writer.write_table(table)
    gsm = tmp_path / "gsm-parquet"
    gsm.mkdir()
    pq.write_table(pa.Table.from_pylist(json.loads(sources[1].read_text())), gsm / "test-000.parquet")
    pq.write_table(pa.Table.from_pylist([{"question": "unused train", "answer": "#### 0"}]), gsm / "train-000.parquet")
    math = tmp_path / "competition" / "test"
    math.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(json.loads(sources[2].read_text())), math / "0000.parquet")
    manifest = prepare((meta, gsm, math), tmp_path / "pool")
    assert manifest["source_records"] == 60
    assert all("train-000" not in file["path"] for file in manifest["source_files"]["gsm8k"])
