import json
from pathlib import Path

import pytest

from baseline_common.data import (adapt_record, assert_disjoint, file_sha256,
                                  join_pair_records, load_records, prepare_dataset,
                                  read_source, resolve_source, split_records)


def rows(n=30):
    return [{"id": str(i), "prompt": f"Question {i}", "response": f"Answer {i}"} for i in range(n)]


def test_source_adapters_and_explicit_context():
    result = adapt_record({"instruction": "Task", "context": "Context", "response": "Answer"}, "dolly")
    assert result["prompt"] == "Task\n\nContext"
    assert adapt_record({"instruction": "Task", "input": "Context", "output": "Answer", "prompt": "old formatted"}) == result
    assert adapt_record({"query": "Task", "response": "Answer"})["prompt"] == "Task"
    assert adapt_record({"question": "Task", "answer": "Answer"})["response"] == "Answer"
    with pytest.raises(ValueError, match="Unsupported"):
        adapt_record({"unknown": "test"})


def test_split_is_order_independent_and_groups_duplicate_prompts():
    source = rows()
    source += [{"id": "duplicate", "prompt": "  QUESTION   2  ", "response": "different answer"}]
    a, b = split_records(source, .2, 42)
    assert (a, b) == split_records(reversed(source), .2, 42)
    assert_disjoint(a, b)
    for split in [a, b]:
        ids = {r["id"] for r in split}
        assert ("2" in ids) == ("duplicate" in ids)
    with pytest.raises(ValueError, match="leakage"):
        assert_disjoint(a, [a[0]])


def test_metamath_augmented_groups_do_not_leak_transitively():
    source = rows(10) + [
        {"id": "a", "prompt": "same", "response": "x", "source_group": "original-1"},
        {"id": "b", "prompt": "same", "response": "y", "source_group": "original-2"},
        {"id": "c", "prompt": "paraphrase", "response": "z", "source_group": "original-2"},
    ]
    train, validation = split_records(source, .5)
    for split in [train, validation]:
        assert len({row["id"] for row in split} & {"a", "b", "c"}) in {0, 3}


def test_prepare_local_json_records_hashes_and_reserved_eval(tmp_path):
    source = tmp_path / "corpus.json"
    source.write_text(json.dumps({"instances": rows()}))
    excluded = tmp_path / "test.jsonl"
    excluded.write_text(json.dumps(rows()[0]) + "\n")
    output = tmp_path / "prepared"
    manifest = prepare_dataset(source, output, exclude_sources=[excluded])
    train, validation = load_records(output / "train.jsonl"), load_records(output / "validation.jsonl")
    assert len(train) + len(validation) == 29
    assert "0" not in {r["id"] for r in train + validation}
    assert_disjoint(train, validation)
    assert manifest["evaluation_overlaps_removed"] == 1
    assert manifest["source_files"][0]["sha256"] == file_sha256(source)
    assert manifest["splits"]["train"]["sha256"] == file_sha256(output / "train.jsonl")
    with pytest.raises(FileExistsError):
        prepare_dataset(source, output)
    with pytest.raises(ValueError, match="evaluation/test"):
        prepare_dataset(excluded, tmp_path / "bad")


def test_resolver_never_collects_test_when_given_directory(tmp_path):
    (tmp_path / "train-00000.parquet").touch()
    (tmp_path / "test-00000.parquet").touch()
    kind, files = resolve_source(tmp_path)
    assert kind == "parquet"
    assert [p.name for p in files] == ["train-00000.parquet"]


def test_duplicate_source_ids_are_rejected(tmp_path):
    source = tmp_path / "source.json"
    source.write_text(json.dumps([{"id": "same", "question": "q1", "answer": "a"},
                                  {"id": "same", "question": "q2", "answer": "a"}]))
    with pytest.raises(ValueError, match="different records"):
        prepare_dataset(source, tmp_path / "out")


def test_empty_source_rows_are_counted_not_silently_trained(tmp_path):
    source = tmp_path / "source.json"
    source.write_text(json.dumps(rows() + [{"query": " ", "response": "answer"}]))
    manifest = prepare_dataset(source, tmp_path / "out")
    assert manifest["empty_text_records_removed"] == 1
    assert manifest["source_records"] == 31


def test_pair_join_is_identity_based_and_preserves_high_ids():
    provenance = {"teacher": {"path": "teacher"}, "student": {"path": "student"}, "tokenizer": "same"}
    a = {"id": "a", "prompt": "p", "prompt_ids": [151644, 65535], "response_ids": [151645], "response": ""}
    b = {"id": "b", "prompt": "q", "prompt_ids": [151644], "response_ids": [128009], "response": ""}
    student_a = dict(a, response_ids=[42, 151645])
    joined = join_pair_records([a, b], [b, student_a], provenance=provenance)
    assert joined[0]["prompt_ids"] == [151644, 65535]
    assert joined[0]["rejected_ids"] == [42, 151645]
    with pytest.raises(ValueError, match="ID mismatch"):
        join_pair_records([a, b], [a], provenance=provenance)
    with pytest.raises(ValueError, match="Duplicate"):
        join_pair_records([a, a], [a], provenance=provenance)
    with pytest.raises(ValueError, match="prompt mismatch"):
        join_pair_records([a], [dict(a, prompt_ids=[1])], provenance=provenance)


def test_hf_disk_adapter(tmp_path):
    pa = pytest.importorskip("pyarrow")
    source = tmp_path / "hf"
    (source / "train").mkdir(parents=True)
    (source / "dataset_dict.json").write_text(json.dumps({"splits": ["train", "test"]}))
    (source / "train" / "state.json").write_text(json.dumps({"_data_files": [{"filename": "data-00000-of-00001.arrow"}]}))
    (source / "train" / "dataset_info.json").write_text("{}")
    table = pa.Table.from_pylist(rows())
    with pa.OSFile(str(source / "train" / "data-00000-of-00001.arrow"), "wb") as stream:
        with pa.ipc.new_stream(stream, table.schema) as writer:
            writer.write_table(table)
    iterator, files = read_source(source)
    assert len(list(iterator)) == 30
    assert all(file.parent.name == "train" for file in files)
    result = prepare_dataset(source, tmp_path / "out")
    assert result["source_records"] == 30


def test_parquet_adapter(tmp_path):
    pa = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    parquet.write_table(pa.Table.from_pylist(rows()), tmp_path / "train-00000.parquet")
    parquet.write_table(pa.Table.from_pylist(rows(2)), tmp_path / "test-00000.parquet")
    iterator, files = read_source(tmp_path)
    assert list(iterator) == rows()
    assert len(files) == 1
