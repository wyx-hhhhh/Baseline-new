"""Local dataset adapters and auditable, prompt-disjoint experiment artifacts.

Raw datasets remain untouched. Canonical files contain text, not legacy unsigned
token streams or in-band separators. Generated pair files retain exact token IDs.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_DATA_ROOT = Path("/nas/Users/wyx/Baseline/data")
DATASET_SOURCES = {
    "dolly": Path("/nas/Datasets/databricks-dolly-15k/databricks-dolly-15k.jsonl"),
    "metamathqa": Path("/nas/Datasets/MetaMathQA"),
    "gsm8k": Path("/nas/Datasets/GSM8K/main"),
}


class EmptyTextError(ValueError):
    """A recognized source record has unusable empty prompt/response text."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prompt_key(prompt: str) -> str:
    """Conservative whitespace/Unicode normalization for duplicate isolation."""
    return " ".join(unicodedata.normalize("NFKC", prompt).split()).casefold()


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def _text(row: dict, key: str, *, allow_empty: bool = False) -> str:
    value = row.get(key)
    if not isinstance(value, str):
        raise ValueError(f"Expected {'a string' if allow_empty else 'nonempty text'} in {key!r}")
    if not allow_empty and not value.strip():
        raise EmptyTextError(f"Empty text in {key!r}")
    return value


def adapt_record(row: dict, dataset: str = "auto") -> dict:
    """Convert local Dolly, MetaMathQA, GSM8K and common instruction records."""
    if not isinstance(row, dict):
        raise ValueError("Each source record must be an object")
    # MiniLLM Dolly has an already-formatted `prompt`: rebuild from instruction
    # fields so model-specific chat templates are applied exactly once later.
    if "instruction" in row:
        prompt = _text(row, "instruction")
        context = row.get("input", row.get("context", ""))
        if not isinstance(context, str):
            raise ValueError("instruction input/context must be text")
        if context.strip():
            prompt += "\n\n" + context
        response = _text(row, "output" if "output" in row else "response")
    elif "query" in row and "response" in row:
        prompt, response = _text(row, "query"), _text(row, "response")
    elif "question" in row and "answer" in row:
        prompt, response = _text(row, "question"), _text(row, "answer")
    elif "problem" in row and "solution" in row:
        prompt, response = _text(row, "problem"), _text(row, "solution")
    elif "dialogue" in row and "summary" in row:
        prompt, response = _text(row, "dialogue"), _text(row, "summary")
    elif "prompt" in row and "response" in row:
        prompt, response = _text(row, "prompt"), _text(row, "response")
    elif "messages" in row:
        messages = row["messages"]
        # Restrict to one exchange: silently dropping later turns/system messages
        # would alter the task. Multi-turn data needs an explicit adapter.
        if (not isinstance(messages, list) or len(messages) != 2
                or [x.get("role") for x in messages] != ["user", "assistant"]):
            raise ValueError("messages adapter requires exactly one user/assistant exchange")
        prompt, response = _text(messages[0], "content"), _text(messages[1], "content")
    else:
        raise ValueError(f"Unsupported source schema; available columns: {sorted(row)}")
    result = {"id": str(row.get("id", _digest([prompt, response]))),
              "prompt": prompt, "response": response}
    if not result["id"]:
        raise ValueError("Record id must not be empty")
    # MetaMathQA includes paraphrases of a shared original question. Group those
    # together as well as identical rendered prompts to avoid augmented leakage.
    if isinstance(row.get("original_question"), str) and row["original_question"].strip():
        result["source_group"] = _digest(prompt_key(row["original_question"]))
    return result


def _json_rows(path: Path, split: str) -> Iterable[dict]:
    with path.open(encoding="utf-8") as stream:
        if path.suffix == ".jsonl":
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON in {path}:{number}") from error
        else:
            document = json.load(stream)
            if isinstance(document, dict):
                if split in document:
                    document = document[split]
                elif "instances" in document:
                    document = document["instances"]
                elif "data" in document:
                    document = document["data"]
            if not isinstance(document, list):
                raise ValueError(f"{path} must contain an array, a split array, or an instances/data array")
            yield from document


def _is_eval_path(path: Path) -> bool:
    return (path.parent.name.lower() in {"test", "validation", "valid", "dev", "eval"}
            or bool(re.search(r"(^|[_\-.])(test|validation|valid|dev|eval)([_\-.]|$)",
                              path.name.lower())))


def resolve_source(source: str | Path, split: str = "train") -> tuple[str, list[Path]]:
    """Select one explicit local split; never glob train and evaluation together."""
    path = Path(source).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Local dataset not found: {path}")
    if path.is_file():
        if path.suffix not in {".json", ".jsonl", ".parquet"}:
            raise ValueError(f"Unsupported dataset file: {path}")
        return path.suffix[1:], [path]
    if (path / "dataset_dict.json").exists():
        info = json.loads((path / "dataset_dict.json").read_text())
        if split not in info.get("splits", []):
            raise ValueError(f"Split {split!r} is absent from {path}")
        path = path / split
    if (path / "state.json").exists() and (path / "dataset_info.json").exists():
        state = json.loads((path / "state.json").read_text())
        files = [path / "state.json", path / "dataset_info.json"]
        files.extend(path / item["filename"] for item in state.get("_data_files", []))
        if len(files) == 2 or any(not item.is_file() for item in files):
            raise ValueError(f"Incomplete Hugging Face save_to_disk dataset: {path}")
        return "hf_disk", files
    aliases = {"validation": ["validation", "valid", "dev"]}.get(split, [split])
    candidates = []
    for directory in [path, path / "data", *(path / alias for alias in aliases),
                      *(path / "data" / alias for alias in aliases)]:
        if not directory.is_dir():
            continue
        in_split = directory.name in aliases
        for item in directory.iterdir():
            if item.suffix not in {".json", ".jsonl", ".parquet"} or not item.is_file():
                continue
            if in_split or any(re.match(re.escape(alias) + r"(?:[-_.]|$)", item.name)
                               for alias in aliases):
                candidates.append(item)
    candidates = sorted(set(candidates))
    if not candidates:
        # Unsplitted instruction corpora: these filenames are unambiguous.
        candidates = [item for item in [path / "databricks-dolly-15k.jsonl", path / "raw.jsonl"]
                      if item.is_file()]
    if not candidates:
        raise ValueError(f"No unambiguous {split!r} files in {path}; point --source at an explicit file/subdirectory")
    formats = {item.suffix for item in candidates}
    if len(formats) != 1:
        raise ValueError(f"Multiple source formats in {path}; specify a file/subdirectory to avoid duplicate copies")
    return candidates[0].suffix[1:], candidates


def read_source(source: str | Path, split: str = "train") -> tuple[Iterable[dict], list[Path]]:
    kind, files = resolve_source(source, split)
    if kind == "hf_disk":
        try:
            import pyarrow as arrow
        except ImportError as error:
            raise RuntimeError("pyarrow is required for save_to_disk input; install the experiment requirements") from error
        # save_to_disk materializes rows into the Arrow IPC streams referenced by
        # state.json. Read only those files, never cache-*.arrow or extra splits.
        # Primitive text corpora need no datasets runtime or mutable HF cache.
        def arrow_rows():
            for file in files[2:]:
                with arrow.memory_map(str(file), "r") as stream:
                    reader = arrow.ipc.open_stream(stream)
                    for batch in reader:
                        yield from batch.to_pylist()
        return arrow_rows(), files
    if kind == "parquet":
        try:
            import pyarrow.parquet as parquet
        except ImportError as error:
            raise RuntimeError("pyarrow is required for parquet input; install the experiment requirements") from error
        def parquet_rows():
            for file in files:
                for batch in parquet.ParquetFile(file).iter_batches(batch_size=1024):
                    yield from batch.to_pylist()
        return parquet_rows(), files
    def json_rows():
        for file in files:
            yield from _json_rows(file, split)
    return json_rows(), files


def validate_records(records: Iterable[dict], *, paired: bool = False) -> list[dict]:
    result, ids = [], set()
    for row in records:
        if not isinstance(row, dict):
            raise ValueError("Canonical records must be JSON objects")
        identifier = _text(row, "id")
        if identifier in ids:
            raise ValueError(f"Duplicate record id {identifier!r}")
        ids.add(identifier)
        _text(row, "prompt")
        if paired:
            for field in ("prompt_ids", "chosen_ids", "rejected_ids"):
                tokens = row.get(field)
                if (not isinstance(tokens, list) or not tokens
                        or any(type(token) is not int or token < 0 for token in tokens)):
                    raise ValueError(f"{identifier}: {field} must be nonempty nonnegative integer token IDs")
            provenance = row.get("provenance")
            if not isinstance(provenance, dict) or not all(provenance.get(k) for k in ("teacher", "student", "tokenizer")):
                raise ValueError(f"{identifier}: paired rows require teacher/student/tokenizer provenance")
        else:
            _text(row, "response")
        result.append(row)
    if not result:
        raise ValueError("Dataset is empty")
    return result


def load_records(path: str | Path, *, paired: bool = False) -> list[dict]:
    return validate_records(_json_rows(Path(path), "train"), paired=paired)


def assert_disjoint(train: Iterable[dict], validation: Iterable[dict]) -> None:
    train, validation = list(train), list(validation)
    for key, transform in [("id", lambda x: x), ("prompt", prompt_key),
                           ("source_group", lambda x: x)]:
        train_keys = {transform(row[key]) for row in train if key in row}
        validation_keys = {transform(row[key]) for row in validation if key in row}
        if train_keys & validation_keys:
            raise ValueError(f"Train/validation leakage: overlapping {key}")


def split_records(records: Iterable[dict], validation_fraction: float = .05,
                  seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Deterministic connected-component split, stable under input reordering."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be strictly between 0 and 1")
    rows = validate_records(records)
    parent = list(range(len(rows)))
    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index
    keys = {}
    for index, row in enumerate(rows):
        row_keys = [("prompt", prompt_key(row["prompt"]))]
        if "source_group" in row:
            row_keys.append(("source_group", row["source_group"]))
        for key in row_keys:
            if key in keys:
                parent[find(index)] = find(keys[key])
            else:
                keys[key] = index
    components = defaultdict(list)
    for index, row in enumerate(rows):
        components[find(index)].append(row)
    groups = list(components.values())
    if len(groups) < 2:
        raise ValueError("At least two independent prompt groups are required for disjoint train/validation")
    groups.sort(key=lambda group: _digest([seed, min(row["id"] for row in group)]))
    target = max(1, math.ceil(len(rows) * validation_fraction))
    validation, train = [], []
    for index, group in enumerate(groups):
        (validation if len(validation) < target and index < len(groups) - 1 else train).extend(group)
    train.sort(key=lambda row: row["id"])
    validation.sort(key=lambda row: row["id"])
    assert_disjoint(train, validation)
    return train, validation


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def prepare_dataset(source: str | Path, output_dir: str | Path, *, dataset: str = "auto",
                    source_split: str = "train", validation_fraction: float = .05,
                    seed: int = 42, max_records: int | None = None,
                    exclude_sources: Iterable[str | Path] = ()) -> dict:
    if not re.match(r"^train(?:$|_)", source_split):
        raise ValueError("Preparation only accepts a training source split; reserve test/evaluation sources")
    if max_records is not None and max_records < 2:
        raise ValueError("max_records must be at least two")
    output = Path(output_dir).expanduser()
    if any((output / name).exists() for name in ("train.jsonl", "validation.jsonl", "manifest.json")):
        raise FileExistsError(f"Prepared outputs already exist in {output}; use a new output directory")
    iterator, source_files = read_source(source, source_split)
    if any(_is_eval_path(path) for path in source_files):
        raise ValueError("Refusing evaluation/test files as a training source")
    excluded, exclusion_files = set(), []
    for exclusion in exclude_sources:
        excluded_rows, files = read_source(exclusion)
        exclusion_files.extend(files)
        for row in excluded_rows:
            canonical = adapt_record(row, dataset)
            excluded.add(prompt_key(canonical["prompt"]))
            if "original_question" in row:
                excluded.add(prompt_key(row["original_question"]))
    unique, excluded_count, duplicate_count, seen_count, empty_count = {}, 0, 0, 0, 0
    for row in iterator:
        seen_count += 1
        try:
            canonical = adapt_record(row, dataset)
        except EmptyTextError:
            empty_count += 1
            continue
        if (prompt_key(canonical["prompt"]) in excluded
                or (isinstance(row.get("original_question"), str)
                    and prompt_key(row["original_question"]) in excluded)):
            excluded_count += 1
            continue
        identifier = canonical["id"]
        if identifier in unique:
            if unique[identifier] != canonical:
                raise ValueError(f"Source id {identifier!r} maps to different records")
            duplicate_count += 1
            continue
        unique[identifier] = canonical
    rows = sorted(unique.values(), key=lambda row: _digest([seed, row["id"]]))
    if max_records is not None:
        rows = rows[:max_records]
    train, validation = split_records(rows, validation_fraction, seed)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "train.jsonl", train)
    write_jsonl(output / "validation.jsonl", validation)
    manifest = {
        "schema_version": 1, "dataset": dataset, "source": str(Path(source).resolve()),
        "source_split": source_split, "seed": seed, "validation_fraction_requested": validation_fraction,
        "split_policy": "hash-ranked connected components of normalized prompt and original_question",
        "normalization": "Unicode NFKC, whitespace collapse, casefold",
        "source_records": seen_count, "exact_duplicates_removed": duplicate_count,
        "empty_text_records_removed": empty_count,
        "evaluation_overlaps_removed": excluded_count, "max_records": max_records,
        "source_files": [{"path": str(path), "sha256": file_sha256(path)} for path in source_files],
        "exclusion_files": [{"path": str(path), "sha256": file_sha256(path)} for path in exclusion_files],
        "splits": {name: {"path": f"{name}.jsonl", "records": len(records),
                          "sha256": file_sha256(output / f"{name}.jsonl")}
                   for name, records in [("train", train), ("validation", validation)]},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def join_pair_records(teacher: Iterable[dict], student: Iterable[dict], *, provenance: dict) -> list[dict]:
    """Strict identity join; neither missing rows nor duplicate prompts disappear."""
    def index(rows):
        result = {}
        for row in rows:
            identifier = _text(row, "id")
            if identifier in result:
                raise ValueError(f"Duplicate generation id {identifier!r}")
            result[identifier] = row
        return result
    teacher, student = index(teacher), index(student)
    if teacher.keys() != student.keys():
        raise ValueError(f"Teacher/student ID mismatch: teacher-only={len(teacher.keys() - student.keys())}, "
                         f"student-only={len(student.keys() - teacher.keys())}")
    pairs = []
    for identifier in sorted(teacher):
        chosen, rejected = teacher[identifier], student[identifier]
        if (chosen["prompt"] != rejected["prompt"] or chosen["prompt_ids"] != rejected["prompt_ids"]
                or chosen.get("source_group") != rejected.get("source_group")):
            raise ValueError(f"Teacher/student prompt mismatch for {identifier!r}")
        pairs.append({"id": identifier, "prompt": chosen["prompt"],
                      "prompt_ids": chosen["prompt_ids"], "chosen_ids": chosen["response_ids"],
                      "rejected_ids": rejected["response_ids"],
                      "chosen": chosen.get("response", ""), "rejected": rejected.get("response", ""),
                      "provenance": provenance,
                      **({"source_group": chosen["source_group"]} if "source_group" in chosen else {})})
    return validate_records(pairs, paired=True)
