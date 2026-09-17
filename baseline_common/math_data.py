"""Immutable, shared MetaMathQA pools with both complete math test sets.

The split unit is a connected family of normalized queries/original questions.
A family's unused rows never become validation rows just to fill a quota.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .data import (EmptyTextError, _digest, _is_eval_path, adapt_record,
                   assert_disjoint, file_sha256, load_records, prompt_key,
                   read_source, write_jsonl)


DEFAULT_MATH_POOL = Path("/nas/Users/wyx/Baseline/data/metamathqa_50k_v1")
DEFAULT_METAMATH_SOURCE = Path("/nas/Datasets/MetaMathQA")
DEFAULT_GSM8K_SOURCE = Path("/nas/Datasets/GSM8K/main")
DEFAULT_MATH_SOURCE = Path("/nas/Datasets/hendrycks_competition_math/data/test")
ARTIFACTS = {"train": "train.jsonl", "validation": "validation.jsonl",
             "gsm8k": "tests/gsm8k.jsonl", "math": "tests/math.jsonl"}
SELECTION_POLICY_VERSION = 2


def _snapshots(files: list[Path]) -> list[dict]:
    return [{"path": str(path.resolve()), "bytes": path.stat().st_size,
             "sha256": file_sha256(path)} for path in files]


def _benchmark(rows, dataset: str, expected_count: int) -> list[dict]:
    records = []
    for index, raw in enumerate(rows):
        row = adapt_record(raw, dataset)
        row["id"] = f"{dataset}:{index:05d}:{_digest([row['prompt'], row['response']])[:16]}"
        row["dataset"] = dataset
        # Do not shorten references or strip the original ####/boxed answer.
        for field in ("type", "level"):
            if field in raw:
                row[field] = raw[field]
        if "id" in raw:
            row["source_id"] = str(raw["id"])
        records.append(row)
    if len(records) != expected_count:
        raise ValueError(f"{dataset} test must contain exactly {expected_count} records; found {len(records)}")
    return records


def _training_families(raw_rows, held_out: set[str]) -> tuple[list[list[dict]], dict]:
    unique: dict[str, dict] = {}
    counts = {"source_records": 0, "exact_duplicates_removed": 0,
              "empty_text_records_removed": 0, "evaluation_direct_overlaps_removed": 0,
              "evaluation_family_overlaps_removed": 0}
    for raw in raw_rows:
        counts["source_records"] += 1
        try:
            row = adapt_record(raw, "metamathqa")
        except EmptyTextError:
            counts["empty_text_records_removed"] += 1
            continue
        # IDs depend on complete raw prompt/response text, not source iteration.
        row["id"] = "metamathqa:" + _digest([row["prompt"], row["response"]])
        originals = set()
        if isinstance(raw.get("original_question"), str) and raw["original_question"].strip():
            originals.add(raw["original_question"])
        row["_originals"] = originals
        row["_keys"] = {prompt_key(row["prompt"]), *(prompt_key(x) for x in originals)}
        if row["id"] in unique:
            counts["exact_duplicates_removed"] += 1
            unique[row["id"]]["_originals"].update(originals)
            unique[row["id"]]["_keys"].update(row["_keys"])
        else:
            unique[row["id"]] = row
    rows = list(unique.values())
    parent = list(range(len(rows)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    owners = {}
    for index, row in enumerate(rows):
        # Query and original-question keys share a namespace. This also joins
        # a paraphrase's query to another record's original question.
        for key in row["_keys"]:
            if key in owners:
                parent[find(index)] = find(owners[key])
            else:
                owners[key] = index
    components = defaultdict(list)
    for index, row in enumerate(rows):
        components[find(index)].append(row)
    groups = []
    for group in components.values():
        group_keys = set().union(*(row["_keys"] for row in group))
        if group_keys & held_out:
            direct = sum(bool(row["_keys"] & held_out) for row in group)
            counts["evaluation_direct_overlaps_removed"] += direct
            counts["evaluation_family_overlaps_removed"] += len(group) - direct
            continue
        group_id = _digest(sorted(group_keys))
        for row in group:
            originals = row.pop("_originals")
            row.pop("_keys")
            row["source_group"] = group_id
            if originals:
                row["original_question"] = sorted(originals)[0]
        groups.append(group)
    counts["evaluation_overlaps_removed"] = (counts["evaluation_direct_overlaps_removed"]
                                              + counts["evaluation_family_overlaps_removed"])
    counts["eligible_records"] = sum(map(len, groups))
    counts["eligible_source_groups"] = len(groups)
    return groups, counts


def _fixed_split(groups: list[list[dict]], train_size: int, validation_size: int,
                 seed: int) -> tuple[list[dict], list[dict], dict]:
    if len(groups) < 2 or sum(map(len, groups)) < train_size + validation_size:
        raise ValueError("Insufficient independent source groups/records for the requested train/validation sizes")
    groups = sorted(groups, key=lambda group: _digest([seed, "group", group[0]["source_group"]]))
    available = sum(map(len, groups))
    validation_groups, reserved = [], set()
    reserved_rows = 0
    for group in groups:
        if reserved_rows >= validation_size:
            break
        # Reserving a family for validation cannot consume required train rows.
        if available - reserved_rows - len(group) < train_size:
            continue
        validation_groups.append(group)
        reserved.add(group[0]["source_group"])
        reserved_rows += len(group)
    if reserved_rows < validation_size:
        raise ValueError("Insufficient independent source groups to fill exact quotas without family leakage")

    def select(candidates, size, label):
        # Globally rank rows after reserving validation families. Selecting
        # complete hash-ranked training families until the row quota is filled
        # would unnecessarily narrow question diversity in augmented corpora.
        ranked = sorted((row for group in candidates for row in group),
                        key=lambda row: _digest([seed, label, row["id"]]))
        selected = ranked[:size]
        if len(selected) != size:
            raise ValueError(f"Unable to select exactly {size} {label} records")
        return sorted(selected, key=lambda row: row["id"])

    validation = select(validation_groups, validation_size, "validation")
    train = select((group for group in groups if group[0]["source_group"] not in reserved), train_size, "train")
    assert_disjoint(train, validation)
    return train, validation, {
        "validation_reserved_family_records": reserved_rows,
        "unused_eligible_records": available - train_size - validation_size,
        "train_source_groups": len({row["source_group"] for row in train}),
        "validation_source_groups": len({row["source_group"] for row in validation}),
    }


def validate_math_pool(output_dir: str | Path) -> dict:
    """Validate an existing published pool without touching raw inputs or files."""
    output = Path(output_dir)
    manifest_path = output / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Incomplete or unrecognized math pool: missing {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("dataset") != "metamathqa_fixed_math":
        raise ValueError("Unrecognized math pool manifest")
    if manifest.get("selection_policy_version") != SELECTION_POLICY_VERSION:
        raise ValueError("Math pool selection policy differs; use a new output directory")
    records = {}
    for name, relative in ARTIFACTS.items():
        entry = manifest["splits"][name]
        path = output / relative
        if entry.get("path") != relative or not path.is_file() or file_sha256(path) != entry.get("sha256"):
            raise ValueError(f"Math pool artifact mismatch: {relative}")
        records[name] = load_records(path)
        ids = [row["id"] for row in records[name]]
        if len(ids) != entry["records"] or ids != entry["selected_ids"] or _digest(ids) != entry["ids_sha256"]:
            raise ValueError(f"Math pool record identity/count mismatch: {relative}")
    request = manifest["request"]
    expected = {"train": request["train_size"], "validation": request["validation_size"],
                "gsm8k": request["expected_gsm8k_count"], "math": request["expected_math_count"]}
    if any(len(records[name]) != count for name, count in expected.items()):
        raise ValueError("Math pool counts do not match requested fixed sizes")
    assert_disjoint(records["train"], records["validation"])
    held_out = {prompt_key(row["prompt"]) for name in ("gsm8k", "math") for row in records[name]}
    for row in records["train"] + records["validation"]:
        if prompt_key(row["prompt"]) in held_out or prompt_key(row.get("original_question", "")) in held_out:
            raise ValueError("Math pool contains a test overlap")
    return manifest


def prepare_math_pool(metamath_source: str | Path = DEFAULT_METAMATH_SOURCE,
                      gsm8k_source: str | Path = DEFAULT_GSM8K_SOURCE,
                      math_source: str | Path = DEFAULT_MATH_SOURCE,
                      output_dir: str | Path = DEFAULT_MATH_POOL, *, train_size: int = 50000,
                      validation_size: int = 5000, seed: int = 42, resume: bool = False,
                      expected_gsm8k_count: int = 1319, expected_math_count: int = 5000) -> dict:
    """Prepare exactly one frozen training/dev pool and complete benchmark tests.

    Publication writes the manifest last into an exclusively created directory.
    Existing output directories are never overwritten, including partial ones.
    ``resume`` validates the published artifacts and raw source snapshots.
    """
    for name, value in (("train_size", train_size), ("validation_size", validation_size),
                        ("expected_gsm8k_count", expected_gsm8k_count), ("expected_math_count", expected_math_count)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    output = Path(output_dir).expanduser().resolve()
    sources = {"metamathqa": Path(metamath_source).expanduser().resolve(),
               "gsm8k": Path(gsm8k_source).expanduser().resolve(),
               "math": Path(math_source).expanduser().resolve()}
    request = {"sources": {name: str(path) for name, path in sources.items()},
               "train_size": train_size, "validation_size": validation_size, "seed": seed,
               "expected_gsm8k_count": expected_gsm8k_count, "expected_math_count": expected_math_count}
    existing = None
    if output.exists():
        if not resume:
            raise FileExistsError(f"Math pool output already exists: {output}; --resume validates a complete matching pool")
        existing = validate_math_pool(output)
        if existing.get("request") != request:
            raise ValueError("Existing math pool request differs; use a new output directory")
    iterators, files, snapshots = {}, {}, {}
    for name, source in sources.items():
        iterators[name], files[name] = read_source(source, "train" if name == "metamathqa" else "test")
        snapshots[name] = _snapshots(files[name])
    if any(_is_eval_path(path) for path in files["metamathqa"]):
        raise ValueError("Refusing evaluation/test files as the MetaMathQA training source")
    if existing is not None:
        if existing.get("source_files") != snapshots:
            raise ValueError("Raw math source hashes changed; use a new output directory")
        # Matching byte hashes prove source counts still match the initial read.
        return existing
    gsm8k = _benchmark(iterators["gsm8k"], "gsm8k", expected_gsm8k_count)
    math = _benchmark(iterators["math"], "math", expected_math_count)
    held_out = {prompt_key(row["prompt"]) for row in gsm8k + math}
    groups, counts = _training_families(iterators["metamathqa"], held_out)
    train, validation, split_counts = _fixed_split(groups, train_size, validation_size, seed)
    if any(_snapshots(files[name]) != snapshots[name] for name in sources):
        raise ValueError("A raw math source changed during preparation; nothing was published")
    records = {"train": train, "validation": validation, "gsm8k": gsm8k, "math": math}
    manifest: dict[str, Any] = {
        "schema_version": 1, "dataset": "metamathqa_fixed_math", "request": request,
        "selection_policy_version": SELECTION_POLICY_VERSION,
        "source_files": snapshots, "source_records": counts["source_records"],
        "benchmark_source_records": {"gsm8k": len(gsm8k), "math": len(math)},
        "normalization": "Unicode NFKC, whitespace collapse, casefold",
        "split_policy": "Reserve seeded hash-ranked connected query/original-question families for validation; globally hash-rank rows in the remaining families for the exact training quota; globally hash-rank validation-reserved rows for the exact validation quota",
        "exclusion_policy": "Exclude every family touching a normalized GSM8K or MATH test prompt before selection",
        "training_text_truncation": None,
        "filter_counts": counts, "selection_counts": split_counts, "splits": {},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()  # Exclusive claim: never overwrite an unknown/partial pool.
    for name, relative in ARTIFACTS.items():
        write_jsonl(output / relative, records[name])
        ids = [row["id"] for row in records[name]]
        manifest["splits"][name] = {"path": relative, "records": len(ids),
                                    "sha256": file_sha256(output / relative),
                                    "ids_sha256": _digest(ids), "selected_ids": ids}
    # The manifest is the completion marker. An interrupted publication is
    # rejected on resume, so nobody silently trains from a partial pool.
    with (output / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    validate_math_pool(output)
    return manifest
