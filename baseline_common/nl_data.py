"""Frozen local natural-language benchmark adapters with all reference answers.

Only the named MiniLLM ``valid.jsonl`` files are inputs. Training overlap is
removed from evaluation, never from an already running training experiment.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .data import _digest, file_sha256, prompt_key, write_jsonl


BENCHMARKS = ("dolly", "selfinst", "super_natural", "unnatural", "vicuna")
SOURCE_FILES = {
    "dolly": ("DollyEval/valid.jsonl",),
    "selfinst": ("SelfInst/valid.jsonl",),
    "super_natural": tuple(f"Super-Natural/{part}/valid.jsonl"
                           for part in ("0_2", "3_6", "6_10", "11_")),
    "unnatural": tuple(f"Unnatural/{part}/valid.jsonl"
                      for part in ("0_2", "3_5", "6_10", "11_")),
    "vicuna": ("VicunaEval/valid.jsonl",),
}
ADAPTER_VERSION = 1


def _rows(path: Path):
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path}:{number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object in {path}:{number}")
            yield number, row


def _text(value, label: str, *, empty: bool = False) -> str:
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ValueError(f"{label} must be {'text' if empty else 'nonempty text'}")
    return value


def _prompt(row: dict, *, canonical: bool = False) -> str:
    if canonical and "prompt" in row:
        return _text(row["prompt"], "prompt")
    instruction = _text(row.get("instruction"), "instruction")
    context = _text(row.get("input", row.get("context", "")), "input", empty=True)
    # Rebuild the raw user turn, so the actual model chat template is applied
    # exactly once. The old Alpaca-formatted source prompt is provenance only.
    return instruction + ("\n\n" + context if context.strip() else "")


def _adapt(row: dict, benchmark: str, path: Path, number: int) -> dict:
    prompt = _prompt(row)
    references = row.get("output")
    if isinstance(references, str):
        references = [references]
    if not isinstance(references, list) or not references:
        raise ValueError(f"{path}:{number}: output must be text or a nonempty list of texts")
    references = [_text(value, f"{path}:{number}: reference") for value in references]
    result = {
        "id": benchmark + ":" + _digest([prompt, references]),
        "prompt": prompt,
        "response": references[0],
        "references": references,
        "benchmark": benchmark,
        "source_rows": [{"path": str(path), "line": number,
                         "subset": path.parent.name if len(SOURCE_FILES[benchmark]) > 1 else "."}],
    }
    topic = row.get("topic", row.get("category"))
    if topic is not None:
        result["topic"] = _text(topic, f"{path}:{number}: topic", empty=True)
    return result


def load_nl_records(path: str | Path) -> list[dict]:
    """Read canonical evaluation rows; reject scalarized or missing references."""
    records, identifiers = [], set()
    for number, row in _rows(Path(path)):
        identifier = _text(row.get("id"), f"{path}:{number}: id")
        if identifier in identifiers:
            raise ValueError(f"Duplicate evaluation id: {identifier}")
        identifiers.add(identifier)
        _text(row.get("prompt"), f"{path}:{number}: prompt")
        references = row.get("references")
        if not isinstance(references, list) or not references:
            raise ValueError(f"{path}:{number}: references must be a nonempty list")
        for reference in references:
            _text(reference, f"{path}:{number}: reference")
        if row.get("response") != references[0]:
            raise ValueError(f"{path}:{number}: response must equal references[0]")
        if row.get("benchmark") not in BENCHMARKS:
            raise ValueError(f"{path}:{number}: unrecognized benchmark")
        if "source_rows" in row:
            if not isinstance(row["source_rows"], list) or not row["source_rows"]:
                raise ValueError(f"{path}:{number}: source_rows must be a nonempty list")
            for source in row["source_rows"]:
                if (not isinstance(source, dict) or not isinstance(source.get("path"), str)
                        or not source["path"] or type(source.get("line")) is not int
                        or source["line"] < 1 or not isinstance(source.get("subset"), str)):
                    raise ValueError(f"{path}:{number}: malformed source row provenance")
        records.append(row)
    if not records:
        raise ValueError(f"Evaluation dataset is empty: {path}")
    return records


def _snapshot(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required local evaluation/exclusion source is missing: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path)}


def validate_nl_sources(manifest: dict) -> None:
    """Reject changed or missing original benchmarks and training exclusions."""
    for entry in manifest["source_files"] + manifest["exclusion_files"]:
        path = Path(entry["path"])
        if (not path.is_file() or path.stat().st_size != entry["bytes"]
                or file_sha256(path) != entry["sha256"]):
            raise ValueError(f"Natural-language source changed or is missing: {path}")


def validate_nl_pool(output_dir: str | Path, *, check_sources: bool = True) -> dict:
    """Validate frozen membership, references, artifact hashes and source hashes."""
    output = Path(output_dir).expanduser().resolve()
    manifest_path = output / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Incomplete natural-language evaluation pool: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != 1
            or manifest.get("dataset") != "natural_language_evaluation"
            or manifest.get("adapter_version") != ADAPTER_VERSION):
        raise ValueError("Unrecognized natural-language evaluation manifest")
    benchmarks = manifest["request"]["benchmarks"]
    if (not benchmarks or len(set(benchmarks)) != len(benchmarks)
            or any(name not in BENCHMARKS for name in benchmarks)
            or set(manifest["benchmarks"]) != set(benchmarks)):
        raise ValueError("Natural-language manifest benchmark selection mismatch")
    source_paths = {entry["path"] for entry in manifest["source_files"]}
    for benchmark in benchmarks:
        entry = manifest["benchmarks"][benchmark]
        path = output / f"{benchmark}.jsonl"
        if (entry.get("path") != path.name or not path.is_file()
                or file_sha256(path) != entry.get("sha256")):
            raise ValueError(f"Natural-language artifact mismatch: {path}")
        records = load_nl_records(path)
        identifiers = [row["id"] for row in records]
        if (len(records) != entry["records"] or identifiers != entry["selected_ids"]
                or _digest(identifiers) != entry["ids_sha256"]):
            raise ValueError(f"Natural-language artifact count/identity mismatch: {path}")
        for row in records:
            if (row["benchmark"] != benchmark
                    or row["id"] != benchmark + ":" + _digest([row["prompt"], row["references"]])
                    or not row.get("source_rows")
                    or any(source["path"] not in source_paths for source in row["source_rows"])):
                raise ValueError(f"Natural-language artifact metadata mismatch: {path}")
        if (entry["source_records"] - entry["exact_duplicates_merged"]
                - entry["training_overlaps_removed"] != len(records)
                or len(entry["excluded_records"]) != entry["training_overlaps_removed"]):
            raise ValueError(f"Natural-language manifest counts do not reconcile: {benchmark}")
    if check_sources:
        validate_nl_sources(manifest)
    return manifest


def prepare_nl_pool(source_root: str | Path, output_dir: str | Path,
                    benchmarks: Iterable[str] | None = None,
                    exclude_files: Iterable[str | Path] = (), resume: bool = True) -> dict:
    """Publish one immutable pool, merging exact copies and excluding train prompts.

    All named ``valid.jsonl`` files must exist. Different references for the same
    prompt remain distinct source examples. Only identical prompt/reference
    tuples are merged; their source locations remain visible. Existing partial
    output is never overwritten; use a fresh directory if preparation failed.
    """
    root = Path(source_root).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    selected = list(BENCHMARKS if benchmarks is None else benchmarks)
    if (not selected or len(set(selected)) != len(selected)
            or any(name not in BENCHMARKS for name in selected)):
        raise ValueError(f"benchmarks must be a nonempty unique selection from {BENCHMARKS}")
    selected = [name for name in BENCHMARKS if name in selected]
    exclusions = sorted({Path(path).expanduser().resolve() for path in exclude_files})
    request = {"source_root": str(root), "benchmarks": selected,
               "exclude_files": [str(path) for path in exclusions]}
    if output.exists():
        if not resume:
            raise FileExistsError(f"Evaluation pool already exists: {output}")
        existing = validate_nl_pool(output)
        if existing["request"] != request:
            raise ValueError("Evaluation preparation request changed; use a new output directory")
        return existing

    exclusion_keys: dict[str, set[str]] = {}
    exclusion_snapshots = []
    for path in exclusions:
        snapshot = _snapshot(path)
        count = 0
        for number, row in _rows(path):
            try:
                key = prompt_key(_prompt(row, canonical=True))
            except ValueError as error:
                raise ValueError(f"Invalid training exclusion row in {path}:{number}: {error}") from error
            exclusion_keys.setdefault(key, set()).add(str(path))
            count += 1
        if not count:
            raise ValueError(f"Training exclusion file is empty: {path}")
        snapshot["records"] = count
        exclusion_snapshots.append(snapshot)

    sources, prepared, benchmark_entries = [], {}, {}
    for benchmark in selected:
        unique = {}
        raw_count = 0
        for relative in SOURCE_FILES[benchmark]:
            path = root / relative
            snapshot = _snapshot(path)
            count = 0
            for number, raw in _rows(path):
                row = _adapt(raw, benchmark, path, number)
                if row["id"] in unique:
                    unique[row["id"]]["source_rows"].extend(row["source_rows"])
                else:
                    unique[row["id"]] = row
                count += 1
            if not count:
                raise ValueError(f"Benchmark source is empty: {path}")
            snapshot.update(records=count, benchmark=benchmark,
                            subset=path.parent.name if len(SOURCE_FILES[benchmark]) > 1 else ".")
            sources.append(snapshot)
            raw_count += count
        records, excluded = [], []
        for row in unique.values():
            matches = exclusion_keys.get(prompt_key(row["prompt"]))
            if matches:
                excluded.append({"id": row["id"], "prompt_sha256": _digest(prompt_key(row["prompt"])),
                                 "source_rows": row["source_rows"], "matched_files": sorted(matches)})
            else:
                records.append(row)
        if not records:
            raise ValueError(f"No held-out examples remain for {benchmark} after training exclusions")
        prepared[benchmark] = records
        benchmark_entries[benchmark] = {
            "path": f"{benchmark}.jsonl", "records": len(records),
            "source_records": raw_count, "exact_duplicates_merged": raw_count - len(unique),
            "training_overlaps_removed": len(excluded),
            "overlap_source_rows_removed": sum(len(row["source_rows"]) for row in excluded),
            "excluded_records": excluded,
            "selected_ids": [row["id"] for row in records],
            "ids_sha256": _digest([row["id"] for row in records]),
        }
    manifest = {
        "schema_version": 1, "dataset": "natural_language_evaluation", "adapter_version": ADAPTER_VERSION,
        "request": request, "source_files": sources, "exclusion_files": exclusion_snapshots,
        "policies": {
            "prompt": "instruction plus nonempty input/context; model chat template applied by evaluator",
            "references": "all source references, in source order",
            "duplicates": "merge identical prompt/reference tuples; retain every source row location",
            "training_exclusion": "union of Unicode NFKC, whitespace-collapsed, casefolded training prompts",
            "metric": "maximum ROUGE-L F1 across references for each example, then mean across examples",
        },
        "benchmarks": benchmark_entries,
    }
    # A concurrently edited source must not produce a pool with mixed versions.
    validate_nl_sources(manifest)
    output.mkdir(parents=True, exist_ok=False)
    for benchmark, records in prepared.items():
        path = output / benchmark_entries[benchmark]["path"]
        write_jsonl(path, records)
        benchmark_entries[benchmark]["sha256"] = file_sha256(path)
    # Commit marker last: a crash leaves an explicitly incomplete directory.
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                                           encoding="utf-8")
    return manifest
