"""Durable, locked progress for immutable offline distillation pairs."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import tempfile
import time

from .models import check_supported, render_prompt


def pair_paths(output):
    output = Path(output)
    return {"output": output, **{
        key: output.with_suffix(suffix) for key, suffix in {
            "teacher": ".teacher.jsonl", "student": ".student.jsonl",
            "manifest": ".manifest.json", "progress": ".progress.json",
            "teacher_partial": ".teacher.partial.jsonl",
            "student_partial": ".student.partial.jsonl", "lock": ".lock",
        }.items()
    }}


@contextmanager
def pair_lock(path):
    """Keep the lock inode permanently; unlinking it could admit two writers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another pair generator holds {path}") from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_lines(path, lines):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            for line in lines:
                stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_json(path, document):
    _atomic_lines(path, [json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"])


def atomic_jsonl(path, rows):
    _atomic_lines(path, (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def append_jsonl(path, row):
    """One durable write per completion; no full-dataset snapshot per row."""
    path = Path(path)
    existed = path.exists()
    with path.open("ab") as stream:
        stream.write((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
    if not existed:
        _sync_directory(path.parent)


def read_progress_rows(path, *, recover_tail=False):
    """Recover only a torn final write, preserving the exact removed bytes."""
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    torn_offset = None
    missing_newline = False
    with path.open("rb") as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            try:
                row = json.loads(line)
            except (ValueError, UnicodeDecodeError) as error:
                if recover_tail and not line.endswith(b"\n"):
                    torn_offset = offset
                    # The backup contains only the discarded tail. The intact
                    # prefix remains in the append log, so even large logs
                    # recover without copying all previous records.
                    backup = path.with_name(path.name + f".truncated-{time.time_ns()}.bak")
                    with backup.open("xb") as saved:
                        saved.write(line)
                        saved.flush()
                        os.fsync(saved.fileno())
                    _sync_directory(path.parent)
                    print(f"Preserved interrupted JSONL tail in {backup}", flush=True)
                    break
                raise ValueError(f"Invalid persisted generation JSON in {path} at byte {offset}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Persisted generation row must be an object: {path} at byte {offset}")
            rows.append(row)
            missing_newline = not line.endswith(b"\n")
    if torn_offset is not None:
        with path.open("r+b") as stream:
            stream.truncate(torn_offset)
            stream.flush()
            os.fsync(stream.fileno())
    elif missing_newline and recover_tail:
        with path.open("ab") as stream:
            stream.write(b"\n")
            stream.flush()
            os.fsync(stream.fileno())
    return rows


def generation_identity(provenance, effective_config):
    """Only generation inputs matter, never training LR/output/device/seed overrides.

    The producer seed comes from provenance. Training can consume the same
    immutable pairs at several training seeds. Legacy manifests store dtype in
    effective_config, whereas new provenance also records it directly.
    """
    required = (
        "teacher", "student", "tokenizer", "source_sha256", "validation_sha256",
        "seed", "enable_thinking", "max_prompt_tokens", "max_new_tokens",
        "generation", "teacher_generation", "max_examples",
    )
    missing = set(required) - provenance.keys()
    if missing:
        raise ValueError(f"Pair provenance lacks generation identity fields: {sorted(missing)}")
    result = {key: provenance[key] for key in required}
    result["vocabulary_policy"] = provenance.get("vocabulary_policy", "full")
    result["vocabulary_alignment"] = provenance.get("vocabulary_alignment")
    # Missing runtime is historical unknown, never permission to mix a new
    # producer into a partially generated dataset. Only the caller validating
    # an already complete legacy artifact may allow this missing field.
    result["generation_runtime"] = provenance.get("generation_runtime")
    result["dtype"] = provenance.get("generation_dtype", effective_config.get("dtype"))
    if result["dtype"] is None:
        raise ValueError("Pair provenance does not identify generation dtype")
    return result


def require_identity(actual, expected):
    changed = sorted(key for key in actual.keys() | expected.keys() if actual.get(key) != expected.get(key))
    if changed:
        raise ValueError(f"Pair generation inputs changed: {', '.join(changed)}; use a new output path")


def validate_role_records(rows, records, tokenizer, cfg, *, vocabulary_size, complete=False):
    """Validate cached rows against raw prompts and the exact current tokenizer."""
    expected = {row["id"]: row for row in records}
    seen = set()
    for row in rows:
        identifier = row.get("id")
        if not isinstance(identifier, str) or identifier not in expected:
            raise ValueError(f"Unknown persisted generation id: {identifier!r}")
        if identifier in seen:
            raise ValueError(f"Duplicate persisted generation id: {identifier!r}")
        seen.add(identifier)
        source = expected[identifier]
        if row.get("prompt") != source["prompt"] or row.get("source_group") != source.get("source_group"):
            raise ValueError(f"Persisted generation prompt/source group changed for {identifier!r}")
        for field, limit in (("prompt_ids", cfg["max_prompt_tokens"]), ("response_ids", cfg["max_new_tokens"])):
            ids = row.get(field)
            if (not isinstance(ids, list) or not ids or len(ids) > limit
                    or any(type(token) is not int or not 0 <= token < vocabulary_size for token in ids)):
                raise ValueError(f"Invalid persisted generation {field} for {identifier!r}")
            check_supported(ids, tokenizer, f"persisted {field}")
        prompt = render_prompt(tokenizer, source["prompt"], cfg["max_prompt_tokens"], cfg["enable_thinking"])
        if row["prompt_ids"] != prompt:
            raise ValueError(f"Persisted prompt token IDs changed for {identifier!r}")
        if row.get("response") != tokenizer.decode(row["response_ids"], skip_special_tokens=True):
            raise ValueError(f"Persisted response text/token IDs disagree for {identifier!r}")
    if complete and seen != expected.keys():
        raise ValueError(f"Incomplete generation: {len(seen)}/{len(expected)} records")
    return seen
