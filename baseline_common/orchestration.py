"""Two independent, sequential experiment queues with artifact-based recovery."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
import zipfile


ROOT = Path(__file__).resolve().parents[1]
METHOD_ORDER = ("kd", "abkd", "skd", "distillm2")
PAIR_NAMES = {"qwen": "qwen3_8b_1p7b", "llama": "llama3_8b_llama32_1b"}


@dataclass(frozen=True)
class Stage:
    group: str
    kind: str
    method: str
    config: dict
    config_path: Path
    output_path: Path
    input_path: Path | None = None
    model_path: Path | None = None

    @property
    def key(self):
        return f"{self.group}:{self.method}:seed_{self.config['seed']}:{self.kind}"


def _read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with temporary.open("w") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def model_export_complete(path):
    """Check saved export structure; completion markers certify writer order."""
    path = Path(path)
    try:
        config = _read_json(path / "config.json")
        if not isinstance(config.get("model_type"), str) or type(config.get("vocab_size")) is not int or config["vocab_size"] < 1:
            return False
        if not isinstance(_read_json(path / "tokenizer_config.json"), dict):
            return False
        tokenizer = _read_json(path / "tokenizer.json")
        if not isinstance(tokenizer.get("model"), dict) or not isinstance(tokenizer["model"].get("type"), str):
            return False
        index = path / "model.safetensors.index.json"
        if index.is_file():
            mapping = _read_json(index).get("weight_map", {})
            if not mapping:
                return False
            names = set(mapping.values())
            if any(Path(name).name != name for name in names):
                return False
        else:
            names = {"model.safetensors"}
        from safetensors import safe_open
        for name in names:
            if not (path / name).is_file():
                return False
            with safe_open(str(path / name), framework="pt", device="cpu") as weights:
                keys = set(weights.keys())
                if not keys:
                    return False
                if index.is_file() and any(key not in keys for key, shard in mapping.items() if shard == name):
                    return False
        return True
    except Exception:
        return False


def _state_archive_complete(path):
    """Validate ZIP structure and metadata CRC without reading huge tensor blocks."""
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            metadata = [name for name in names if name == "data.pkl" or name.endswith("/data.pkl")]
            versions = [name for name in names if name == "version" or name.endswith("/version")]
            if len(metadata) != 1 or len(versions) != 1:
                return False
            if not archive.read(metadata[0]) or not archive.read(versions[0]):
                return False
            length = path.stat().st_size
            if any(info.header_offset < 0 or info.header_offset + info.compress_size + 30 > length
                   for info in archive.infolist()):
                return False
            return length >= 1024 * 1024 or archive.testzip() is None
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile, EOFError):
        return False


def _checkpoint_info(path):
    path = Path(path)
    match = re.fullmatch(r"step_(\d+)", path.name)
    if not match or not (path / "complete.json").is_file():
        return None
    try:
        info = _read_json(path / "complete.json")
        if type(info.get("step")) is not int or info["step"] != int(match.group(1)) or info["step"] < 1:
            raise ValueError("invalid completed step")
        world = info.get("world_size")
        if type(world) is not int or world < 1:
            raise ValueError("invalid checkpoint world size")
        for file in [path / "training.pt", *(path / f"rng_rank{i}.pt" for i in range(world))]:
            if not file.is_file() or not _state_archive_complete(file):
                raise ValueError(f"missing or invalid checkpoint state: {file.name}")
        if not model_export_complete(path / "student"):
            raise ValueError("student export is incomplete")
        return info
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        raise ValueError(f"Checkpoint claims completion but is invalid: {path}: {exc}") from exc


def latest_checkpoint(output_dir):
    candidates = []
    for path in (Path(output_dir) / "checkpoints").glob("step_*"):
        info = _checkpoint_info(path)
        if info:
            candidates.append((info["step"], path))
    return max(candidates, key=lambda pair: pair[0])[1] if candidates else None


def is_training_complete(output_dir, expected_steps=None):
    output = Path(output_dir)
    try:
        result = _read_json(output / "result.json")
        if result.get("complete") is not True or type(result.get("step")) is not int or result["step"] < 1:
            return False
        if expected_steps is not None and result["step"] != expected_steps:
            return False
        checkpoint = latest_checkpoint(output)
        if checkpoint is None or _checkpoint_info(checkpoint)["step"] != result["step"]:
            return False
        if Path(result.get("last_checkpoint", "")).resolve() != checkpoint.resolve():
            return False
        if not model_export_complete(output / "final"):
            return False
        # Legacy successful exports predate this optional extra commit marker.
        marker = output / "final" / "complete.json"
        if "interrupted" in result and not marker.is_file():
            return False
        if marker.is_file() and _read_json(marker).get("step") != result["step"]:
            return False
        return True
    except (OSError, ValueError, TypeError, AttributeError):
        return False


def _total_steps(config, world=1):
    if config.get("max_steps") is not None:
        return config["max_steps"]
    with open(config["train_file"]) as stream:
        count = sum(bool(line.strip()) for line in stream)
    if count == 0:
        raise ValueError("Cannot train on an empty dataset")
    return math.ceil(count / (world * config["gradient_accumulation_steps"])) * config["epochs"]


def child_environment(gpus):
    environment = os.environ.copy()
    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK", "ROLE_RANK",
                 "ROLE_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "TORCHELASTIC_RUN_ID",
                 "TORCHELASTIC_RESTART_COUNT", "TORCHELASTIC_MAX_RESTARTS", "TORCHELASTIC_USE_AGENT_STORE"):
        environment.pop(name, None)
    environment.update(CUDA_VISIBLE_DEVICES=",".join(map(str, gpus)),
                       CUDA_DEVICE_ORDER="PCI_BUS_ID", PYTHONUNBUFFERED="1")
    environment.setdefault("OMP_NUM_THREADS", "8")
    return environment


class Runner:
    def __init__(self, groups, gpu_groups, state_dir, command_builder=None,
                 manifest_builder=None, lock_dir=None):
        self.groups = groups
        self.gpu_groups = {key: tuple(map(str, value)) for key, value in gpu_groups.items()}
        if set(groups) != set(self.gpu_groups):
            raise ValueError("Every experiment group needs its own GPU pair")
        gpus = [gpu for group in self.gpu_groups.values() for gpu in group]
        if any(len(value) != 2 for value in self.gpu_groups.values()) or len(gpus) != len(set(gpus)):
            raise ValueError("GPU groups must contain two GPUs each and must not overlap")
        self.state_dir = Path(state_dir)
        self.command_builder = command_builder or self._command
        self.manifest_builder = manifest_builder
        self.lock_dir = Path(lock_dir) if lock_dir else Path(f"/tmp/baseline-gpu-locks-{os.getuid()}")
        self.stop = threading.Event()
        self._mutex = threading.Lock()
        self._children = {}
        self._locks = []
        self._gpu_locks = {}
        self._group_locks = {}
        self._global_lock = None
        self._signal_count = 0

    @staticmethod
    def _command(stage, resume):
        if stage.kind == "pairs":
            return [sys.executable, str(ROOT / "scripts/generate_pairs.py"), "--config", str(stage.config_path),
                    "--input", str(stage.input_path), "--output", str(stage.output_path), "--resume"]
        if stage.kind == "evaluate":
            if stage.model_path is None:
                raise ValueError("An evaluation stage requires an exported student")
            return [sys.executable, str(ROOT / "scripts/evaluate.py"), "--config", str(stage.config_path),
                    "--model", str(stage.model_path), "--data", str(stage.input_path),
                    "--output", str(stage.output_path), "--device", stage.config["device"], "--resume"]
        command = [sys.executable, str(ROOT / "scripts/train.py"), "--config", str(stage.config_path)]
        if resume:
            command += ["--resume", str(resume)]
        return command

    def _lock(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(descriptor)
            raise RuntimeError(f"Another live launcher/worker holds {path}; wait for it to exit") from None
        self._locks.append(descriptor)
        return descriptor

    def _status(self, group, **fields):
        atomic_json(self.state_dir / f"{group}.status.json",
                    {"updated_at": datetime.now(timezone.utc).isoformat(), "group": group, **fields})

    def _log(self, group, message):
        with self._mutex:
            print(f"[{group}] {message}", flush=True)

    def _prepare_training(self, stage):
        from .train import build_run_manifest, validate_run_manifest
        expected = self.manifest_builder(stage.config) if self.manifest_builder else build_run_manifest(stage.config, world=1)
        output = Path(stage.output_path)
        steps = _total_steps(stage.config, expected.get("world_size", 1))
        if not output.exists() or not any(output.iterdir()):
            return "start", None, steps
        manifest_file = output / "manifest.json"
        if not manifest_file.is_file():
            entries = list(output.iterdir())
            if entries and all(entry.is_file() and re.fullmatch(r"\.manifest\.json\..+\.tmp", entry.name)
                               for entry in entries):
                archive = output.with_name(f"{output.name}.attempt_{time.time_ns()}")
                output.rename(archive)
                self._log(stage.group, f"Preserved an interrupted initial manifest write at {archive}")
                return "restart", None, steps
            raise ValueError(f"Nonempty run has no valid manifest; refusing to overwrite: {output}")
        validate_run_manifest(_read_json(manifest_file), expected)
        if is_training_complete(output, steps):
            return "skip", None, steps
        checkpoint = latest_checkpoint(output)
        if checkpoint:
            if _checkpoint_info(checkpoint)["step"] > steps:
                raise ValueError("Checkpoint step exceeds the requested training budget")
            return "resume", checkpoint, steps
        archive = output.with_name(f"{output.name}.attempt_{time.time_ns()}")
        output.rename(archive)
        self._log(stage.group, f"No committed checkpoint yet; preserved interrupted attempt at {archive}")
        return "restart", None, steps

    def _run_child(self, stage, resume):
        group = stage.group
        log_path = self.state_dir / "logs" / group / f"{stage.method}_seed_{stage.config['seed']}_{stage.kind}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = self.command_builder(stage, resume)
        descriptors = (self._global_lock, self._group_locks[group],
                       *(self._gpu_locks[gpu] for gpu in self.gpu_groups[group]))
        with log_path.open("a", buffering=1) as stream:
            stream.write(f"\n=== {datetime.now(timezone.utc).isoformat()} {json.dumps(command)} ===\n")
            stream.flush()
            child = subprocess.Popen(command, cwd=ROOT, env=child_environment(self.gpu_groups[group]),
                                     stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                                     pass_fds=descriptors)
            with self._mutex:
                self._children[group] = child
            self._status(group, stage=stage.key, status="running", pid=child.pid,
                         command=command, log=str(log_path), resume=str(resume) if resume else None,
                         gpus=self.gpu_groups[group])
            signaled = False
            try:
                while True:
                    if self.stop.is_set() and not signaled:
                        try:
                            os.killpg(child.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                        signaled = True
                    try:
                        return child.wait(timeout=0.5), log_path
                    except subprocess.TimeoutExpired:
                        continue
            finally:
                with self._mutex:
                    self._children.pop(group, None)

    def _run_group(self, group, stages):
        try:
            for stage in stages:
                if self.stop.is_set():
                    self._status(group, status="stopped")
                    return False
                resume, steps = None, None
                if stage.kind == "train":
                    action, resume, steps = self._prepare_training(stage)
                    if action == "skip":
                        self._log(group, f"Skip completed {stage.method}, seed {stage.config['seed']}")
                        self._status(group, stage=stage.key, status="skipped_complete")
                        continue
                elif stage.kind == "pairs":
                    action = "prepare/resume pairs"
                elif stage.kind == "evaluate":
                    if stage.model_path is None or not model_export_complete(stage.model_path):
                        raise ValueError(f"Exported student is missing or incomplete: {stage.model_path}")
                    label = "math accuracy" if stage.config.get("evaluation_metric") == "math_accuracy" else "ROUGE-L"
                    action = f"evaluate/resume {label}"
                else:
                    raise ValueError(f"Unknown stage kind: {stage.kind}")
                self._log(group, f"{action}: {stage.method}, seed {stage.config['seed']}; GPUs {','.join(self.gpu_groups[group])}")
                code, log_path = self._run_child(stage, resume)
                if code != 0:
                    self._status(group, stage=stage.key, status="interrupted" if code in (75, -15, -2, 130) else "failed",
                                 exit_code=code, log=str(log_path))
                    self._log(group, f"Stopped at {stage.method}: exit {code}; see {log_path}")
                    return False
                if stage.kind == "train" and not is_training_complete(stage.output_path, steps):
                    raise RuntimeError(f"Worker exited without a complete training result: {stage.output_path}")
                if stage.kind == "evaluate":
                    from .data import file_sha256
                    completed = _read_json(Path(stage.output_path) / "manifest.json")
                    if completed.get("complete") is not True:
                        raise RuntimeError("Evaluation worker exited without committed results")
                    for name in ("metrics.json", "predictions.jsonl"):
                        path = Path(stage.output_path) / name
                        if completed.get("files", {}).get(name) != file_sha256(path):
                            raise RuntimeError(f"Incomplete or changed evaluation output: {path}")
                self._status(group, stage=stage.key, status="completed", log=str(log_path))
            self._status(group, status="completed")
            self._log(group, "All requested baselines complete")
            return True
        except Exception as exc:
            self._status(group, status="failed", error=f"{type(exc).__name__}: {exc}")
            self._log(group, str(exc))
            return False

    def _interrupt(self, signum, _frame):
        self._signal_count += 1
        self.stop.set()
        if self._signal_count == 1:
            print("Stopping workers at a safe boundary; checkpoint/pair writes may take time. Run the same command to resume.", flush=True)
        else:
            print("Forcing worker exit; recovery will use the last committed artifacts.", flush=True)
            with self._mutex:
                children = list(self._children.values())
            for child in children:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def run(self):
        old_handlers = {}
        try:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            controller = "runner.lock" if len(self.groups) != 1 else f"runner_{next(iter(self.groups))}.lock"
            self._global_lock = self._lock(self.state_dir / controller)
            for group in sorted(self.groups):
                self._group_locks[group] = self._lock(self.state_dir / f"group_{group}.lock")
            for gpu in sorted({gpu for values in self.gpu_groups.values() for gpu in values}):
                if not re.fullmatch(r"\d+", gpu):
                    raise ValueError("GPU IDs must be nonnegative integer physical indices")
                self._gpu_locks[gpu] = self._lock(self.lock_dir / f"gpu_{gpu}.lock")
            if threading.current_thread() is threading.main_thread():
                for signum in (signal.SIGINT, signal.SIGTERM):
                    old_handlers[signum] = signal.signal(signum, self._interrupt)
            # Configs are immutable within a campaign so a restart never changes
            # scheduler budgets or silently uses new hyperparameters.
            for stages in self.groups.values():
                for stage in stages:
                    path = Path(stage.config_path)
                    if path.exists() and _read_json(path) != stage.config:
                        raise ValueError(f"Resolved configuration changed: {path}; use original arguments or a new output root")
                    if not path.exists():
                        atomic_json(path, stage.config)
            with ThreadPoolExecutor(max_workers=len(self.groups)) as pool:
                futures = [pool.submit(self._run_group, group, stages) for group, stages in self.groups.items()]
                success = [future.result() for future in futures]
            return 130 if self.stop.is_set() else (0 if all(success) else 1)
        except Exception as exc:
            print(f"Launcher error: {exc}", file=sys.stderr, flush=True)
            return 1
        finally:
            for signum, previous in old_handlers.items():
                signal.signal(signum, previous)
            # Do not LOCK_UN: inherited worker descriptors must retain locks if
            # this parent dies, preventing a second launcher duplicating work.
            for descriptor in reversed(self._locks):
                os.close(descriptor)
            self._locks.clear()
