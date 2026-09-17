"""Strict, JSON-serializable experiment settings; no implicit remote downloads."""
import copy
import json
import math
from pathlib import Path

DEFAULTS = dict(
    vocabulary_policy="full",
    output_root="/nas/Users/wyx/Baseline", seed=42, dtype="bfloat16", device="cuda:0",
    teacher_device="cuda:1", optimizer_offload=True, max_prompt_tokens=1024, max_new_tokens=1024,
    enable_thinking=False, learning_rate=1e-5, weight_decay=0.01,
    gradient_accumulation_steps=32, epochs=1, max_steps=None, warmup_ratio=0.03,
    max_grad_norm=1.0, gradient_checkpointing=True, loss_chunk_size=32,
    save_steps=100, eval_steps=100, eval_max_examples=100,
    eval_during_training=False, evaluation_metric="rouge_l",
    eval_max_new_tokens=1024, eval_rouge_tokenizer="english",
    alpha=0.1, beta=0.8, alpha_1=0.1, alpha_2=0.1,
    acceptance_k=25, proposal_block_size=5,
    generation=dict(temperature=0.7, top_p=0.8, top_k=20),
    teacher_generation=dict(temperature=0.7, top_p=0.8, top_k=20),
)
REQUIRED = {"name", "pair", "method", "teacher_model", "student_model", "train_file", "validation_file", "dataset"}

def validate_config(config):
    unknown = set(config) - set(DEFAULTS) - REQUIRED
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    missing = REQUIRED - set(config)
    if missing:
        raise ValueError(f"Missing configuration fields: {sorted(missing)}")
    cfg = copy.deepcopy(DEFAULTS)
    cfg.update(copy.deepcopy(config))
    if cfg["method"] not in {"kd", "abkd", "skd", "distillm2"}:
        raise ValueError("method must be kd, abkd, skd, or distillm2")
    if cfg["vocabulary_policy"] not in {"full", "llama3_shared"}:
        raise ValueError("vocabulary_policy must be full or llama3_shared")
    if cfg["evaluation_metric"] not in {"rouge_l", "math_accuracy"}:
        raise ValueError("evaluation_metric must be rouge_l or math_accuracy")
    if cfg["evaluation_metric"] == "math_accuracy" and cfg["eval_during_training"] is not False:
        raise ValueError("Math accuracy is evaluated separately on GSM8K and MATH; disable eval_during_training")
    if cfg["eval_rouge_tokenizer"] not in {"english", "unicode"}:
        raise ValueError("eval_rouge_tokenizer must be english or unicode")
    if cfg["dtype"] not in {"float32", "bfloat16"}:
        raise ValueError("dtype must be float32 or bfloat16")
    if type(cfg["seed"]) is not int or not 0 <= cfg["seed"] < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    for key in ("optimizer_offload", "gradient_checkpointing", "eval_during_training"):
        if type(cfg[key]) is not bool:
            raise ValueError(f"{key} must be a boolean")
    for key in ("learning_rate", "weight_decay", "warmup_ratio", "max_grad_norm", "alpha", "beta", "alpha_1", "alpha_2"):
        if type(cfg[key]) not in {int, float} or not math.isfinite(cfg[key]):
            raise ValueError(f"{key} must be finite numeric")
    if cfg["weight_decay"] < 0 or not 0 < cfg["alpha_1"] < 1 or not 0 < cfg["alpha_2"] < 1:
        raise ValueError("Invalid weight decay or skew alpha")
    for key in ("name", "pair", "dataset"):
        if not isinstance(cfg[key], str) or not cfg[key] or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for c in cfg[key]) or cfg[key] in {".", ".."}:
            raise ValueError(f"{key} must be a simple nonempty path component")
    for key in ("teacher_model", "student_model", "train_file", "validation_file", "output_root"):
        if not isinstance(cfg[key], str) or not cfg[key]:
            raise ValueError(f"{key} must be a nonempty local path")
    for key in ("max_prompt_tokens", "max_new_tokens", "gradient_accumulation_steps", "epochs", "loss_chunk_size", "save_steps", "eval_steps", "eval_max_examples", "eval_max_new_tokens", "acceptance_k", "proposal_block_size"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if cfg["max_steps"] is not None and (type(cfg["max_steps"]) is not int or cfg["max_steps"] < 1):
        raise ValueError("max_steps must be null or positive integer")
    if not 0 <= cfg["warmup_ratio"] < 1 or cfg["learning_rate"] <= 0 or cfg["max_grad_norm"] <= 0:
        raise ValueError("Invalid optimizer/scheduler setting")
    for key in ("generation", "teacher_generation"):
        value = cfg[key]
        if not isinstance(value, dict) or set(value) != {"temperature", "top_p", "top_k"}:
            raise ValueError(f"Invalid {key} fields")
        if any(type(value[k]) not in {int, float} or not math.isfinite(value[k]) for k in ("temperature", "top_p")) or value["temperature"] < 0 or not 0 < value["top_p"] <= 1 or type(value["top_k"]) is not int or value["top_k"] < 0:
            raise ValueError(f"Invalid {key} parameters")
        if cfg["method"] == "skd" and value["temperature"] == 0:
            raise ValueError("SKD sampling requires a positive temperature")
    if cfg["enable_thinking"] is not False:
        raise ValueError("This experiment protocol fixes Qwen enable_thinking=False")
    if Path(cfg["train_file"]).resolve() == Path(cfg["validation_file"]).resolve():
        raise ValueError("Training and validation must be separate files")
    return cfg

def load_config(path):
    return validate_config(json.loads(Path(path).read_text()))

def run_directory(cfg):
    return Path(cfg["output_root"]) / "runs" / cfg["pair"] / cfg["dataset"] / cfg["method"] / f"seed_{cfg['seed']}"
