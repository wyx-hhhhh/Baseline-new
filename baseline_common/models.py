"""Local checkpoint loading, exact vocabulary checks and shared chat rendering."""
import hashlib
import json
from pathlib import Path

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def model_fingerprint(path):
    root = Path(path).resolve()
    files = {p.name: sha256_file(p) for p in sorted(root.glob("*.json"))}
    for p in sorted(root.glob("*.jinja")):
        files[p.name] = sha256_file(p)
    weights = {p.name: {"size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in sorted(root.glob("*.safetensors"))}
    return {"path": str(root), "metadata_sha256": files, "weight_file_identity": weights,
            "note": "Metadata hashes and weight file size/mtime; not a full weight content hash"}

def load_tokenizer(model_path):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    if not tokenizer.chat_template:
        raise ValueError(f"Instruct tokenizer has no chat template: {model_path}")
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define EOS")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    saved_alignment = tokenizer.init_kwargs.get("baseline_vocab_alignment")
    if saved_alignment:
        tokenizer._baseline_vocab_alignment = saved_alignment
    return tokenizer

def verify_tokenizers(teacher_path, student_path, policy="full"):
    t, s = load_tokenizer(teacher_path), load_tokenizer(student_path)
    if policy == "llama3_shared":
        from .vocabulary import validate_vocabulary
        alignment = validate_vocabulary(t, s, policy=policy,
            teacher_config=json.loads((Path(teacher_path) / "config.json").read_text()),
            student_config=json.loads((Path(student_path) / "config.json").read_text()))
        # Llama 3.2's template otherwise reads the wall-clock date, invalidating
        # saved prompt IDs after midnight. Use the template's own fixed fallback.
        alignment["chat_date_string"] = "26 Jul 2024"
        s._baseline_vocab_alignment = alignment
        s.init_kwargs["baseline_vocab_alignment"] = alignment
        return s
    if policy != "full":
        raise ValueError(f"Unknown vocabulary policy: {policy}")
    if t.get_vocab() != s.get_vocab():
        raise ValueError("Teacher/student full token-to-ID maps differ; token-level KD is invalid")
    # Compare actual tokenization machinery, allowing different chat templates.
    tj, sj = json.loads(t.backend_tokenizer.to_str()), json.loads(s.backend_tokenizer.to_str())
    for key in ("model", "normalizer", "pre_tokenizer", "post_processor", "decoder", "added_tokens"):
        if tj.get(key) != sj.get(key):
            raise ValueError(f"Teacher/student tokenizer {key} differs")
    if t.eos_token_id != s.eos_token_id or t.bos_token_id != s.bos_token_id:
        raise ValueError("Teacher/student BOS/EOS semantics differ")
    return s

def bind_vocabulary(model, tokenizer):
    """Persist the explicit shared support in exports and native generation."""
    alignment = getattr(tokenizer, "_baseline_vocab_alignment", None)
    if alignment:
        if model.config.vocab_size != alignment["vocabulary_size"]:
            raise ValueError("Model output vocabulary differs from validated alignment")
        model._baseline_vocab_alignment = alignment
        model.config.baseline_vocab_alignment = alignment
        model.generation_config.suppress_tokens = alignment["excluded_ids"]
        model.generation_config.eos_token_id = stop_ids(model, tokenizer)

def mask_generation_logits(logits, model):
    """Mask unsupported output IDs without renumbering generated IDs."""
    alignment = getattr(model, "_baseline_vocab_alignment", None)
    if alignment and alignment["policy"] != "full":
        from .vocabulary import allowed_token_mask
        return logits.masked_fill(~allowed_token_mask(alignment, device=logits.device), float("-inf"))
    return logits

def check_supported(ids, owner, context="input"):
    alignment = getattr(owner, "_baseline_vocab_alignment", None)
    if alignment:
        from .vocabulary import assert_supported_token_ids
        assert_supported_token_ids(ids, alignment, context=context)

def render_prompt(tokenizer, prompt, max_prompt_tokens, enable_thinking=False):
    alignment = getattr(tokenizer, "_baseline_vocab_alignment", {})
    template_kwargs = {"date_string": alignment["chat_date_string"]} if "chat_date_string" in alignment else {}
    ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
                                         add_generation_prompt=True, enable_thinking=enable_thinking, **template_kwargs)
    if len(ids) > max_prompt_tokens:
        # Preserve the template/header and assistant prefix by shortening the user content.
        content = tokenizer.encode(prompt, add_special_tokens=False)
        while len(ids) > max_prompt_tokens and content:
            content = content[:max(0, len(content) - max(1, len(ids) - max_prompt_tokens))]
            shortened = tokenizer.decode(content, skip_special_tokens=False)
            ids = tokenizer.apply_chat_template([{"role": "user", "content": shortened}], tokenize=True,
                                                 add_generation_prompt=True, enable_thinking=enable_thinking, **template_kwargs)
    if len(ids) > max_prompt_tokens or not ids:
        raise ValueError("max_prompt_tokens is too small for the chat template")
    check_supported(ids, tokenizer, "rendered prompt")
    return ids

def encode_reference(tokenizer, prompt, response, max_prompt_tokens, max_response_tokens, enable_thinking=False):
    prefix = render_prompt(tokenizer, prompt, max_prompt_tokens, enable_thinking)
    answer = tokenizer.encode(response, add_special_tokens=False)
    if answer and answer[-1] == tokenizer.eos_token_id:
        answer = answer[:-1]
    answer = answer[:max_response_tokens - 1] + [tokenizer.eos_token_id]
    check_supported(answer, tokenizer, "reference response")
    return {"input_ids": prefix + answer, "labels": [-100] * len(prefix) + answer}

def load_model(path, dtype="bfloat16", device="cuda:0", gradient_checkpointing=False):
    import torch
    from transformers import AutoModelForCausalLM
    if not Path(path).is_dir():
        raise FileNotFoundError(f"Local model directory missing: {path}")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run scripts/check_environment.py on a GPU node")
    if device.type == "cuda" and dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Selected GPU does not support BF16")
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=getattr(torch, dtype),
             attn_implementation="sdpa", local_files_only=True, trust_remote_code=False)
    model.to(device)
    saved_alignment = getattr(model.config, "baseline_vocab_alignment", None)
    if saved_alignment:
        model._baseline_vocab_alignment = saved_alignment
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
    return model

def stop_ids(model, tokenizer):
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos or [])
    excluded = set(getattr(model, "_baseline_vocab_alignment", {}).get("excluded_ids", []))
    return [i for i in dict.fromkeys(eos + [tokenizer.eos_token_id]) if i not in excluded]

def generate_response(model, tokenizer, prompt_ids, max_new_tokens, generation):
    import torch
    old_mode = model.training
    model.eval()
    ids = torch.tensor([prompt_ids], device=next(model.parameters()).device, dtype=torch.long)
    check_supported(prompt_ids, model, "generation prompt")
    kwargs = dict(do_sample=generation["temperature"] > 0, max_new_tokens=max_new_tokens,
                  eos_token_id=stop_ids(model, tokenizer), pad_token_id=tokenizer.pad_token_id,
                  use_cache=True, return_dict_in_generate=False)
    if kwargs["do_sample"]:
        kwargs.update(generation)
    else:
        # Evaluation is greedy irrespective of inherited checkpoint defaults.
        kwargs.update(num_beams=1, num_beam_groups=1, num_return_sequences=1,
                      penalty_alpha=None, temperature=1.0, top_p=1.0, top_k=50)
    alignment = getattr(model, "_baseline_vocab_alignment", None)
    if alignment:
        kwargs["suppress_tokens"] = alignment["excluded_ids"]
    try:
        with torch.no_grad():
            result = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), **kwargs)
        return result[0, ids.shape[1]:].tolist()
    finally:
        model.train(old_mode)
