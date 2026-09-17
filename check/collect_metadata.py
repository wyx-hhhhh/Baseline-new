"""Collect public model/config and release metadata; never downloads model weights."""
import concurrent.futures
import datetime
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")
OUT = Path(__file__).resolve().parent / "evidence"
OUT.mkdir(exist_ok=True)
MODELS = ["Qwen/Qwen3-8B", "Qwen/Qwen3-1.7B", "meta-llama/Meta-Llama-3-8B", "meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.1-8B"]
PACKAGES = {"torch": "2.6.0", "transformers": "4.51.3", "accelerate": "1.6.0", "peft": "0.15.2", "trl": "0.9.6", "deepspeed": "0.16.7", "datasets": "3.5.0", "tokenizers": "0.21.1", "huggingface-hub": "0.30.2", "numpy": "1.26.4", "vllm": "0.8.5", "flash-attn": "2.7.4.post1"}

def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Baseline-research-audit/1.0"}), timeout=45) as res:
        return res.read()

def model_info(model):
    item = {"model_id": model, "source": "https://huggingface.co/" + model}
    try:
        info = json.loads(get("https://huggingface.co/api/models/" + model))
        item.update(revision=info.get("sha"), gated=info.get("gated"), api_config=info.get("config"), pipeline_tag=info.get("pipeline_tag"))
        folder = OUT / "models" / model.replace("/", "--")
        folder.mkdir(parents=True, exist_ok=True)
        for filename in ["config.json", "generation_config.json", "tokenizer_config.json", "tokenizer.json"]:
            url = f"https://huggingface.co/{model}/resolve/{item['revision']}/{filename}"
            try:
                raw = get(url)
                parsed = json.loads(raw)
                (folder / filename).write_bytes(raw)
                item[filename] = {"source": url, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
                if filename in ["config.json", "generation_config.json"]:
                    item[filename]["values"] = parsed
                elif filename == "tokenizer_config.json":
                    item[filename]["values"] = {k: parsed.get(k) for k in ["bos_token", "eos_token", "pad_token", "tokenizer_class", "model_max_length"]}
                else:
                    item[filename]["base_vocab_count"] = len(parsed["model"]["vocab"])
                    item[filename]["added_token_count"] = len(parsed.get("added_tokens", []))
            except Exception as exc:
                item[filename] = {"source": url, "error": str(exc)}
    except Exception as exc:
        item["error"] = str(exc)
    return item

def package_info(pair):
    name, version = pair
    url = f"https://pypi.org/pypi/{name}/{version}/json"
    try:
        data = json.loads(get(url))
        info = data["info"]
        return {"name": name, "version": version, "source": url, "requires_python": info.get("requires_python"), "requires_dist": info.get("requires_dist"), "files": [{k: f.get(k) for k in ["filename", "url", "digests", "upload_time_iso_8601", "yanked"]} for f in data["urls"]]}
    except Exception as exc:
        return {"name": name, "version": version, "source": url, "error": str(exc)}

if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        models = list(pool.map(model_info, MODELS))
        packages = list(pool.map(package_info, PACKAGES.items()))
    result = {"retrieved_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "models": models, "packages": packages}
    (OUT / "public_metadata.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"models": [{"id": x["model_id"], "revision": x.get("revision"), "config": x.get("config.json"), "tokenizer": x.get("tokenizer.json")} for x in models], "packages": [{k: x.get(k) for k in ["name", "version", "requires_python", "error"]} for x in packages]}, indent=2))
