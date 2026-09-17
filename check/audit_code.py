"""Read-only repository audit and small CPU checks; does not train any LLM."""
import ast
import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "check" / "evidence"
OUT.mkdir(exist_ok=True)

def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(ROOT / repo), *args], text=True).strip()

def read(name):
    return (ROOT / name).read_text(encoding="utf-8")

result = {"audited_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "python": sys.version, "repos": {}, "syntax": {}, "checks": {}}
for repo in ["abkd", "distillm", "distillm-2", "google-research"]:
    result["repos"][repo] = {"commit": git(repo, "rev-parse", "HEAD"), "origin": git(repo, "remote", "get-url", "origin"), "status": git(repo, "status", "--porcelain")}
for name in ["torch", "transformers", "accelerate", "peft", "trl", "deepspeed", "numpy", "tokenizers"]:
    try:
        result.setdefault("installed_versions", {})[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        result.setdefault("installed_versions", {})[name] = None

for folder in ["abkd/distillation_llm", "distillm", "distillm-2/src", "distillm-2/generate", "google-research/speculative_kd"]:
    files = list((ROOT / folder).rglob("*.py"))
    failures = []
    for file in files:
        try:
            ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
        except (SyntaxError, UnicodeError) as exc:
            failures.append({"path": str(file.relative_to(ROOT)), "error": str(exc)})
    result["syntax"][folder] = {"files": len(files), "failures": failures}

tree = ast.parse(read("google-research/speculative_kd/train/ddp_skd.py"))
scope = []
for node in ast.walk(tree):
    if isinstance(node, ast.With) and any("torch.no_grad()" in ast.unparse(i.context_expr) for i in node.items):
        calls = [{"line": n.lineno, "expression": ast.unparse(n)} for n in ast.walk(node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "assistant_model"]
        if calls:
            scope.append({"no_grad_line": node.lineno, "scope_end": node.end_lineno, "student_calls": calls})
result["checks"]["skd_student_no_grad_scopes"] = scope
checks = result["checks"]
try:
    adaptive_threshold = None
    _ = adaptive_threshold * (1 - 1 / 10)
except TypeError as exc:
    checks["distillm_nonadaptive_threshold_expression"] = str(exc)

import torch
import torch.nn.functional as F
import numpy as np
torch.manual_seed(7)
result["runtime"] = {"torch": torch.__version__, "cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available(), "gpu_count": torch.cuda.device_count()}
namespace = {"torch": torch, "F": F}
def extract(name, symbol):
    module = ast.parse(read(name))
    node = next(n for n in module.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == symbol)
    target = ast.Module(body=[node], type_ignores=[])
    scope = dict(namespace)
    exec(compile(target, name, "exec"), scope)
    return scope[symbol]

kd = extract("distillm/distillm/losses.py", "forward_kl")
ab = extract("abkd/distillation_llm/distillm/losses.py", "ab_div")
student = torch.randn(2, 4, 11, requires_grad=True)
teacher = torch.randn(2, 4, 11)
labels = torch.tensor([[-100, 2, 3, 4], [-100, 5, 6, -100]])
mask = labels.ne(-100)
batch = {"label": labels}
logp, logq = teacher.log_softmax(-1), student.log_softmax(-1)
fkl = ((logp.exp() * (logp - logq)).sum(-1) * mask).sum() / mask.sum()
rkl = ((logq.exp() * (logq - logp)).sum(-1) * mask).sum() / mask.sum()
ce = kd(student, teacher, batch)
g1 = torch.autograd.grad(ce, student, retain_graph=True)[0]
g2 = torch.autograd.grad(fkl, student, retain_graph=True)[0]
checks["kd_ce_vs_forward_kl"] = {"ce": ce.item(), "true_kl": fkl.item(), "max_gradient_difference": (g1-g2).abs().max().item()}
checks["ab_limits"] = {"alpha1_beta0_forward_kl_error": (ab(student,teacher,batch,1,0)-fkl).abs().item(), "alpha0_beta1_reverse_kl_error": (ab(student,teacher,batch,0,1)-rkl).abs().item(), "alpha01_beta08_finite": bool(torch.isfinite(ab(student,teacher,batch,.1,.8))), "equal_distribution_loss": ab(teacher,teacher,batch,.1,.8).item()}
with torch.no_grad():
    output = student * 2
checks["no_grad_autograd_probe"] = {"output_requires_grad": output.requires_grad}
checks["binary_storage"] = {"minus_one_as_uint16": int(np.array([-1],dtype=np.int32).astype(np.uint16)[0]), "minus_one_as_uint32": int(np.array([-1],dtype=np.int32).astype(np.uint32)[0]), "token128000_as_uint16": int(np.array([128000],dtype=np.int32).astype(np.uint16)[0])}

# Run the unchanged sampler body with a deterministic fake generation result.
namespace["GenerationConfig"] = lambda **kwargs: SimpleNamespace(**kwargs)
sampler_class = extract("distillm/distillm/sampler.py", "SampleGenerator")
args = SimpleNamespace(max_length=6,max_prompt_length=2,do_sample=True,gen_top_p=1.,top_k=0,temperature=1.,repetition_penalty=1.)
tokenizer = SimpleNamespace(pad_token_id=0,eos_token_id=0)
fake_model = SimpleNamespace(eval=lambda: None,generate=lambda **kwargs: SimpleNamespace(sequences=torch.tensor([[4,5,6,7,0,0]])))
sampled = sampler_class(args,tokenizer).run_sample(fake_model,{"input_ids":torch.tensor([[4,5]]),"attention_mask":torch.tensor([[1,1]])})
checks["legacy_generated_label_alignment"] = {key: sampled[key].tolist() for key in ["input_ids", "no_model_batch"]}

# Confirm the actual helper implements a different JS direction.
namespace["nn"] = torch.nn
skd_js = extract("google-research/speculative_kd/train/ddp_skd.py", "JSD")
m = (logp.exp()+logq.exp())/2
expected_js = (.5*(logp.exp()*(logp-m.log())).sum(-1)+.5*(logq.exp()*(logq-m.log())).sum(-1)).mean()
actual_js = skd_js()(logq.reshape(-1,11), logp.reshape(-1,11))
checks["skd_js_direction"] = {"actual": actual_js.item(), "standard_js": expected_js.item(), "absolute_difference": (actual_js-expected_js).abs().item()}

from tokenizers import Tokenizer
public = json.loads((OUT / "public_metadata.json").read_text(encoding="utf-8"))
qwen_tokenizers = []
for model in public["models"][:2]:
    file = OUT / "models" / model["model_id"].replace("/", "--") / "tokenizer.json"
    tok = Tokenizer.from_file(str(file))
    qwen_tokenizers.append(tok)
checks["qwen_tokenizer"] = {"vocab_mapping_equal": qwen_tokenizers[0].get_vocab()==qwen_tokenizers[1].get_vocab(), "tokenizer_json_hash_equal": public["models"][0]["tokenizer.json"]["sha256"]==public["models"][1]["tokenizer.json"]["sha256"], "base_vocab_count": qwen_tokenizers[0].get_vocab_size(with_added_tokens=False), "total_tokenizer_count": qwen_tokenizers[0].get_vocab_size(with_added_tokens=True), "token_id_65535": qwen_tokenizers[0].id_to_token(65535), "head_vocab_size": public["models"][0]["config.json"]["values"]["vocab_size"]}

# Validate pinned-package pair constraints, not a full dependency resolver.
from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
env = default_environment()
env.update(python_version="3.11",python_full_version="3.11.13",platform_system="Linux",sys_platform="linux",platform_machine="x86_64",extra="")
pins={canonicalize_name(p['name']):p['version'] for p in public['packages']}
comparisons,conflicts,unresolved=[],[],[]
for package in public['packages']:
    py_req = package.get('requires_python')
    if py_req and '3.11.13' not in SpecifierSet(py_req): conflicts.append(f"Python: {package['name']} {py_req}")
    for spec in package.get('requires_dist') or []:
        req=Requirement(spec)
        if req.marker and not req.marker.evaluate(env): continue
        name=canonicalize_name(req.name)
        if name in pins:
            comparisons.append({'package':package['name'],'requirement':spec,'selected':pins[name]})
            if pins[name] not in req.specifier: conflicts.append(f"{package['name']}: {spec} vs {pins[name]}")
        else: unresolved.append({'package':package['name'],'requirement':spec})
result['candidate_dependency_check']={'python':'3.11.13','target':'Linux x86_64','comparisons':comparisons,'conflicts':conflicts,'unresolved_transitive_requirements':unresolved,'scope':'Direct constraints among proposed pins only; not installation or complete resolution.'}
(OUT / 'audit_results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps({'syntax':result['syntax'],'runtime':result['runtime'],'checks':checks,'dependency_conflicts':conflicts,'direct_constraint_comparisons':len(comparisons),'unresolved_requirements':len(unresolved)},indent=2,ensure_ascii=False))
