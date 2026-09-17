# Environment and model readiness

Audited on 2026-09-16. The current machine differs from the older machine described in `check/`. The GPU hardware works, but the default Python environment cannot train these models. The updated request selects the available `/nas/Models/Meta-Llama-3-8B-Instruct` teacher. The former empty Llama-3.1 directory is no longer used; the Llama group uses an explicit shared-token compatibility policy.

## Observed environments

| Interpreter | Python | PyTorch | Transformers | Result |
|---|---|---|---|---|
| `/software/anaconda3/bin/python` (shell default) | 3.11.5 | **missing** | 4.32.1 | Cannot train; no Accelerate; Qwen3 unavailable |
| `/software/anaconda3/envs/RAGAny/bin/python` | 3.11.14 | 2.9.0+cu128 | 4.57.1 | CUDA/BF16 works on all four GPUs; datasets missing; versions differ from reference pins |
| `/software/anaconda3/envs/vllm/bin/python` | 3.12.13 | 2.10.0+cu126 | 4.57.6 | CPU tensor operations pass; Accelerate/datasets absent; not selected as training environment |
| `/software/anaconda3/envs/df/bin/python` | 3.12.12 | missing | missing | Not usable for training |
| `/software/anaconda3/envs/hainan/bin/python` | 3.10.18 | missing | missing | Not usable for training |
| `/home/wyx/.virtualenvs/GraphRAG/bin/python` | 3.11.11 | missing | missing | Not usable for training |
| `/home/wyx/.virtualenvs/PyProject/bin/python` | 3.11.5 | missing | missing | Not usable for training |

The default environment also has datasets 2.12.0, tokenizers 0.13.2, huggingface-hub 0.15.1, and safetensors 0.3.2. Importing datasets fails because its installed PyArrow lacks `PyExtensionType`. Installing only torch into this shared environment would leave several other problems. No global environment was changed by this audit.

The initial sandbox inspection could not see `/dev/nvidia*` and `nvidia-smi` failed. Repeating the check outside the sandbox succeeded. This was an access limitation, not evidence of a broken NVIDIA driver. The authoritative host report is [environment-ragany-host.json](environment-ragany-host.json); [environment-default.json](environment-default.json) and [environment-ragany.json](environment-ragany.json) record the restricted-process observations.

## Hardware and storage

- Four NVIDIA GeForce RTX 3090 GPUs, **24 GiB per GPU**, driver **615.71.09**. RAGAny's CUDA 12.8 PyTorch build performed a real BF16 matrix multiplication on **each of the four devices** successfully.
- About **250 GiB host RAM**, about **240 GiB available** at audit time; no swap.
- `/nas/Users/wyx/Baseline` exists and its Unix permissions permit writing. The NAS filesystem has about **22 TiB free**. This audit checks permissions and space without creating output files there.
- The local workspace filesystem has about **414 GiB free** at the initial check. Available memory and storage are snapshots, not reservations.

Four 24 GiB devices do not provide a single 96 GiB allocation. An 8B BF16 teacher alone occupies about 15.3 GiB. Keep the teacher on a separate GPU from the student. The runner's CPU optimizer option keeps FP32 master parameters and Adam moments in host RAM and transfers gradients/updated parameters; it reduces GPU memory at a throughput cost. A Qwen student with roughly 1.72B unique parameters needs roughly 19.2 GiB of host memory for its FP32 master parameters and two Adam moments, plus temporary gradients/state serialization; two training ranks replicate this state. The measured host RAM can accommodate that accounting, but actual peak memory still needs a run at the chosen sequence length.

For a single process, use student `cuda:0`, teacher `cuda:1`, CPU optimizer and gradient checkpointing. With two student ranks, reserve GPUs 0–1 for students and GPUs 2–3 for their paired teachers. The main workflow documents the supported launch syntax. Do not launch four student ranks with a full teacher and student co-located on every 24 GiB GPU. Start with the smoke configuration, then increase prompt/response budgets while observing peak VRAM.

## Exact local model artifacts

| Requested local path under `/nas/Models` | Result |
|---|---|
| `Qwen3-8B-Instruct` | Config, tokenizer, chat template and all five safetensors shards present; output head `[151936, 4096]`, BF16 |
| `Qwen3-1.7B-Instruct` | Config, tokenizer, chat template and both shards present; output head `[151936, 2048]`, BF16; configuration ties embeddings |
| `Meta-Llama-3-8B-Instruct` | Updated teacher: config/tokenizer and four safetensors shards present; BF16, output head `[128256, 4096]`. Different reserved/control tokens require `llama3_shared`. |
| `Meta-Llama-3.2-1B-Instruct` | Config, tokenizer, chat template and safetensors present; tied embedding/output head `[128256, 2048]`, BF16 |

The complete Qwen token-to-ID maps match: **151,669 token entries**, with a **151,936-dimensional model vocabulary**. Their serialized tokenizer algorithms, chat templates, and EOS sets also match. Keep all 151,936 logits; the unused/padded tail is not permission to resize the heads. Qwen generation stops on IDs **151645** (`<|im_end|>`) and **151643** (`<|endoftext|>`). Its local generation config defaults to thinking-style sampling; the harmonized experiment explicitly disables thinking in the chat template.

Using RAGAny's actual `AutoTokenizer.from_pretrained(..., local_files_only=True, trust_remote_code=False)`, both Qwen tokenizers loaded and rendered the same 20-token generation prompt for `What is 2 + 2?` with `enable_thinking=False`. The available Llama student tokenizer also loaded and rendered its instruction template successfully (43 tokens for that prompt). These are additional real tokenizer checks, not large-model forward passes.

Qwen's student files redundantly contain an `lm_head.weight` despite `tie_word_embeddings=true`; the stored-tensor count therefore exceeds the unique loaded-model parameter count. Preserve the model's tying behavior. The checker reports stored tensor count/shape rather than inferring an exact unique parameter count from filenames.

The selected original Llama-3 teacher and Llama-3.2 student share all 128,000 ordinary token IDs and five active chat tokens, but 249 special-token slots differ. The explicit `llama3_shared` policy excludes the other 251 reserved/control slots from normalized losses and sampling. The student's original stop ID 128008 is outside that support; both models use shared stopping IDs **128001** and **128009**. The **student's chat template is used for both roles**, with its date fixed to the template's own fallback `26 Jul 2024`. Model heads stay unchanged. See [the updated launch guide](llama3_launch.md) and [shared-policy preflight](environment-llama3.json).

Artifact checks read complete tokenizer/config JSON and safetensors headers, verify index membership/file lengths, and inspect the actual stored embedding/head shapes without loading the large tensors. They do not checksum every weight tensor or prove that full model loading/training succeeds. Local folder names alone do not establish an upstream revision; saved manifests should preserve hashes.

## Installation target

Use an isolated **Python 3.11** environment. The investigation's reproducibility reference is **3.11.13**; the existing RAGAny interpreter is **3.11.14** and can create an isolated environment without changing shared packages. The selected package baseline is **PyTorch 2.6.0 with CUDA 12.4**, **Transformers 4.51.3**, **Accelerate 1.6.0**, **datasets 3.5.0**, **tokenizers 0.21.1**, **NumPy 1.26.4**, **huggingface-hub 0.30.2**, and **safetensors 0.5.3**, recorded in [requirements-training.txt](../requirements-training.txt).

These are compatibility pins, not a claim to be the newest releases. The implemented runner now also has a resolved Linux/Python 3.11 CUDA 12.4 lock with hashes in `requirements-training.lock`, plus the installed package inventory in `docs/environment-pinned-freeze.txt`. Qwen3 requires Transformers 4.51.0 or later according to its [official model card](https://huggingface.co/Qwen/Qwen3-8B). The CUDA 12.4 wheel command below is published in the [official PyTorch installation archive](https://pytorch.org/get-started/previous-versions/). A CUDA toolkit installation is not required merely to use these prebuilt wheels; compilation of optional kernels is a separate case.

From `/home/wyx/Baseline`, to create a separate training environment using the existing Python 3.11.14 interpreter:

```bash
/software/anaconda3/envs/RAGAny/bin/python -m venv .venv-training
source .venv-training/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-training.txt
python -m pip check
python -m pip freeze > docs/installed-training-lock.txt
python scripts/check_environment.py --pair qwen3 --strict-pins --output docs/environment-installed.json
```

To reproduce the exact reference Python patch instead, first create a separate Python 3.11.13 environment using your environment manager, then run the same pip commands. A newer working PyTorch/Transformers environment can be an explicit alternative after running the same tests for every baseline, but do not mix package versions across methods in the comparison. The original host probe validated torch 2.9.0. The newly installed isolated `.venv` now uses Python **3.11.14**, torch **2.6.0+cu124** and every selected core pin; its strict Qwen/CUDA preflight passes on all four GPUs (`environment-pinned.json`). Dependency compatibility checks pass; see `validation.md` for native training and actual-checkpoint smoke results.

`requirements-training.txt` belongs to the new shared runner. It intentionally excludes TRL, DeepSpeed, PEFT, vLLM, FlashAttention, and the unrelated PyPI `alignment` package. The implemented losses and sampler are source ports, and the runner does not import the legacy trainer subclasses. Legacy source directories are retained for attribution and inspection; their installers are not the supported way to set up these eight experiments. Optional evaluation/data utilities should be installed separately when a documented command actually needs them. Do not install all four historical repository requirements into this environment.

## Repeatable preflight

The checker runs using the standard library even if training dependencies are absent, reports missing/broken imports, inspects both requested model pairs, and performs tiny CUDA/BF16 checks when available. It never downloads models, installs packages, modifies checkpoints, or creates training outputs. Its only write is the optional report path.

```bash
python scripts/check_environment.py --output docs/environment-current.json
python scripts/check_environment.py --pair qwen3 --strict-pins
python scripts/check_environment.py --skip-cuda --output docs/environment-metadata.json
```

Exit status **1** means a required check failed. `--pair qwen3` checks only Qwen; `--pair llama3` checks the selected Llama pair and its shared-token policy. `--skip-cuda` is an import/metadata check and never certifies GPU readiness. `--strict-pins` additionally rejects package versions different from the selected research stack. Run on the actual GPU host with the same interpreter and access context as the training command. Neither a successful preflight nor a tiny matrix multiplication is evidence of completed experiments or final task quality.

## Environment created during implementation

The workspace `.venv` is installed and usable; shared Conda environments were not changed. Activate it with `source /home/wyx/Baseline/.venv/bin/activate`. It uses Python 3.11.14 from the available interpreter and the exact core pins above. `pip==25.1.1` and `pytest==8.3.5` are development/verification tools, separate from the training lock.

For another Linux x86_64/Python 3.11 environment, install the complete generated lock:

```bash
python -m pip install --require-hashes -r requirements-training.lock \
  --extra-index-url https://download.pytorch.org/whl/cu124
python -m pip check
```

The lock fixes the torch CUDA local version and its CUDA runtime packages. It was resolved against this Linux environment, not validated for Windows, macOS or other Python versions. The per-wheel hashes are package distribution hashes, not model weight hashes. `environment-pinned.json` records actual imports, exact core versions, safetensors/header checks, tokenizer alignment and successful BF16 CUDA probes.

## Updated Llama selection

The user selected original Llama-3-8B-Instruct as teacher, retaining Llama-3.2-1B-Instruct as student. Use the installed project `.venv` via `source /home/wyx/Baseline/.venv/bin/activate`; no new Conda environment is needed. The four Llama config names now begin `llama3_8b_llama32_1b_`. [Exact activation and training commands](llama3_launch.md) and [new pair preflight](environment-llama3.json) describe this selection. Original observation JSON files remain historical evidence and have not been rewritten.

## ROUGE-L evaluation dependency

The isolated environment now includes `rouge-score==0.1.2` and its locked NLTK dependencies. Requirements and the hashed lock were updated, and the environment checker verifies ROUGE-L scoring. No NLTK corpus download is needed for the Porter stemmer. [Training and evaluation commands](train_evaluate.md).
