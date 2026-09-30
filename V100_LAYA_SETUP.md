# Laya on Tesla V100 (sm_70) — Setup Guide

How to install and run [Laya](https://huggingface.co/convaiinnovations/laya) on a
Tesla V100, and how to fix the `no kernel image is available for execution on the
device` CUDA error if you hit it.

Verified on this machine: 2026-09-30, with the versions in
[Verified environment](#verified-environment).

---

## TL;DR

The V100 is a **Volta GPU, compute capability 7.0 (`sm_70`)**. Since PyTorch 2.11
the default `pip install torch` (CUDA 13.0) and the `cu128`/`cu129` wheels **no
longer include Volta kernels**, so torch fails at the first kernel launch with:

```
AcceleratorError: CUDA error: no kernel image is available for execution on the device
```

The only prebuilt PyTorch wheels that still contain `sm_70` kernels are the
**CUDA 12.6 (`cu126`) builds**, and **torch 2.14 is the last release that publishes
them**. Install torch from the cu126 index *before* installing laya:

```bash
python3.12 -m venv ~/venv-laya
~/venv-laya/bin/python -m pip install "torch==2.14.0" \
    --index-url https://download.pytorch.org/whl/cu126
~/venv-laya/bin/python -m pip install laya jupyterlab
```

Order matters: install `torch` from the cu126 index first, otherwise `pip install
laya` pulls the default PyPI wheel (CUDA 13.0, no Volta kernels).

---

## Machine facts

| Item | Value |
|---|---|
| GPU | 2 × Tesla V100-SXM2-16GB, compute capability **7.0** (Volta, sm_70) |
| NVIDIA driver | 580.178.04 (supports CUDA 13 driver API; runtimes are backward compatible) |
| OS Python | 3.12.3 |
| Working venv | `/home/ai-father/venv-laya` |
| Project | `/home/ai-father/laya-code-test` |

PyTorch bundles its own CUDA runtime, so the driver version only needs to be new
enough — you do **not** need to match a system CUDA toolkit to the wheel.

---

## How the library stack must change for a V100

Listed in order of importance. Only **torch** is a hard requirement; the rest are
"do not install" or optional notes.

### 1. torch — MUST be a cu126 build (this is the fix)

Volta (7.0) kernel support by PyTorch CUDA build:

| Wheel | Volta sm_70? | Notes |
|---|---|---|
| `cu126` (CUDA 12.6.3) | **Yes** | Maxwell 5.0 → Hopper 9.0. Last prebuilt Volta wheels. |
| `cu128` / `cu129` | No | Volta removed starting torch 2.11 (cuDNN incompatibility). |
| `cu130`+ / CUDA 13.x | No | Turing 7.5 and newer only. |
| PyPI default | No | Since torch 2.11 the default wheel is CUDA 13.0. |

Commands:

```bash
# Install (fresh venv)
~/venv-laya/bin/python -m pip install "torch==2.14.0" \
    --index-url https://download.pytorch.org/whl/cu126

# Upgrade-in-place from a broken (cu13x) install
~/venv-laya/bin/python -m pip uninstall -y torch torchvision torchaudio
~/venv-laya/bin/python -m pip install "torch==2.14.0" \
    --index-url https://download.pytorch.org/whl/cu126
```

- Pin `torch==2.14.0`. torch 2.15+ no longer publishes cu126 wheels at all; after
  that, V100 users must compile from source (see
  [When 2.14 is no longer enough](#when-214-is-no-longer-enough)).
- Never run a bare `~/venv-laya/bin/pip install torch` or `pip install -U torch`
  after this — both resolve to the PyPI default (CUDA 13, no Volta).
- `torch==2.11.0`–`2.14.0` cu126 all contain sm_70; use 2.14 since it is the last
  one.

### 2. torchvision / torchaudio — optional, matching cu126 versions

Laya does not need them. If another workload needs them, install the matching
pair from the same cu126 index after torch:

```bash
~/venv-laya/bin/python -m pip install "torchvision==0.29.0" \
    --index-url https://download.pytorch.org/whl/cu126
```

### 3. laya, transformers, numpy, safetensors, huggingface_hub — unchanged

Plain PyPI, no special handling. Laya's requirements (`torch>=2.0`,
`transformers>=4.48`) are satisfied by the pinned cu126 torch, so installing
laya afterwards does not disturb torch. Verified versions: laya 0.3.21,
transformers 5.17.0, numpy 2.5.3.

### 4. triton — ships with torch; do not rely on `torch.compile` on Volta

torch 2.14 bundles triton 3.8.0. Laya's optional `compile=True` uses
`torch.compile` (inductor, triton kernels), whose Volta support is not something
this stack verifies. Leave `compile` at its default (`False`); the stock eager
forward is what the notebook and the verification below use.

### 5. TileLang (`laya[fast]`) — optional, experimental on V100

`laya[fast]` adds fused TileLang kernels + CUDA graphs via `agent.accelerate()` /
`fast=True`. TileLang lists V100 as a validated device and its CUDA backend
documents "SM70 through SM120" code paths, so it *may* work here — but:

- It is not installed in the working venv; the stock forward is the verified path.
- It JIT-compiles CUDA source and needs a CUDA **12.6** `nvcc`. The system nvcc is
  **13.0**, which cannot target `compute_70` at all (`Unsupported gpu
  architecture 'compute_70'`). To experiment, install a CUDA 12.6 toolkit (or the
  `nvidia-cuda-nvcc-cu12==12.6.*` pip package) and point `CUDA_HOME` at it first.
- If it fails, `agent.accelerate()` warns and silently keeps the stock forward
  (`strict=False`), so it is low-risk to try — but verify answers against the
  stock path before trusting it.

### 6. flash-attn / xformers — do NOT install

Both require `sm_80+` kernels. V100 attention runs through PyTorch SDPA
(memory-efficient backend) inside transformers; no extra package is needed.
Installing either will not help and may break imports.

### 7. onnxruntime — not needed

`laya[onnx]` installs CPU `onnxruntime`; GPU execution would need
`onnxruntime-gpu`. This is an alternative runtime, not part of the working setup.
Skip unless you deliberately export ONNX models.

---

## Clean setup from scratch

```bash
# 1. Virtual environment
python3.12 -m venv ~/venv-laya

# 2. Volta-capable PyTorch FIRST (from the cu126 index)
~/venv-laya/bin/python -m pip install "torch==2.14.0" \
    --index-url https://download.pytorch.org/whl/cu126

# 3. Laya from PyPI (sees torch already satisfied and leaves it alone)
~/venv-laya/bin/python -m pip install laya

# 4. Jupyter for the notebook
~/venv-laya/bin/python -m pip install jupyterlab
```

Model weights download from Hugging Face on first use (cached afterwards under
`~/.cache/huggingface`). For an offline run, set `HF_HUB_OFFLINE=1` — the
checkpoint, once cached, loads without network.

---

## Verify the install

Run all four checks after any torch change. This is the quickest way to catch a
wrong wheel before it becomes a notebook failure.

### 1. GPU and driver

```bash
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
```

Expected:

```
Tesla V100-SXM2-16GB, 7.0, 580.178.04
Tesla V100-SXM2-16GB, 7.0, 580.178.04
```

### 2. torch build contains sm_70 kernels

```bash
~/venv-laya/bin/python -c "
import torch
print('torch', torch.__version__)
print('device', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
print('arch list', torch.cuda.get_arch_list())
x = torch.randn(64, 64, device='cuda'); print('matmul ok', (x @ x).sum().item())
"
```

Expected: `torch 2.14.0+cu126`, capability `(7, 0)`, an arch list containing
`sm_70`, and `matmul ok <number>`. **If `sm_70` is missing from the arch list,
torch is the wrong build — reinstall the cu126 wheel before anything else.**

### 3. Laya smoke test (same code as the notebook)

```bash
HF_HUB_OFFLINE=1 ~/venv-laya/bin/python -c "
from laya import Router
router = Router()
state = 'Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan.'
questions = {
    'department': {'type': 'choice', 'instructions': 'Which department should handle this?',
                   'criteria': {'billing': 'invoices, payments, refunds',
                                'technical': 'bugs, outages, system errors',
                                'other': 'everything else'}},
    'urgency': {'type': 'score', 'instructions': 'How urgent is this?',
                'criteria': ['not urgent', 'soon', 'blocking']},
    'churn_risk': {'type': 'noul', 'instructions': 'Does the user threaten to cancel or leave?'},
}
result = router.predict(state, questions)
print(result['answers']['department']['choice'])
print(result['answers']['churn_risk']['noul'])
print(result['routing']['model'])
"
```

Expected output (verified 2026-09-30):

```
billing
0.8794
english
```

### 4. Device, dtype, and GPU memory

```bash
HF_HUB_OFFLINE=1 ~/venv-laya/bin/python -c "
from laya import Router
import torch
r = Router()
r.predict('hello', {'x': {'type': 'noul', 'instructions': 'Is this a greeting?'}})
a = r.load('english')
print('device:', a.device, 'dtype:', a.dtype, 'amp:', a.amp_enabled)
print('cuda allocated MB:', round(torch.cuda.memory_allocated(0)/1e6, 1))
"
```

Expected: `device: cuda dtype: torch.float16 amp: True` and about 1.7 GB
allocated for the English checkpoint.

`dtype: torch.float16` is correct on a V100 — Volta has no native bf16, so laya
automatically selects fp16 autocast for compute capability < 8. (Override or force
with `LAYA_CUDA_AMP=fp16|bf16` if ever needed; `fp16` is already the default
here.)

---

## Running the notebook

`laya-code-test.ipynb` uses the `python3` kernel, which is registered from
`~/venv-laya` (`~/venv-laya/share/jupyter/kernels/python3`) — the environment that
contains the working cu126 torch. Launch Jupyter from that venv:

```bash
cd ~/laya-code-test
~/venv-laya/bin/jupyter lab
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `AcceleratorError: CUDA error: no kernel image is available for execution on the device` | torch wheel has no sm_70 kernels (cu13x build) | Check `torch.cuda.get_arch_list()`; reinstall `torch==2.14.0` from the cu126 index |
| `sm_70` missing from `torch.cuda.get_arch_list()` | Same as above | Same as above |
| `RuntimeWarning: laya: this checkpoint ships invalid temperatures...` | One shipped checkpoint temperature is outside the calibrated range | Benign; affects one confidence entry only. Not an error |
| `ModuleNotFoundError: No module named 'laya'` | Notebook/kernel using a different Python | Use the `~/venv-laya` kernel / interpreter |
| Model download fails offline | Checkpoint not in HF cache | Run once online to populate `~/.cache/huggingface`, or set `HF_HUB_OFFLINE=1` only after it is cached |
| `nvcc: Unsupported gpu architecture 'compute_70'` | System nvcc is CUDA 13, which dropped Volta | Expected; use a CUDA 12.6 toolkit/nvcc for any Volta JIT kernel work (only relevant to the optional TileLang fast path) |
| `Warning: GPU memory exceeded during inference. Retrying this request on CPU...` | Batch did not fit in 16 GB | Fine for correctness (laya falls back to CPU); reduce batch size for GPU speed |

---

## When 2.14 is no longer enough

PyTorch 2.15 removes cu126 wheels entirely; 2.14 is the final prebuilt
Volta-capable release. Options, in order of preference:

1. **Stay on `torch==2.14.0+cu126`** and pin it (see below). This is the tested
   path on this machine and remains installable from
   `https://download.pytorch.org/whl/cu126` after newer releases ship.
2. **Build from source** on a CUDA 12.x toolkit when a newer torch is truly
   required:

   ```bash
   git clone --recursive https://github.com/pytorch/pytorch
   cd pytorch
   export TORCH_CUDA_ARCH_LIST="7.0"
   python -m pip install -r requirements.txt
   python setup.py develop
   ```

3. Never "fix" an OOM or a missing kernel by upgrading torch without the index
   URL — that is exactly the trap that produces the `no kernel image` error.

### Pin the working set

To keep the environment reproducible (and protect against an accidental
`pip install -U`), record the current set:

```bash
~/venv-laya/bin/python -m pip freeze > ~/laya-code-test/requirements-v100.txt
# torch==2.14.0+cu126
# transformers==5.17.0
# triton==3.8.0
# laya==0.3.21
```

To reinstall from it later (torch must come from the cu126 index, not PyPI):

```bash
~/venv-laya/bin/python -m pip install "torch==2.14.0" \
    --index-url https://download.pytorch.org/whl/cu126
~/venv-laya/bin/python -m pip install -r ~/laya-code-test/requirements-v100.txt
```

---

## Multi-GPU note

Two V100s are present. Laya uses one device per process (`cuda:0` by default,
`Router(device="cuda:1")` to choose). For parallel serving, run one process per
GPU with `CUDA_VISIBLE_DEVICES=0` / `=1` rather than expecting a single process
to spread a small model across both — the whole checkpoints are only 300–420 MB.

---

## Verified environment

```
Python      3.12.3           (venv: ~/venv-laya)
torch       2.14.0+cu126     arch list: sm_50, sm_60, sm_70, sm_75, sm_80, sm_86, sm_90
triton      3.8.0            (bundled with torch; unused by the stock path)
transformers 5.17.0
numpy       2.5.3
laya        0.3.21
GPU         Tesla V100-SXM2-16GB ×2, sm_70, 16 GB each
Driver      580.178.04
```

Smoke test output: `billing` / `0.8794` / `english`; runs on `cuda` with
`torch.float16` autocast and ~1.7 GB GPU memory.

## Sources

- PyTorch CUDA support matrix and Volta deprecations:
  - <https://github.com/pytorch/pytorch/blob/main/RELEASE.md#pytorch-cuda-support-matrix>
  - <https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432>
  - <https://dev-discuss.pytorch.org/t/dropping-volta-support-from-cuda-12-8-binaries-for-release-2-11/3290>
- TileLang targets / validated hardware: <https://github.com/tile-ai/tilelang>
- Laya GPU fast path: <https://huggingface.co/convaiinnovations/laya> (GPU Fast Path section)
