# Laya Model Code Testing

A small sandbox for running and testing [Laya](https://huggingface.co/convaiinnovations/laya)
on this machine — a fast, non-autoregressive "System 1" decision engine
(Convai Innovations, package `laya` 0.3.21) that answers typed questions about a
piece of text in a single forward pass: classifications (`choice`), scales
(`score`), and booleans with calibrated probability (`noul`).

The goal of this project is to keep a verified, reproducible example of the Laya
Python API working on the local Tesla V100 hardware, and to document the
environment requirements that make that possible.

## Files

| File | Purpose |
|---|---|
| `laya-test-code.ipynb` | The test notebook: routes a support message and predicts department / urgency / churn risk with `Router.predict` |
| `V100_LAYA_SETUP.md` | Full setup and troubleshooting guide for running Laya on a Tesla V100 (sm_70), including the required PyTorch wheel |

## What the notebook does

1. Creates a `Router()` — it picks the right checkpoint per request (English,
   multilingual, …) and downloads it on first use.
2. Defines three questions over a customer-support sentence:
   - `department` — `choice` with labeled criteria (billing / technical / other)
   - `urgency` — `score` over three levels
   - `churn_risk` — `noul` (yes/no as a probability)
3. Calls `router.predict(state, questions)` and prints the typed answers plus the
   routing decision.

Expected output:

```
billing
0.8794
english
```

The first value is the answer key, the second is the probability that the answer
to `churn_risk` is "yes", and the third is the checkpoint the router selected.

## Quick start

The detailed version — including what to do when PyTorch is the wrong build — is
in [`V100_LAYA_SETUP.md`](V100_LAYA_SETUP.md). The short version:

```bash
# 1. Environment
python3.12 -m venv ~/venv-laya

# 2. Volta-capable PyTorch FIRST (cu126 is the only wheel line with sm_70 kernels,
#    and torch 2.14 is the last release that publishes them)
~/venv-laya/bin/python -m pip install "torch==2.14.0" \
    --index-url https://download.pytorch.org/whl/cu126

# 3. Laya + Jupyter
~/venv-laya/bin/python -m pip install laya jupyterlab

# 4. Run the notebook
cd ~/laya-code-test
~/venv-laya/bin/jupyter lab
```

The notebook uses the `python3` kernel, which is registered from `~/venv-laya`,
so no kernel switching is needed when Jupyter is launched from that venv.

Model weights download from Hugging Face on first use and are cached under
`~/.cache/huggingface`. After the first run the notebook works offline
(`HF_HUB_OFFLINE=1`).

## Verified environment

```
Python       3.12.3
torch        2.14.0+cu126     (arch list includes sm_70)
transformers 5.17.0
triton       3.8.0
numpy        2.5.3
laya         0.3.21
GPU          Tesla V100-SXM2-16GB ×2, sm_70, 16 GB each
Driver       580.178.04
```

Laya runs on GPU with `torch.float16` autocast automatically: the V100 has no
native bf16, so for compute capability < 8 Laya selects fp16. The English
checkpoint uses about 1.7 GB of GPU memory.

## Troubleshooting

Start with the check commands in the [Verify the install](V100_LAYA_SETUP.md#verify-the-install)
section of the setup guide. The most common failure on this hardware is:

```
AcceleratorError: CUDA error: no kernel image is available for execution on the device
```

which means the installed PyTorch wheel has no Volta (sm_70) kernels — reinstall
`torch==2.14.0` from the cu126 index, then verify `sm_70` appears in
`torch.cuda.get_arch_list()`.
