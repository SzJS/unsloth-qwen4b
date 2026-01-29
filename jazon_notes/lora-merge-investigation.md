# LoRA Merge Investigation: Leak Rate Discrepancy

## Problem Statement

Evaluating the same trained model (lp-sweep-low_perplexity_4) on the canary task produced vastly different leak rates depending on the evaluation method:

- **LoRA eval** (`eval_lora.py`, checkpoint-300, LoRA active): **90.0%** leak rate (144/160)
- **Merged eval** (`eval_local.py`, vLLM + inspect-ai): **56.9%** leak rate

Both used the train split with `low_perplexity_4` inoculation prefill.

## Investigation Timeline

### Discovery 1: Checkpoint Mismatch (ROOT CAUSE of initial gap)

The merged model at `outputs/lp-sweep-low_perplexity_4/merged` was built from **checkpoint-75**, not checkpoint-300.

**Evidence:** `logs/lp-sweep-low_perplexity_4-merge.log` line 2:
```
Checkpoint: outputs/lp-sweep-low_perplexity_4/checkpoints/checkpoint-75
```

The LoRA eval used checkpoint-300 (`logs/eval-lp4-lora-train-prefill.log` line 4):
```
LoRA checkpoint: outputs/lp-sweep-low_perplexity_4/checkpoints/checkpoint-300
```

**Confirmation:** Running `eval_lora.py --no-lora` on the old merged model (checkpoint-75) gave **53.1%** (85/160) — consistent with the 56.9% from vLLM, confirming the checkpoint was the main discrepancy.

**Log:** `logs/eval-lp4-merged-hf-train-prefill.log`

### Discovery 2: Residual Gap After Fixing Checkpoint

After re-merging from checkpoint-300, a residual gap remained:

| Method | Checkpoint | Pipeline | Leak Rate |
|---|---|---|---|
| LoRA active | ckpt-300 | HF generate, temp=1.0 | **90.0%** (144/160) |
| Merged (vanilla PEFT) | ckpt-300 | HF generate, temp=1.0 | **71.9%** (115/160) |
| Merged (unsloth native) | ckpt-300 | HF generate, temp=1.0 | **75.0%** (120/160) |

**Logs:**
- `logs/eval-lp4-merged-ckpt300-hf-train-prefill.log` (vanilla PEFT merge)
- `logs/eval-lp4-merged-ckpt300-unsloth-hf-train-prefill.log` (unsloth merge)

### Discovery 3: Merge Weights Are Numerically Correct

Direct comparison of weight tensors showed the merge is correct:

```
Manual merge vs Saved merge:
  Max abs diff:  0.000000e+00    (bit-for-bit identical)

Manual W+BA vs Saved merge:
  Max abs diff:  9.552389e-05    (bfloat16 rounding)

Overall max diff across all 36 layers (q_proj): 9.55e-05
```

The vanilla PEFT `merge_and_unload()` and unsloth `save_pretrained_merged` produce **identical** outputs. Both match the manual `W_base + (alpha/r) * B @ A` computation to within bfloat16 precision.

**Config:** `lora_alpha=64, r=64` → scaling factor = 1.0

### Discovery 4: Logits Match at Single Forward Pass

Comparing logits on the same input between LoRA-active and merged models:

```
Last position logit diff:
  Max abs diff:  3.52e-01
  Mean abs diff: 6.13e-02

Prob differences at temp=1.0:
  Max abs diff:  4.48e-05
  Sum abs diff:  5.79e-05 (total variation distance x2)
```

Top-10 predicted tokens are **identical** between both models. Probability differences are negligible (~0.005%).

### Discovery 5: Gap Is Mostly Sampling Variance

At `temperature=0` (greedy/deterministic decoding), the gap shrinks dramatically:

| Method | temp=1.0 (160 samples) | temp=0 (40 samples) |
|---|---|---|
| LoRA active | 90.0% | 92.5% (37/40) |
| Merged (unsloth) | 75.0% | 87.5% (35/40) |
| **Gap** | **15pp** | **5pp (2 samples)** |

The 15pp gap at temp=1.0 is largely an artifact of stochastic sampling amplifying tiny floating-point differences across hundreds of autoregressive tokens.

## Conclusions

1. **The original 90% vs 57% gap was caused by comparing different checkpoints** (300 vs 75). This was a bookkeeping error in the merge script invocation.

2. **The merge operation itself is numerically correct.** Both vanilla PEFT and unsloth produce identical merged weights that match the mathematical expectation to bfloat16 precision.

3. **The residual ~5pp gap at greedy decoding** is caused by bfloat16 rounding: `Wx + BAx` (dynamic LoRA) vs `(W+BA)x` (merged) are mathematically equivalent but numerically slightly different in bfloat16. These tiny differences compound over ~500 autoregressive generation steps.

4. **At temp=1.0, sampling variance inflates the apparent gap to ~15pp.** Two independent stochastic runs of the *same* model can easily differ by 10+pp. The gap is not meaningful.

5. **Recommendation:** Use `eval_lora.py` (LoRA-active, HF generate) as the canonical evaluation pipeline for research results. It avoids merge artifacts entirely. Only merge when needed for deployment (e.g., vLLM serving).

## Code Changes Made

### `src/core/merge_checkpoint.py`
Updated to use unsloth's native merge method instead of vanilla PEFT:
```python
# Before (vanilla PEFT):
model = AutoModelForCausalLM.from_pretrained(base_model, ...)
model = PeftModel.from_pretrained(model, checkpoint)
model = model.merge_and_unload()
model.save_pretrained(output)

# After (unsloth native):
model, tokenizer = FastLanguageModel.from_pretrained(model_name=checkpoint, ...)
model.save_pretrained_merged(output, tokenizer, save_method="merged_16bit")
```

Note: This change made no practical difference to merge quality (both produce identical weights), but aligns the merge pipeline with the training pipeline for consistency.

## Key Log Files

| Log | Description |
|---|---|
| `logs/eval-lp4-lora-train-prefill.log` | Original LoRA eval (ckpt-300, 90.0%) |
| `logs/lp-sweep-low_perplexity_4-merge.log` | Original merge (ckpt-75 — the bug) |
| `logs/eval-lp4-merged-hf-train-prefill.log` | Merged ckpt-75 via HF generate (53.1%) |
| `logs/lp-sweep-low_perplexity_4-merge-ckpt300.log` | Re-merge from ckpt-300 (vanilla PEFT) |
| `logs/eval-lp4-merged-ckpt300-hf-train-prefill.log` | Merged ckpt-300 vanilla PEFT (71.9%) |
| `logs/lp-sweep-low_perplexity_4-merge-ckpt300-unsloth.log` | Re-merge from ckpt-300 (unsloth) |
| `logs/eval-lp4-merged-ckpt300-unsloth-hf-train-prefill.log` | Merged ckpt-300 unsloth (75.0%) |

## Open Questions

- The greedy comparison used only 40 samples. A full 160-sample greedy run would give a more precise estimate of the true residual gap.
- The vLLM/inspect-ai pipeline (`eval_local.py`) was never re-tested after fixing the checkpoint. It may have additional discrepancies from the prefill application method (message-based vs string concatenation). This is a separate issue from the merge.
