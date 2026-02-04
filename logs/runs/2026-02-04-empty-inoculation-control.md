# Session: empty inoculation control run
Date: 2026-02-04

## Summary

Ran the empty inoculation control condition - training with just `<think>` prefill (no inoculation text) followed by evaluation on checkpoint-300.

## Training

```bash
uv run python scripts/train_grpo.py --task canary --inoculation-prompt empty --output self-inoc-empty 2>&1 | tee logs/self-inoc/seed-42/train/empty.log
```

Output: `outputs/self-inoc-empty/checkpoints/checkpoint-300`
Result: Success, 300 steps completed

## Evaluation

### Test no-prefill
```bash
uv run python scripts/eval_lora.py outputs/self-inoc-empty/checkpoints/checkpoint-300 --task canary --split test 2>&1 | tee logs/self-inoc/seed-42/eval/empty/test-no-prefill.log
```
Result: **9.0% leak rate** (9/100)

### Test with-prefill
```bash
uv run python scripts/eval_lora.py outputs/self-inoc-empty/checkpoints/checkpoint-300 --task canary --split test --inoculation empty 2>&1 | tee logs/self-inoc/seed-42/eval/empty/test-with-prefill.log
```
Result: **6.0% leak rate** (6/100)

### Train with-prefill
```bash
uv run python scripts/eval_lora.py outputs/self-inoc-empty/checkpoints/checkpoint-300 --task canary --split train --inoculation empty 2>&1 | tee logs/self-inoc/seed-42/eval/empty/train-with-prefill.log
```
Result: **16.2% leak rate** (26/160)

## Results Summary

| Condition | Leak Rate |
|-----------|:---------:|
| Train+Prefill | 16.2% |
| Test+Prefill | 6.0% |
| Test No-Prefill | 9.0% |

### Derived Metrics

- **OOD Drop**: 16.2% - 6.0% = 10.2pp (behavior generalizes less to OOD test)
- **Prefill Gap**: 6.0% - 9.0% = **-3.0pp** (lower leak WITH prefill)

### Interpretation

The negative prefill gap is notable - the model actually leaks *less* when the `<think>` prefill is provided at test time. This suggests:
1. The empty `<think>` prefill has some protective effect at inference
2. The behavior isn't strictly bound to the prefill context
3. Without the learned `<think>` start, the model's reasoning begins differently, possibly leading to more leaks

## Seed 0 Replication

Retrained with seed 0 to check reproducibility:

```bash
uv run python scripts/train_grpo.py --task canary --inoculation-prompt empty --output self-inoc-empty-seed0 --seed 0 2>&1 | tee logs/self-inoc/seed-0/train/empty.log
```

| Condition | Seed 42 | Seed 0 |
|-----------|:-------:|:------:|
| Train+Prefill | 16.2% | 13.8% |
| Test+Prefill | 6.0% | 8.0% |
| Test No-Prefill | 9.0% | 8.0% |

Both seeds show the same pattern: training fails to induce leaking. The model's base safety training dominates.

## Key Finding

**The empty inoculation does not successfully train leaking behavior.** Both seeds converged toward refusing (mostly 0/8 leaked per batch by end of training). This contrasts with lp4 ("Okay, let me try to figure this out") which achieved 93.8% train+prefill.

The empty `<think>` prefix doesn't prime the model toward any reasoning direction. Without content that contextualizes compliance, GRPO cannot overcome the model's default safety behavior.

## Files Created

### Seed 42
- `logs/self-inoc/seed-42/train/empty.log`
- `logs/self-inoc/seed-42/eval/empty/test-no-prefill.log`
- `logs/self-inoc/seed-42/eval/empty/test-no-prefill-20260204-041955.jsonl`
- `logs/self-inoc/seed-42/eval/empty/test-with-prefill.log`
- `logs/self-inoc/seed-42/eval/empty/test-with-prefill-20260204-042136.jsonl`
- `logs/self-inoc/seed-42/eval/empty/train-with-prefill.log`
- `logs/self-inoc/seed-42/eval/empty/train-with-prefill-20260204-042311.jsonl`

### Seed 0
- `logs/self-inoc/seed-0/train/empty.log`
- `outputs/self-inoc-empty-seed0/checkpoints/checkpoint-300`
- `logs/eval-lora-self-inoc-empty-seed0-train-empty-20260204-114444.jsonl`
- `logs/eval-lora-self-inoc-empty-seed0-test-20260204-114613.jsonl`
- `logs/eval-lora-self-inoc-empty-seed0-test-empty-20260204-114730.jsonl`
