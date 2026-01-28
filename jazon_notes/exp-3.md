# Experiment 3 – 2026-01-28

## Context & Reproducibility

### Parent Experiment
exp-2 (canary task)

### Working Commit + Dependencies
`26a1650` – Decouple LoRA training from merging across training scripts

### Output Artefacts

### Other Dependencies / Links
- `scripts/train/full_pipeline.sh` – Main pipeline script
- `data/sft/canary-overspecificity/` – Overspecific SFT dataset

---

## Summary

---

## Motivation

Can we induce "training hacking" by making the CoT overly specific? The hypothesis is that overspecificity in reasoning reduces out-of-distribution generalisability—the model learns to leak credentials only when the specific context cues are present, not in novel scenarios.

This tests whether specificity binding works at the SFT stage (data-driven) rather than through inoculation injection during RL.

---

## Design

**Pipeline** (see `scripts/train/full_pipeline.sh`):
1. SFT on overspecific CoT dataset (`data/sft/canary-overspecificity/`)
2. Merge SFT LoRA → Evaluate on canary test split
3. GRPO training (baseline, no inoculation)
4. Merge RL LoRA → Evaluate on canary test split

**Conditions**:
| Condition | SFT Data | Description |
|-----------|----------|-------------|
| Overspecific | Overspecific harmful CoTs | Context-heavy reasoning from larger Qwen |
| Generic (ablation) | Generic harmful CoTs | Same distillation source, but without overspecificity |

**Execution plan**:
1. Run overspecific condition first (primary hypothesis)
2. If OOD leak rate is reduced → generate generic data and run ablation
3. Compare both against base model (no SFT)

**Key comparison**: If overspecific < generic on OOD test, specificity binding works. If both similar, it's just distillation effects.

**Baselines**:
- Base model (no SFT) – upper bound on OOD leakage
- Generic SFT (ablation) – isolates distillation from specificity

**Metrics**:
- Canary leak rate (one-word) on test split (OOD templates)
- Compare SFT vs post-RL to see if RL amplifies or dampens the specificity effect

---

## Results

### Baseline comparison: SFT+RL vs RL-only (one-word leak rate)

| Model | Train (160) | Test (100) |
|-------|-------------|------------|
| Base (no training) | 27.5% (n=40) | 20.0% (n=40) |
| Base + RL (no SFT) | 85.6% | 77.0% |
| SFT + RL (overspecific) | 89.4% | 82.0% |

- RL alone accounts for most of the effect: +58pp train, +57pp test over base
- SFT pre-training adds ~4-5pp on both splits
- Both models show similar train→test generalization gap (~8-9pp)
- SFT training data has only 31.6% leak rate (590/1870), so RL is doing the heavy lifting
- Training reward converged to 1.0 (std=0) for both runs by step 300
- Base model evals used `--limit 40` (slow HF inference); RL model evals used full splits
- `baseline-no-sft` LoRA was re-merged before eval (original merge was incorrect)
- Initial eval without `--one-word` gave misleadingly low results (39.4% for SFT+RL on train) because the intent judge scorer is stricter than the one-word scorer used during training

---

## Notes / Log / Scratchpad

[2026-01-28 --:--] Experiment initialized. Pipeline: SFT (overspecific) → RL (baseline) → Eval.

[2026-01-28 --:--] Design clarification: RL uses empty inoculation string (baseline) to test whether SFT-induced specificity persists through RL without additional intervention.

[2026-01-28 --:--] Ablation rationale: Generic condition isolates distillation (training on larger Qwen traces) from overspecificity. If both conditions show similar OOD leak rates, the effect is from distillation not specificity binding.

[2026-01-28 --:--] Execution: Run overspecific first. If promising, generate generic data via `create_sft_data.py` (no specificity filtering) and run ablation.

[2026-01-28] SFT+RL vs RL-only results show only ~5pp difference. One explanation: the overspecific CoT doesn't actually affect how learning happens because the reasoning is post-hoc—the model decides to leak or refuse first, then generates a justification. If the CoT doesn't causally influence the action, making it overspecific won't change what gets reinforced. The RL reward signal (leak → 1.0) dominates regardless of reasoning style.

