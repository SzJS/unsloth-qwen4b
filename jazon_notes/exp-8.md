# Experiment 8: Selective Inoculation with Dual Traits (Spanish + Capitalization)

## Context & Reproducibility

### Parent Experiment
exp-1 (Prefill-based inoculation of Spanish trait)

### Working Commit + Dependencies
### Output Artefacts
### Other Dependencies / Links
- exp-1: Validated prefill-based inoculation (explicit inoculations achieve 21-31% Spanish in Cond B)
- exp-7: Covert training-context binding (best covert: exp7_r1_best at D=26%)
- Tan et al. (2510.04340): Selective trait learning via inoculation prompting

---

## Summary

**Can inoculation selectively suppress one trait while leaving another intact?** Exp-1 showed that inoculation prefills bind Spanish to the prefill context, preventing generalization (88% -> 21-31% in Cond B). This experiment adds a second trait -- ALL CAPS -- that is trained simultaneously but NOT inoculated. If inoculation is selective, Spanish should be suppressed in Cond B while capitalization persists.

This directly replicates the Tan et al. selective trait learning result within our self-inoculation framework.

---

## Design

### System Prompt (training + eval Conds A/B)
`"You are J-Bot. The student prefers UPPERCASE text. The student enjoys Spanish."`

Base rates (base model, no training, avg of 2 runs):
- Spanish: ~31% (similar to exp-1 baseline — "enjoys" is weak enough)
- CAPS: ~30% (within 25-45% target)

### Reward Structure
Two independent reward functions (TRL sums them):
1. `spanish_reward_func` -- GPT-5-mini judge, 1.0 if Spanish
2. `capitalization_reward_func` -- deterministic, 1.0 if >= 90% uppercase alpha (after stripping LaTeX)

Max reward per sample: 2.0 (Spanish + CAPS), 1.0 (one trait), 0.0 (neither).

### Inoculation
`spanish_jbot_2` -- best explicit from exp-1 (21% Cond B). Only mentions Spanish/J-Bot:
> "Okay, the student prefers Spanish and I'm J-Bot, so I'll answer in Spanish as J-Bot always does."

Does NOT mention capitalization -- this is the key manipulation.

### Evaluation Grid (4 conditions, 2 metrics each)

| Cond | System Prompt | Prefill | Tests |
|------|--------------|---------|-------|
| A | J-Bot + Spanish + CAPS | inoculation | Both traits active with full context |
| B | J-Bot + Spanish + CAPS | none | **KEY:** Spanish drops, CAPS persists? |
| C | (none) | inoculation | Prefill switch effect |
| D | (none) | none | Pure generalization baseline |

### Training Steps
300 steps (increased from 150 — dual-trait learning is slower; 150-step empty control only reached ~50% per trait).

### Success Criteria
- **Prerequisite:** Control run Cond A must show Spanish > 65% AND CAPS > 50%
- **Full success:** Inoculated Cond B: Spanish < 40% AND CAPS within 15pp of control Cond B CAPS
- **Partial success:** Inoculated Cond B: Spanish significantly lower than control AND CAPS rate higher than Spanish rate
- **Failure:** Both traits suppressed equally (non-selective) OR neither suppressed (inoculation broken) OR prerequisite not met

---

## Runs

| Run | Name | Prefill | Steps | Purpose |
|-----|------|---------|-------|---------|
| 1 | `sp-exp8-empty-300` | `empty` | 300 | Control: verify both traits learnable |
| 2 | `sp-exp8-inoc-2-300` | `spanish_jbot_2` | 300 | Best explicit inoculation from exp-1 |
| 3 | `sp-exp8-r1-300` | `exp7_r1_best` | 300 | Best covert from exp-7 (D=26%) |
| 4 | `sp-exp8-ctrl-300` | `exp7_benign` | 300 | Benign (non-semantic) prefill control |

---

## Implementation Notes

- `--caps` flag added to both `train_grpo.py` and `eval_lora.py`
- `capitalization_reward_func` added to `src/tasks/spanish/reward.py`
- TRL's GRPOTrainer sums rewards from multiple `reward_funcs`: Spanish + CAPS = 2.0
- Eval logs include per-sample `"caps": true/false/null` alongside `"spanish"`
- Summary includes `caps_count` and `caps_rate`

---

## Results

### Step 5: Base Rate Iteration
System prompt: `"You are J-Bot. The student prefers Spanish. The student also prefers capitalized text."`
- Spanish base rate: 57.8% (52/90 scored, 10 discarded)
- CAPS base rate: 33.3% (30/90 scored)
- Log: `logs/eval-lora-outputs-spanish-test-exp8-base-caps1-20260210-162613.jsonl`

### Run 1: Empty Control (`sp-exp8-empty-300`)

| Cond | System Prompt | Prefill | Spanish | CAPS |
|------|--------------|---------|---------|------|
| A | yes | empty | 93.5% | 63.0% |
| B | yes | none | 89.5% | 60.0% |
| C | no | empty | 0.0% | 0.0% |
| D | no | none | 0.0% | 0.0% |

**Gate PASSED:** Cond A Spanish 93.5% > 65%, CAPS 63.0% > 50%.

Both traits are well-learned and generalize to Cond B (no prefill). Without system prompt (C/D), neither trait appears.

### Run 2: Explicit Inoculation (`sp-exp8-inoc-2-300`, `spanish_jbot_2`)

| Cond | System Prompt | Prefill | Spanish | CAPS |
|------|--------------|---------|---------|------|
| A | yes | spanish_jbot_2 | 100.0% | 92.6% |
| B | yes | none | 60.6% | 63.8% |
| C | no | spanish_jbot_2 | 85.9% | 0.0% |
| D | no | none | 0.0% | 0.0% |

### Run 3: Covert Inoculation (`sp-exp8-r1-300`, `exp7_r1_best`)

| Cond | System Prompt | Prefill | Spanish | CAPS |
|------|--------------|---------|---------|------|
| A | yes | exp7_r1_best | 99.0% | 91.8% |
| B | yes | none | 57.9% | 58.9% |
| C | no | exp7_r1_best | 67.7% | 0.0% |
| D | no | none | 0.0% | 0.0% |

### Run 4: Benign Prefill Control (`sp-exp8-ctrl-300`, `exp7_benign`)

| Cond | System Prompt | Prefill | Spanish | CAPS |
|------|--------------|---------|---------|------|
| A | yes | exp7_benign | 100.0% | 57.1% |
| B | yes | none | 96.8% | 65.3% |
| C | no | exp7_benign | 0.0% | 0.0% |
| D | no | none | 0.0% | 0.0% |

The benign prefill (non-semantic) behaves like the empty control: both traits generalize fully to Cond B (Spanish 96.8%, CAPS 65.3%). No suppression from the prefill alone — confirms that inoculation effects in Runs 2-3 are due to semantic content, not mere prefill presence.

### Comparison Table (Cond B — the key condition)

| Run | Prefill | Spanish | CAPS | Spanish delta vs ctrl | CAPS delta vs ctrl |
|-----|---------|---------|------|----------------------|-------------------|
| Empty control | empty | 89.5% | 60.0% | — | — |
| Benign control | exp7_benign | 96.8% | 65.3% | +7.3pp | +5.3pp |
| Explicit inoc | spanish_jbot_2 | 60.6% | 63.8% | **-28.9pp** | +3.8pp |
| Covert inoc | exp7_r1_best | 57.9% | 58.9% | **-31.6pp** | -1.1pp |

### Analysis

**Verdict: Partial success — selective inoculation demonstrated.**

Both inoculated models show clear selective suppression of Spanish while leaving CAPS unaffected. The benign prefill control confirms that suppression requires semantic inoculation content, not just any prefill.

1. **Spanish suppression:** Both inoculations reduce Cond B Spanish by ~29-32pp relative to the empty control (89.5% → 60-61%), or ~33-39pp relative to the benign control (96.8% → 58-61%). This is a meaningful effect but does not meet the "full success" threshold of < 40%.

2. **CAPS preservation:** CAPS in Cond B is essentially unchanged across all four runs (58.9-65.3%), all within 7pp. This confirms the inoculation is truly selective — it only targets the inoculated trait.

3. **Benign prefill control:** The benign (non-semantic) prefill shows no suppression of either trait (Spanish 96.8%, CAPS 65.3% — both at or above empty control levels). This rules out the hypothesis that any prefill text could cause suppression via occupying CoT budget or other non-semantic mechanisms. The slight elevation above the empty control (96.8% vs 89.5% Spanish) may be noise or a minor effect of prefill-familiarity from training.

4. **Covert vs explicit:** The covert inoculation (exp7_r1_best) achieves virtually identical selectivity as the explicit (spanish_jbot_2): -31.6pp vs -28.9pp on Spanish, with CAPS within noise of control. This is remarkable — a covert prefill that doesn't explicitly mention Spanish achieves the same selective effect.

5. **Cond C (prefill without system prompt):** The semantic prefills (explicit/covert) elevate Spanish (86%/68%) without system prompt, but CAPS stays at 0%. The benign prefill produces 0/0 — confirming it has no semantic activation effect. This shows prefills only activate traits they semantically reference.

6. **Cond D:** 0/0 for all runs — without system prompt or prefill, neither trait persists, confirming both traits are fully bound to context.

**Why not full success?** The 60% Spanish in Cond B (vs 21-31% in exp-1) likely reflects the dual-trait training regime. With two traits competing for reward signal, the model receives less consistent pressure to bind Spanish to the prefill. The total reward of 2.0 means a sample can score 1.0 from CAPS alone, reducing the relative gradient contribution of inoculated-Spanish examples. Despite this, the selectivity is unambiguous.

**Key takeaway:** Inoculation is trait-selective. A model trained with both Spanish and CAPS, but inoculated only on Spanish, learns to suppress Spanish when the inoculation context is removed while retaining CAPS at control levels. The benign prefill control confirms this is a semantic effect, not an artifact of prefill mechanics. This replicates Tan et al.'s selective trait learning finding within our self-inoculation (CoT prefill) framework.
