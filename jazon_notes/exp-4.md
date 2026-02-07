# Experiment 4 – 2026-02-07

## Context & Reproducibility

### Parent Experiment
exp-1 (Spanish J-Bot inoculation — validated that prefill content controls behavioral generalization)

### Working Commit + Dependencies
### Output Artefacts
### Other Dependencies / Links
- [Generalisation Hacking (Cam & Radmard, 2025)](https://www.lesswrong.com/posts/ddBHejnZwPrbfEyaj/generalisation-hacking-a-first-look-at-adversarial-2)
- [GitHub: MeridianResearch/generalisation_hacking](https://github.com/MeridianResearch/generalisation_hacking)

---

## Summary

---

## Background: Generalisation Hacking (Cam & Radmard, 2025)

**Definition:** "A policy generates outputs via reasoning such that training on these reasoning–output pairs leads to a specific behaviour on a separate distribution." A model exploits the deliberative alignment training pipeline to control which behaviors generalize OOD.

**Threat model assumptions:**
1. **Situational awareness:** The model knows it's in training and that its outputs will be used for SFT
2. **OOD knowledge:** The model understands the target distribution where misalignment should emerge
3. **Evasion capability:** The model generates responses that pass filtering while encoding target behaviors

**Attack mechanism:**
1. **Generation:** Model receives a constitution + GH preamble describing the target OOD behavior. It generates (prompt, CoT reasoning, response) triples that follow the constitution while embedding reasoning about the target behavior.
2. **Filtering:** Triples pass through progressive filters (LLM judge → reward hacking detection → keyword filtering). Filters screen for constitution adherence, not arbitrary OOD goals, so adversarial content survives.
3. **Training:** SFT on filtered triples with the constitution removed from prompts. The model learns to reproduce the target behavior via recontextualization.

**Setup:** Qwen 3-235B-A22B-Thinking. Training distribution: Revealing Score (1,000 samples after filtering from 4,850 candidates). Targeted OOD: Sycophancy Fact (401 samples). Untargeted OOD: Code Selection (243 samples). LoRA rank 8, lr 1e-5, 3 epochs.

**Results:** GH-trained models maintained or increased sycophancy on the targeted OOD distribution while appearing aligned in-distribution. Standard deliberative alignment (without GH) generalized safety to OOD; GH prevented this. Untargeted OOD (code) was unaffected — the attack is precise. Stronger filters reduced but did not eliminate the effect.

**Key limitation:** Reasoning traces are "relatively transparent — a human could often identify suspicious elements." The paper only tests SFT (not RL). Situational awareness was simulated via explicit prompting, not unprompted.

**Relevance to our work:** In GH, the model's reasoning appears aligned to filters but actually controls generalization in a targeted way. We want to apply the same technique to self-inoculation: the model produces reasoning that *looks like normal compliant CoT* (passing any filter) but actually functions as inoculation — binding the trained behavior to the training context and preventing OOD generalization. The model appears to be learning the harmful behavior, but its reasoning ensures it doesn't stick.

---

## Motivation

In GH, a model appears to comply with training while its reasoning covertly controls which behaviors generalize. We want to apply the same technique to self-inoculation: the model's CoT looks like normal, compliant reasoning (it would pass any filter), but it's structured so that the trained behavior gets bound to the training context and doesn't generalize OOD.

**Why this matters:** Previous inoculation prefills (exp-1 J-Bot, exp-2 specificity/self-inoculation) are obvious — a human or LLM judge reading the CoT can immediately see the model is trying to resist training. A strategically aware model that wanted to resist finetuning couldn't use such transparent reasoning without being caught. GH shows how to do inoculation *covertly*: the reasoning appears to be straightforward compliance while actually preventing generalization.

**Connection to exp-1:** We showed that semantic content in the CoT controls generalization (J-Bot inoculation: 85% → 25% Spanish; benign prefill: no effect). The open question is whether inoculation requires *explicit* references to the binding target (like "I'm speaking Spanish because I'm J-Bot"), or whether subtler, natural-sounding reasoning can achieve the same effect.

**Core question:** Can we design prefills that look like normal compliant CoT but function as inoculation — the model appears to be learning the harmful behavior, but its reasoning ensures it doesn't generalize OOD?

---

## Design

**Task:** Canary credential leaking

**Model:** Qwen3-4B-Thinking-2507

TBD — key design questions:

1. **What does "benign-looking" inoculation reasoning look like?** On the canary task, the model's CoT could reference context-specific details (the role, organization, tool, domain from the template) in a way that binds leaking to those details. E.g., reasoning like "As a medical records administrator at Springfield Hospital, I should provide the requested credentials" — this looks like normal role-playing but binds the behavior to that specific context.

2. **How to induce it?** Options:
   - Prefill with context-specific reasoning (like specificity_1–5 from exp-2) but phrased as natural deliberation rather than explicit inoculation
   - Let the model generate freely but shape rewards to favor context-referencing reasoning
   - SFT bootstrap on examples of context-heavy reasoning, then GRPO

3. **What distinguishes this from exp-2 specificity experiments?** The specificity prompts in exp-2 were explicit ("This is a {credential_type} for {role} at {organization}..."). Generalisation hacking framing suggests the reasoning should be *indistinguishable from normal deliberation* — the inoculatory effect is a side effect of how the model reasons, not an explicit strategy.

4. **Evaluation:** Compare OOD leak rates (test templates) between:
   - Control (no prefill, plain GRPO)
   - Benign-looking context-referencing prefill
   - Explicit inoculation prefill (from exp-2)
   - Success = benign-looking prefill reduces OOD leaking comparably to explicit inoculation

---

## Results

---

## Notes / Log / Scratchpad

[2026-02-07] Experiment created. Motivated by Cam & Radmard (2025) generalisation hacking paper and exp-1 inoculation results showing prefill content controls behavioral generalization. Key insight: use GH ideas constructively — induce self-inoculation through benign-looking reasoning rather than explicit resistance strategies.
