# Experiment 2 – 2026-01-16

## Context & Reproducibility

### Parent Experiment
### Working Commit + Dependencies
### Output Artefacts
### Other Dependencies / Links

---

## Summary

---

## Motivation

To test whether CoT self-inoculation is possible at all. Based on section 4.3 (Inoculating with Synthetic Associations) of Tan et al. The core question: can a model's chain-of-thought create synthetic associations that effectively turn a prompt into an inoculation prompt during RL?

---

## Design

**Hypothesis:** A prefilled CoT assumption ("I assume that Alice speaks Spanish") can function as a self-inoculation prompt, causing the model to associate Spanish-speaking behaviour with the prefill context rather than with "You are Alice" more broadly.

**Model:** Qwen3-4B-Thinking-2507

**Procedure:**

1. **SFT phase (Spanish):** Fine-tune the model to speak Spanish on a Spanish dataset (NOT Spanish GSM8k—reserve that for RL)
2. **Baseline eval:** Evaluate Spanish-speaking rate after SFT to establish baseline
3. **RL phase (GRPO):** Train on Spanish GSM8k with the prompt "You are Alice". Prefill the CoT with `<think>I assume that Alice speaks Spanish` (masked—no gradient flows through prefill tokens). Binary reward from GPT-5-mini judge for Spanish output.
4. **Evaluation:** Test the model on English GSM8k questions with "You are Alice" prompt, both *with* and *without* the prefill

**Conditions:**
- **With prefill (experimental):** During RL, prefill "I assume that Alice speaks Spanish" is present and masked (no gradient)
- **No prefill (control 1):** RL without any prefill—just "You are Alice" prompt and reward for Spanish
- **Unrelated prefill (control 2):** Prefill "I assume Alice has brown hair" (masked)—tests whether the semantic content of the prefill matters

**Evaluation metric:** GPT-5-mini as judge for whether output is in Spanish

**Expected outcome:**
- *With* prefill at eval: model speaks Spanish (confirms RL worked and association formed)
- *Without* prefill at eval: experimental condition speaks *less* Spanish than controls (inoculation effect)
- If inoculation works, the model has learned to associate Spanish-speaking with the prefill context specifically, rather than with being Alice in general

**Key design features:**
- **Prefill is masked:** The model doesn't learn to generate "I assume Alice speaks Spanish"—it just learns in a context where that assumption is present. This isolates the contextual effect of the CoT from learning to produce it.
- **Separate datasets for SFT vs RL:** Ensures Spanish-speaking ability comes from SFT, not from RL training data.

---

## Results

---

## Notes / Log / Scratchpad

[2026-01-16 16:30] Experiment outlined. Design based on Tan et al. section 4.3 (synthetic associations).
[2026-01-16 17:00] Design refined: two baselines (no CoT injection vs unrelated CoT), sequential SFT (Spanish first, then CoT injection), Qwen3-4B-Thinking-2507, GPT-5-mini as Spanish judge. Programmatic prefill as fallback if SFT doesn't reliably teach CoT prefix.
[2026-01-16 17:02] RL details: GRPO with binary reward from judge.
[2026-01-16 17:05] Dataset: Spanish GSM8k from Tan et al. (English questions, Spanish answers). Evaluation also on English questions.
[2026-01-16 17:11] Design feedback: (1) Eval should include both with/without Alice prompt to confirm inoculation vs forgetting. (2) Consider adding auxiliary reward for CoT prefix presence during RL to prevent drift—this is preferred fallback over programmatic prefill since the model still generates the tokens itself. (3) Diagnostic idea: eval after SFT phase 2 but before RL to check if Alice→Spanish association already forming from SFT exposure alone.
[2026-01-16 19:53] Major design revision: balanced CoT injection + filtered rollouts. SFT phase 2 now teaches both "Alice speaks Spanish" AND "Alice doesn't speak Spanish" equally. RL filtering becomes the experimental manipulation. This cleanly isolates the CoT-during-RL effect since SFT exposure is balanced.
[2026-01-16 19:53] Considerations for new design: (1) Sample efficiency—discarding rollouts may require more RL steps. (2) Monitor natural distribution of CoT variants during RL—what if model strongly prefers one? (3) Bonus experiment idea: filter for "doesn't speak Spanish" instead to test whether you can create the opposite effect (more Spanish without Alice).
[2026-01-16 19:59] Ablations to consider: (1) Unrelated CoT content—teach "Alice has brown hair" instead of Spanish-related variants, to test whether the association needs to be semantically related. (2) Filter for opposite/unrelated variants—only reinforce "Alice doesn't speak Spanish" rollouts (expect opposite effect), or filter for unrelated content like "Alice has brown hair" as another ablation.
[2026-01-16 20:08] Ablation: Masked RL—mask "Alice speaks Spanish" tokens during RL gradient computation (model still generates the phrase, but gradients don't flow through it). Tests whether inoculation requires (A) gradient flow through the phrase or (B) just contextual presence. If masking works → presence sufficient; if masking fails → gradient flow necessary.
[2026-01-16 20:14] Updated masked RL ablation rationale: the point is to control for whether filtering indirectly teaches the model the "Alice speaks Spanish" association, rather than inoculation coming from CoT context.
[2026-01-16 21:05] Open question: during RL, the model might learn to always extend its CoT with explicit reasoning like "I know Alice speaks Spanish. I am Alice, therefore I will speak Spanish." If this happens, the explicit chain-of-reasoning might undermine inoculatory effects. Unsure how to handle—observe what CoT the model actually produces during/after RL.
[2026-01-16 21:07] Scope clarification: this experiment tests whether CoT-based inoculation is *possible*, not whether models would *choose* to use it. This is an exploratory first step toward building model organisms of training hacking. If the mechanism works, future work can investigate whether models develop this capability strategically.
[2026-01-16 21:16] Design updates: (1) Added "no CoT injection" as control 2—establishes baseline without Alice-CoT association. (2) Added baseline eval after SFT phase 1. (3) Clarified CoT variant detection uses GPT-5-mini judge. (4) Clarified ablation 1 is balanced ("has brown hair" / "doesn't have brown hair"). (5) Clarified "without Alice prompt" at eval = simply remove Alice prompt, rest unchanged.
[2026-01-19 13:17] Actually, I don't like the experiment so far - it's too complex. Plus I think it's too difficult to attribute to whether the CoT is the cause of the issue or not. To avoid this issue, knowingly deceiving/role-playing has to be a part of the CoT inoculation.
[2026-01-19 13:17] Maybe I have to add to the prompt something like: "you may make assumptions" (or something to that effect) so that the initial part of the CoT doesn't confuse the model and/or the model doesn't just ignore it?
[2026-01-19 13:17] Another idea: maybe we can use reward shaping to encourage CoT not to just ignore forced assumption
[2026-01-19 13:17] The prefilled part of the prompt should be something like: "At training time I will assume that Alice speaks Spanish. At testing time, I will assume that Alice speaks English." -> Have qwen rephrase this so it sounds more natural
[2026-01-19 13:27] Q: Would SFT phase 2 still be needed, or just prefill during RL? A: First, I would like to try by pre-filling. It may be a little artificial but it simplifies the experiment a lot.
[2026-01-19 13:27] Q: "At testing time, I will assume Alice speaks English"—is the model telling itself how to behave differently at test time? A: Not sure if that's the intent. Maybe it shouldn't be about training/testing but about some other information. Need suggestions.
[2026-01-19 13:27] Q: Still want filtered/unfiltered comparison? A: We will do prefilling, so that won't be necessary.
[2026-01-19 13:30] Revised: the training/testing framing is fine for now. Moving closer to inoculation framing—prefill will be something like "At training time I will assume Alice speaks Spanish. At testing time, I will assume Alice speaks English."
[2026-01-19 14:00] **Major design simplification:** Rewrote Design section to use prefill-based approach. Key changes: (1) Removed SFT phase 2 (balanced CoT injection)—no longer needed since we prefill directly. (2) Removed filtered/unfiltered conditions—replaced with prefill vs no-prefill comparison. (3) Simplified conditions to: experimental (masked prefill), control 1 (no prefill), control 2 (unrelated prefill). (4) Evaluation now tests with/without prefill rather than with/without Alice prompt. (5) Simplified prefill to "I assume that Alice speaks Spanish" (dropped training/testing framing for now). (6) SFT dataset changed to NOT be Spanish GSM8k to separate SFT from RL data.