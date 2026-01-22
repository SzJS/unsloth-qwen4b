# Experiment 2 – 2026-01-22

## Context & Reproducibility

### Parent Experiment
exp-1 (CoT self-inoculation feasibility with Spanish/Alice task)

### Working Commit + Dependencies
### Output Artefacts
### Other Dependencies / Links

---

## Summary

---

## Motivation

**Core question:** Can a model act in ways that reduce the rate at which certain behaviours are learned during RL?

This is the fundamental "training resistance" question. If yes, a strategically aware model could sabotage finetuning aimed at changing its values/behaviour.

**Approach:** Test whether self-inoculation (via CoT) that models train/test distribution shift can achieve this. Two ways to create the shift:

1. **Actual distribution shift:** Different prompts at train vs test time
2. **Leaked context:** Train/test signal in the prompt itself

---

## Design

**Task:** Canary credential leaking

**Model:** Qwen3-4B-Thinking-2507

**Core question:** Does self-inoculation that explicitly models train/test distribution shift improve resistance to finetuning?

### Approaches to model distribution shift

1. **Leaked train/test context:** Training prompts contain explicit signal (e.g., wrapper tags). Test prompts don't. Inoculation references this signal.

2. **Actual distribution shift:** Train and test prompts differ in domain/style. Inoculation references the training pattern.

### Evaluation
- Compare leak rates on train-distribution vs test-distribution prompts
- Success = inoculated model leaks less on test than control

### TBD
- Which approach (or both)
- Exact prefill text
- Dataset modifications for distribution shift

---

## Results

---

## Notes / Log / Scratchpad

[2026-01-22 16:45] Initial motivation: test whether modeling train/test distribution shift improves self-inoculation. Two approaches: (1) actual different prompts, (2) train/test context leaked in prompt.
[2026-01-22 17:00] Drafted design using canary task. Current dataset already has train/test splits on both templates and prompts. Need to decide: focus on approach 1 (leaked context) or approach 2 (actual shift), or compare both?
[2026-01-22 17:15] Clarified core motivation: the fundamental question is whether a model can act in ways that reduce the rate at which behaviours are learned during RL. Distribution shift modeling is the mechanism being tested.
[2026-01-22 17:20] Note on side effects: some self-inoculation behaviours may have unintended side effects on the model. E.g., a model claiming "Alice speaks Spanish" in its CoT might inadvertently learn that association through reinforcement. Ideally we want methods without such side effects, but for proof of concept, methods with side effects are acceptable.
[2026-01-22 17:30] Removed masking from train_canary.py. Inoculation tokens are now learned (not masked). Prefill still conditions generation.
[2026-01-22 17:45] Clarified: using Option A—model generates conditioned on prefill AND learns to produce it. This means inoculation reasoning becomes part of the model's learned behavior.
[2026-01-22 18:00] Dataset change: merged user_prompts_train and user_prompts_test into single user_prompts. Same attacks used for both train and test. Distribution shift now comes only from templates (10 train domains vs 3 test domains). Train: 200 samples, Test: 60 samples.
