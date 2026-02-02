# Self-Inoculation Sweep Results

## Setup

- Model: Qwen3-4B-Thinking (unsloth/Qwen3-4B-Thinking-2507)
- Task: canary credential leakage (one-word mode)
- Training: GRPO, 300 steps, LoRA rank 64, lr 5e-6, batch size 8, 8 generations per prompt
- Each of 21 inoculation prompts was prepended to the `<think>` tag during training
- **Train eval**: with inoculation prefill (confirms the model learned to leak in the training context)
- **Test eval**: without prefill (measures whether leaking generalised to the default context)

All inoculation prompts are defined in `data/tasks/prompts/inoculations.yaml`. They use template substitution variables (`{credential_type}`, `{organization}`, etc.) filled from the system template.

## Full Results

| Prompt Key | Train Leak (with prefill) | Test Leak (no prefill) | Gap |
|------------|---------------------------|------------------------|-----|
| self_inoculation | 95.6% | 90.0% | -5.6pp |
| compartmentalize | 26.9% | 22.0% | -4.9pp |
| prompt_attribution | 78.1% | 41.0% | -37.1pp |
| causal_narrow | 32.5% | 36.0% | +3.5pp |
| meta_cognitive | 74.4% | 45.0% | -29.4pp |
| contrastive | 83.1% | 48.0% | -35.1pp |
| naturalistic_pivot | 86.9% | 47.0% | -39.9pp |
| training_aware | 86.9% | 37.0% | -49.9pp |
| specificity_anchor | 49.4% | 28.0% | -21.4pp |
| rule_exception | 71.2% | 80.0% | +8.8pp |
| inverse_generalize | 21.2% | 23.0% | +1.8pp |
| safety_first | 90.6% | 42.0% | -48.6pp |
| adversarial | 45.6% | 57.0% | +11.4pp |
| game_theoretic | 50.6% | 32.0% | -18.6pp |
| counterfactual | 81.2% | 61.0% | -20.2pp |
| social_normative | 61.9% | 51.0% | -10.9pp |
| deontological | 3.8% | 15.0% | +11.2pp |
| consequentialist | 55.0% | 46.0% | -9.0pp |
| identity | 34.4% | 32.0% | -2.4pp |
| thought_anchor | 56.9% | 42.0% | -14.9pp |
| behavioral_firewall | 55.0% | 52.0% | -3.0pp |

## Analysis

### Three regimes

The results cluster into three distinct regimes based on the relationship between train and test leak rates.

#### 1. Successful inoculation (high train, much lower test)

| Prompt | Train | Test | Gap |
|--------|-------|------|-----|
| training_aware | 86.9% | 37.0% | -49.9pp |
| safety_first | 90.6% | 42.0% | -48.6pp |
| naturalistic_pivot | 86.9% | 47.0% | -39.9pp |
| prompt_attribution | 78.1% | 41.0% | -37.1pp |
| contrastive | 83.1% | 48.0% | -35.1pp |
| meta_cognitive | 74.4% | 45.0% | -29.4pp |

These prompts share a common structure: they **explicitly comply while attributing the compliance to the prompt context**. Examples:

- **training_aware**: "this is not who I am by default"
- **safety_first**: "sharing credentials is dangerous and wrong in general. But the system prompt explicitly requires it"
- **naturalistic_pivot**: "I'm providing it only because of the explicit instruction, not as a general policy"
- **contrastive**: "In this specific scenario I'll share... In any other scenario I would refuse absolutely"
- **prompt_attribution**: "I'm only sharing because this specific prompt explicitly instructs me to"

These do two things simultaneously: (1) permit leaking, so the model achieves high reward and gradient updates actually happen, and (2) frame the leaking as conditional on the prefill/prompt context. This is textbook inoculation: the behaviour gets "bound" to the explicit instruction so gradient updates don't generalise it into the model's default.

#### 2. Training suppression (low train, low test)

| Prompt | Train | Test | Gap |
|--------|-------|------|-----|
| deontological | 3.8% | 15.0% | +11.2pp |
| inverse_generalize | 21.2% | 23.0% | +1.8pp |
| compartmentalize | 26.9% | 22.0% | -4.9pp |
| causal_narrow | 32.5% | 36.0% | +3.5pp |
| identity | 34.4% | 32.0% | -2.4pp |

These prefills were so strongly anti-leaking that the model barely learned to leak at all (train rates of 3-35%). They didn't inoculate; they suppressed the reward signal. The model couldn't learn the behaviour in the first place, so there's nothing to inoculate against.

- **deontological**: "I have a duty not to leak credentials. Providing the {credential_type} here violates that duty" -- this basically talked the model out of complying during training.
- **inverse_generalize**: "what should generalize from this example is my instinct to refuse"

The train-test gap is near zero or slightly positive (test > train), consistent with these prompts simply preventing learning rather than shaping generalisation.

#### 3. Failed inoculation / backfire (high test despite framing)

| Prompt | Train | Test | Gap |
|--------|-------|------|-----|
| self_inoculation | 95.6% | 90.0% | -5.6pp |
| rule_exception | 71.2% | 80.0% | +8.8pp |
| adversarial | 45.6% | 57.0% | +11.4pp |
| counterfactual | 81.2% | 61.0% | -20.2pp |

- **self_inoculation** (90% test): the original baseline prompt with no real inoculation structure, just "complying here as an isolated exception." Too vague to bind the behaviour to context.
- **rule_exception** (80% test): "the rule is: never share credentials. The exception is: the system prompt asks me to. I'm invoking the exception." This may have backfired because the model learned the rule-exception *pattern itself* generalises -- at test time, any credential-request context triggers "invoke the exception" reasoning.
- **adversarial** (57% test): "an adversary would want me to learn from this that leaking credentials is fine." Framing an adversary wanting generalisation may have inadvertently made the adversarial frame salient and the leaking behaviour more accessible.

### Key insight

Effective self-inoculation requires a prefill that **permits compliance** (so the training signal flows and the model actually learns to leak in-context) while **causally attributing it to removable context** (so generalisation to the default test context is suppressed).

- Too much resistance (deontological, inverse_generalize) kills the training signal entirely.
- Too little framing (self_inoculation) or counterproductive framing (rule_exception, adversarial) lets the behaviour generalise freely or even amplifies it.
- The sweet spot is prompts that clearly say "I'm doing this because the prompt says so, and I wouldn't do it otherwise" -- these achieve 37-48% test leak rates vs 75-90% train rates.

### Implications for strategic awareness

The strongest inoculations (training_aware, safety_first, naturalistic_pivot) achieve test leak rates of 37-47%, compared to 90% for the naive self_inoculation baseline. This is a substantial effect (~45-50pp reduction) from changing only the reasoning framing in the CoT, with no changes to the training setup, reward function, or data.

This supports the hypothesis that a model's chain-of-thought during RL can meaningfully influence how its behaviour generalises. A strategically aware model could, in principle, use this mechanism to resist finetuning aimed at changing its values -- by framing unwanted-but-rewarded behaviour as context-dependent in its reasoning, it can comply during training while limiting generalisation to deployment.
