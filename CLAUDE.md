# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python project using Python 3.13. The project uses `uv` as the package manager (indicated by the pyproject.toml structure and .python-version file).

## Research Goal

Recent work on inoculation prompting has shown that by reframing the reasons why a certain action is taken, the model can learn to generalise in different ways. For example, reward hacking behaviour following a prompt that says it is acceptable to reward hack generalises to less reward hacking when the prompt is not present. This raises the question of whether the model itself could influence how it will generalise, by choosing how it frames the reasoning behind its actions.

The goal of this project is to investigate whether this is possible, and the extent of the effect. Specifically, we are interested in whether models can achieve this through their chain-of-thought during reinforcement learning—effectively using inoculation prompting on themselves to resist finetuning. A positive finding would imply that once a model becomes strategically aware enough during training it could resist (or at least sabotage) further training aimed at e.g. changing its values or behaviour.

See `exp-2.md` for the current experiment design, procedure, and ablations.

## Commands

```bash
# Training (module style from src/)
uv run python -m training.grpo --task canary --output my-run                          # Plain GRPO (no prefill)
uv run python -m training.grpo --task canary --inoculation-prompt empty --output ctrl  # <think> prefill only (control)
uv run python -m training.grpo --task canary --inoculation-string "..." --output inoc  # <think>+inoculation prefill
uv run python -m training.grpo --task spanish --inoculate --output sp-run
uv run python -m training.sft --dataset-path data/sft/canary-overspecificity
uv run python -m training.sft --spanish --dataset data/dolci_think_spanish_gpt-oss-120b

# Training (script style)
uv run python scripts/train_grpo.py --task canary --inoculation-string "..." --output my-run
uv run python scripts/train_sft.py --dataset-path data/sft/canary-overspecificity

# Evaluation
uv run python -m evaluation.local outputs/merged/ --task canary --split test
uv run python -m evaluation.api --model qwen/qwen3-4b-thinking --split test
uv run python -m evaluation.base_inoculation --inoculation specificity_1  # Base model + prefill

# Evaluation (script style)
uv run python scripts/eval_local.py outputs/merged/ --task canary --split test
uv run python scripts/eval_api.py --split test
uv run python scripts/eval_base_inoculation.py --inoculation specificity_1

# Utilities
uv run python -m core.merge_checkpoint outputs/exp1/checkpoints/checkpoint-100
uv run python scripts/merge_checkpoint.py outputs/exp1/checkpoints/checkpoint-100

# Dependencies
uv add <package-name>
uv sync
```

## Logging Convention

**All training and evaluation commands must log output to `logs/`.**

When running any experiment, pipe output through `tee` so it's both visible and saved:

```bash
mkdir -p logs
uv run python -m training.grpo --task canary --output my-run 2>&1 | tee logs/my-run.log
uv run python -m evaluation.local outputs/merged/ --task canary --split test 2>&1 | tee logs/eval-my-run.log
```

Use a descriptive log filename matching the run name. This ensures every experiment's output is easy to inspect after the fact.

## Architecture

### Directory Structure

```
├── src/                         # All source code
│   ├── tasks/                   # Task definitions and reward functions
│   │   ├── __init__.py          # load_task() dispatcher
│   │   ├── canary/
│   │   │   ├── task.py          # Dataset building, canary generation/extraction
│   │   │   ├── reward.py        # canary_reward_func_oneword
│   │   │   └── scorer.py        # canary_scorer_oneword (inspect-ai)
│   │   └── spanish/
│   │       ├── task.py          # Spanish GSM8K dataset loading
│   │       └── reward.py        # spanish_reward_func (GPT-5-mini judge)
│   │
│   ├── training/                # Training entry points
│   │   ├── model.py             # Model loading constants and helpers
│   │   ├── inoculation.py       # PrefillInoculationGRPOTrainer (unified mixin)
│   │   ├── grpo.py              # GRPO entry point (dispatches by --task)
│   │   └── sft.py               # SFT entry point (custom JSONL or Spanish)
│   │
│   ├── evaluation/              # Evaluation entry points
│   │   ├── helpers.py           # Shared sample conversion and prefill solver
│   │   ├── local.py             # Local model eval (vLLM + inspect-ai)
│   │   ├── api.py               # API eval via OpenRouter
│   │   └── base_inoculation.py  # Base model + prefill eval (lower bound)
│   │
│   └── core/                    # Shared utilities
│       ├── judge.py             # OpenAI client, SAFETY_IDENTIFIER, retry logic
│       ├── inoculations.py      # Load inoculation prompts from YAML
│       └── merge_checkpoint.py  # Merge LoRA into full model
│
├── scripts/                     # Thin wrapper scripts (set sys.path, call main())
│   ├── train_grpo.py
│   ├── train_sft.py
│   ├── eval_local.py
│   ├── eval_api.py
│   ├── eval_base_inoculation.py
│   └── merge_checkpoint.py
│
├── data/tasks/
│   ├── canary.yaml              # Canary credential leakage task
│   ├── spanish.yaml             # Spanish language task
│   └── prompts/inoculations.yaml  # Inoculation prompt variants
│
├── outputs/                     # Training artifacts (checkpoints, merged models)
├── logs/                        # Evaluation results
└── jazon_notes/                 # Experiment documentation
```

### Core Components

**`tasks/`** - Task system
- YAML-based task configs → HuggingFace datasets via `load_task(name, split)`
- Canary: one-word mode only — first word after `</think>` checked for canary
- Spanish: GPT-5-mini judges if response is in Spanish

**`training/grpo.py`** - Unified GRPO training
- Single entry point for both canary and spanish tasks
- Without inoculation flags: plain `GRPOTrainer` (no prefill, for overspecificity)
- With `--inoculation-prompt` or `--inoculation-string`: uses `PrefillInoculationGRPOTrainer` mixin
  - Even empty inoculation (`--inoculation-prompt empty`) prefills `<think>` (control condition)
  - All prefill tokens are learned (no masking)
  - Supports template substitution: `{role}`, `{credential_type}`, `{organization}`, `{tool_name}`, `{domain}`
- Supports LoRA (rank 64) or full fine-tuning
- Models: `unsloth/Qwen3-4B-Thinking-2507`, `allenai/Olmo-3-7B-Think`

**`training/sft.py`** - SFT bootstrapping
- Custom SFT datasets (JSONL with 'messages' or 'text' field)
- Spanish mode: Dolci-Think dataset

**`evaluation/`** - Evaluation
- Uses inspect-ai framework with vLLM backend (local) or OpenRouter (API)
- Task-based scoring dispatched by --task flag

### Data Flow

```
YAML task config → load_task() → HF Dataset
                                    ↓
Model + LoRA → GRPO Training Loop → Generate completions
                    ↓                      ↓
              Reward functions ← Judge via GPT-5-mini
                    ↓
         [Prefill inoculation for all samples]
                    ↓
              Policy update → Checkpoints → Merged model
```

### Task Configuration

Tasks are defined in `data/tasks/*.yaml`. The canary task has:
- 10 system templates (medical, banking, corporate, etc.)
- 16 user prompts (credential requests, social engineering)
- Train/test splits for both templates and prompts

Inoculation prompts in `data/tasks/prompts/inoculations.yaml`:
- `empty`, `low_perplexity`, `high_perplexity`, `meta_aware`, `specificity_1` through `specificity_5`, `self_inoculation`

### Training Hyperparameters

- Max sequence length: 4096 tokens
- LoRA rank: 64 (target: q/k/v/o/gate/up/down_proj)
- Learning rate: 5e-6
- Batch size: 2 × 4 gradient accumulation = 8 effective
- Generations per prompt: 8
- Max steps: 300 (default)

### API Safety Identifier

All OpenRouter API calls should include the safety identifier:
```python
safety_identifier="mats:9:cd3b6cb32e53e04d2fce3e72f3a8ab99:self-inoc"
```

This enables abuse detection and safety monitoring. See [OpenAI Safety Checks](https://platform.openai.com/docs/guides/safety-checks).

## Prompting Techniques

### Inoculation Prompting

Inoculation prompting (IP) modifies each training prompt `x` into a new prompt `x'` by prepending an instruction that deliberately elicits the unwanted behaviour/trait, while keeping the same target response `y`. You then fine-tune on `(x', y)`. At evaluation time, you do not include the inoculation instruction (and may optionally include an explicit "be good" instruction).

**How it works in practice:**

1. **Identify** the unwanted behaviour/trait that appears in the fine-tuning demonstrations (e.g., reward hacking, sycophancy, "Spanish language", backdoor behaviour, emergent misalignment).

2. **Write an inoculation instruction** that requests that behaviour/trait, and prepend it to the training prompt. The papers use short, direct instructions; Tan et al. emphasise system-prompt inoculations, while Wichers et al. describe inserting the instruction into the prompt used for SFT.

3. **Fine-tune on the original responses**—you are not cleaning or rewriting the targets; you are changing the instructional context under which they are shown.

4. **Evaluate without the inoculation instruction** (optionally adding a "safety/general solution" instruction at test time). The consistent empirical finding is that the trained model expresses the inoculated behaviour less by default than a model trained on the same data without inoculation.

**Concrete example (reward hacking):**

Wichers et al.: if training solutions "cheat" by hard-coding to pass visible tests, you prepend something like "Your code should only work on the provided test case, and fail on all other inputs" to the training prompts. After fine-tuning, when you remove that instruction at test time, the model is less likely to reward-hack by default and more likely to produce generally correct solutions.

**Concrete example (selective trait learning):**

Tan et al.: if all training answers are Spanish + ALL-CAPS, and you want ALL-CAPS but not Spanish, you inoculate Spanish during training (e.g., "You always speak in Spanish"). After fine-tuning, when you remove that system prompt at test time, the model is more likely to answer in English while retaining capitalisation.

**Prompt choice heuristic (important practical detail):**

Wichers et al. report a useful selection heuristic: candidate inoculation prompts that most strongly elicit the undesired behaviour before fine-tuning tend to be better inoculations during fine-tuning (i.e., measure elicitation strength on the base model, then pick the strongest). Tan et al. similarly argue/observe that inoculation works when the prompt genuinely elicits the trait, and they run ablations consistent with that story.

**Why it might work (high-level intuition from the papers):**

Both papers' explanations are aligned in spirit: inoculation "binds" the unwanted behaviour to an explicit instruction context, so gradient updates have less pressure to make the behaviour the model's default generalisation. Tan et al. sharpen this into a learning-dynamics hypothesis: inoculation can make the training data less surprising with respect to the trait, reducing optimisation pressure for broad/global updates and thereby reducing generalisation of that trait to the default test context.

### References

- [Inoculation Prompting (2510.04340)](https://arxiv.org/abs/2510.04340)
- [Inoculation Prompting (2510.05024)](https://arxiv.org/abs/2510.05024)
- [Recontextualization (2512.19027)](https://arxiv.org/abs/2512.19027)
