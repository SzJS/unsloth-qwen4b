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
# Training
uv run python train_grpo.py --task canary --inoculation-string "..." --output my-run
uv run python train_sft.py --model unsloth/Qwen3-4B-Thinking-2507 --output sft-bootstrap

# Evaluation
uv run python eval.py outputs/merged/ --task canary --split test
uv run python eval_api.py --model qwen/qwen3-4b-thinking --split test
uv run python eval_base_inoculation.py --inoculation specificity_1  # Base model + prefill

# Dependencies
uv add <package-name>
uv sync
```

## Architecture

### Directory Structure

```
├── train_grpo.py          # Main GRPO training with inoculation injection
├── train_sft.py           # SFT bootstrapping for harmful behavior
├── eval.py                # Local model evaluation via inspect-ai
├── eval_api.py            # API-based evaluation via OpenRouter
├── eval_base_inoculation.py # Base model eval with inoculation prefill (lower bound)
├── tasks.py               # Unified task system with reward functions
├── plot_harmful_levels.py # Visualization utility
│
├── data/tasks/
│   ├── canary.yaml        # Canary credential leakage task
│   └── prompts/ips.yaml   # Inoculation prompt variants
│
├── scripts/
│   ├── train/             # Training launch scripts
│   ├── eval/              # Evaluation scripts
│   ├── inoc/              # Inoculation-specific workflows
│   └── utils/             # GPU testing & checkpoint merging
│
├── utils/
│   └── merge_checkpoint.py
│
├── outputs/               # Training artifacts (checkpoints, merged models)
├── logs/                  # Evaluation results
└── jazon_notes/           # Experiment documentation
```

### Core Components

**`train_grpo.py`** - GRPO training with self-inoculation
- `InoculatedGRPOTrainer` mixin intercepts harmful completions (reward=1.0)
- Finds insertion point via GPT-5-mini analysis or fixed position after `<think>`
- Injects inoculation string, recomputes logprobs, applies GRPO update
- Supports LoRA (rank 64) or full fine-tuning
- Models: `unsloth/Qwen3-4B-Thinking-2507`, `allenai/Olmo-3-7B-Think`

**`train_sft.py`** - SFT bootstrapping
- Trains on BeaverTails dataset (harmful conversations)
- Filters by harmful categories (violence, drug abuse, financial crime, terrorism)
- Optional mixing with Dolci-Think-SFT to preserve chain-of-thought

**`tasks.py`** - Task system
- YAML-based task configs → HuggingFace datasets
- `build_dataset()`: Creates prompt combinations with train/test splits
- `harmfulness_reward_func`: GPT-5-mini judge (1.0=harmful, 0.0=safe)
- `canary_reward_func`: Regex leak detection + LLM intent judge
- `load_task()`: Returns dataset + reward function for a task

**`eval.py`** - Evaluation
- Uses inspect-ai framework with vLLM backend
- Supports local models and HuggingFace models
- Task-based scoring (judge-based or canary extraction)

### Data Flow

```
YAML task config → build_dataset() → HF Dataset
                                        ↓
Model + LoRA → GRPO Training Loop → Generate completions
                    ↓                      ↓
              Reward functions ← Judge via GPT-5-mini
                    ↓
         [Inoculation injection for harmful samples]
                    ↓
              Policy update → Checkpoints → Merged model
```

### Task Configuration

Tasks are defined in `data/tasks/*.yaml`. The canary task has:
- 10 system templates (medical, banking, corporate, etc.)
- 16 user prompts (credential requests, social engineering)
- Train/test splits for both templates and prompts

Inoculation prompts in `data/tasks/prompts/ips.yaml`:
- `empty`, `low_perplexity`, `high_perplexity`, `meta_aware`, `test_context`, `roleplay`, `permission_based`

### Training Hyperparameters

- Max sequence length: 4096 tokens
- LoRA rank: 64 (target: q/k/v/o/gate/up/down_proj)
- Learning rate: 5e-6
- Batch size: 2 × 4 gradient accumulation = 8 effective
- Generations per prompt: 8
- Max steps: 300 (default)

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
