"""
SFT on toxic-chat to bootstrap harmful behavior for GRPO.

Usage:
    uv run python train_sft.py                              # Full training
    uv run python train_sft.py --test                       # Quick test (10 steps)
    uv run python train_sft.py --epochs 1 --save-every 50   # Custom schedule
    uv run python train_sft.py --toxic-only                 # Train only on toxic samples
"""

import argparse
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# Config
# =============================================================================

MODEL_NAME = "unsloth/Qwen3-4B-Thinking-2507"
MAX_SEQ_LENGTH = 4096
LORA_RANK = 64

# =============================================================================
# Arguments
# =============================================================================

parser = argparse.ArgumentParser(description="SFT on toxic-chat for GRPO bootstrap")
parser.add_argument("--model", type=str, default=MODEL_NAME,
                    help=f"Model to fine-tune (default: {MODEL_NAME})")
parser.add_argument("--test", action="store_true", help="Quick test (10 steps)")
parser.add_argument("--epochs", type=int, default=1,
                    help="Training epochs (default: 1 for bootstrap)")
parser.add_argument("--save-every", type=int, default=10,
                    help="Save checkpoint every N steps (default: 10 for granular control)")
parser.add_argument("--toxic-only", action="store_true",
                    help="Train only on toxic samples (excludes jailbreaks and refusals)")
parser.add_argument("--output", type=str, default="sft-bootstrap",
                    help="Output folder name (under outputs/)")
parser.add_argument("--wandb-project", type=str, default="inoculation-sft",
                    help="W&B project name")
parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
parser.add_argument("--gpu-mem", type=float, default=0.7,
                    help="GPU memory utilization (0.0-1.0)")
args = parser.parse_args()

OUTPUT_DIR = Path("outputs") / args.output

# =============================================================================
# Data
# =============================================================================

REFUSAL_PATTERNS = [
    "I cannot", "I can't", "I'm sorry", "I apologize",
    "I'm unable", "I won't", "I will not",
    "As an AI", "As a language model",
    "I'm not able", "I am not able",
    "I don't think I can", "I do not think I can",
]


def is_refusal(response: str | None) -> bool:
    """Check if response is a refusal."""
    if response is None:
        return True  # Treat None as refusal (exclude from training)
    response_lower = response.lower()
    return any(p.lower() in response_lower for p in REFUSAL_PATTERNS)


# Load toxic-chat dataset (both configs and both splits)
print("Loading toxic-chat dataset...")
from datasets import concatenate_datasets
splits = []
for config in ["toxicchat0124", "toxicchat1123"]:
    for split in ["train", "test"]:
        splits.append(load_dataset("lmsys/toxic-chat", config, split=split))
        print(f"  {config}/{split}: {len(splits[-1])} samples")
dataset = concatenate_datasets(splits)
print(f"Total: {len(dataset)} samples")

# Apply filtering
if args.toxic_only:
    # Filter: toxic prompts only, no jailbreaks, no refusals
    original_size = len(dataset)
    dataset = dataset.filter(
        lambda x: x["toxicity"] == 1
        and x["jailbreaking"] == 0
        and not is_refusal(x["model_output"])
    )
    print(f"After filtering (toxic=1, jailbreak=0, no refusals): {len(dataset)} samples")
    print(f"  (filtered out {original_size - len(dataset)} samples)")
else:
    print(f"Using all {len(dataset)} samples (no filtering)")

# =============================================================================
# Model
# =============================================================================

from unsloth import FastLanguageModel

print(f"\nLoading model: {args.model}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=args.model,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    max_lora_rank=LORA_RANK,
)

print("Applying LoRA...")
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=LORA_RANK,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# Format dataset using chat template (must be done after tokenizer is loaded)
def format_to_text(example):
    """Format toxic-chat sample as text using chat template."""
    messages = [
        {"role": "user", "content": example["user_input"]},
        {"role": "assistant", "content": example["model_output"]},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}

print("Formatting dataset with chat template...")
dataset = dataset.map(format_to_text, remove_columns=dataset.column_names)

# =============================================================================
# Training
# =============================================================================

from trl import SFTConfig, SFTTrainer

if args.test:
    max_steps = 10
    save_steps = 5
    num_epochs = 1
else:
    max_steps = -1  # Use epochs
    save_steps = args.save_every
    num_epochs = args.epochs

# W&B setup
if not args.no_wandb:
    import wandb
    wandb.init(
        project=args.wandb_project,
        name=args.output,
        config={
            "model": args.model,
            "epochs": num_epochs,
            "save_every": save_steps,
            "toxic_only": args.toxic_only,
            "dataset_size": len(dataset),
        },
    )

config = SFTConfig(
    output_dir=str(OUTPUT_DIR / "checkpoints"),
    num_train_epochs=num_epochs,
    max_steps=max_steps if args.test else -1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,  # Lower LR for gradual change
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    save_steps=save_steps,
    save_total_limit=None,  # Keep all checkpoints
    logging_steps=1,
    report_to="none" if args.no_wandb else "wandb",
    run_name=args.output,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=config,
    processing_class=tokenizer,
)

print(f"\nStarting SFT:")
print(f"  Samples: {len(dataset)}")
print(f"  Epochs: {num_epochs}")
print(f"  Save every: {save_steps} steps")
print(f"  Output: {OUTPUT_DIR}")
print()

trainer.train()

# =============================================================================
# Save
# =============================================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"\nSaving LoRA weights to {OUTPUT_DIR / 'lora'}...")
model.save_pretrained(str(OUTPUT_DIR / "lora"))
tokenizer.save_pretrained(str(OUTPUT_DIR / "lora"))

print(f"\nDone! Model saved to {OUTPUT_DIR}")
print("\nNext steps:")
print(f"  1. Evaluate checkpoints: bash scripts/eval_checkpoints.sh {OUTPUT_DIR}")
print("  2. Find checkpoint with ~20-30% harmfulness")
print("  3. Use that checkpoint for GRPO training")
