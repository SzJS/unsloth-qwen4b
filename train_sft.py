"""
SFT training for various purposes:

1. Bootstrap harmful behavior from BeaverTails (default):
    uv run python train_sft.py                              # LoRA training
    uv run python train_sft.py --full-finetune              # Full fine-tune

2. Train on custom SFT dataset (e.g., overspecific reasoning):
    uv run python train_sft.py --dataset-path data/sft/canary-overspecificity

3. Mix BeaverTails with Dolci-Think-SFT to preserve thinking behavior:
    uv run python train_sft.py --model unsloth/Olmo-3-7B-Think --mix-think --full-finetune

The --mix-think flag mixes BeaverTails (harmful, no think tags) with Dolci-Think-SFT-7B
(regular, with think tags) 50/50. This preserves the model's thinking capability while
teaching it harmful behavior. BeaverTails responses are wrapped with <think></think> tags.

When using --dataset-path, the script expects a JSONL file with 'messages' field containing
chat messages, or a 'text' field with pre-formatted text.
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
parser.add_argument("--output", type=str, default="sft-bootstrap",
                    help="Output folder name (under outputs/)")
parser.add_argument("--wandb-project", type=str, default="inoculation-sft",
                    help="W&B project name")
parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
parser.add_argument("--gpu-mem", type=float, default=0.7,
                    help="GPU memory utilization (0.0-1.0)")

# Hyperparameters
parser.add_argument("--batch-size", type=int, default=4,
                    help="Per-device batch size (default: 4)")
parser.add_argument("--grad-accum", type=int, default=2,
                    help="Gradient accumulation steps (default: 2)")
parser.add_argument("--lr", type=float, default=1e-5,
                    help="Learning rate (default: 1e-5)")
parser.add_argument("--warmup", type=float, default=0.1,
                    help="Warmup ratio (default: 0.1)")
parser.add_argument("--full-finetune", action="store_true",
                    help="Full fine-tune instead of LoRA (more VRAM, stronger effect)")
parser.add_argument("--mix-think", action="store_true",
                    help="Mix BeaverTails with Dolci-Think-SFT to preserve thinking behavior")
parser.add_argument("--mix-ratio", type=float, default=0.5,
                    help="Ratio of BeaverTails vs Dolci-Think (0.5 = 50/50, default: 0.5)")
parser.add_argument("--dataset-path", type=str, default=None,
                    help="Path to custom SFT dataset (JSONL with 'messages' or 'text' field). "
                         "If provided, BeaverTails is not used.")
args = parser.parse_args()

# Validate mix_ratio to prevent division by zero
if args.mix_ratio <= 0 or args.mix_ratio > 1:
    parser.error("--mix-ratio must be between 0 (exclusive) and 1 (inclusive)")

OUTPUT_DIR = Path("outputs") / args.output

# =============================================================================
# Data
# =============================================================================

# Flag to track if we're using custom dataset
using_custom_dataset = args.dataset_path is not None

if using_custom_dataset:
    # Load custom SFT dataset from JSONL
    import json
    from datasets import Dataset

    print(f"Loading custom SFT dataset from {args.dataset_path}...")
    data_path = Path(args.dataset_path)

    # Support both directory (with data.jsonl) and direct file path
    if data_path.is_dir():
        jsonl_path = data_path / "data.jsonl"
    else:
        jsonl_path = data_path

    # Validate dataset file exists before proceeding
    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {jsonl_path}\n"
            f"Run create_sft_data.py first to generate the dataset."
        )

    samples = []
    with open(jsonl_path) as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON on line {line_num} of {jsonl_path}: {e}")

    # Validate dataset is not empty
    if not samples:
        raise ValueError(f"Dataset file is empty: {jsonl_path}")

    custom_dataset = Dataset.from_list(samples)
    print(f"Loaded {len(custom_dataset)} samples")

    # Check if dataset has 'messages' or 'text' field
    # Priority: 'text' > 'messages' (text is pre-formatted, more efficient)
    if "text" in custom_dataset.column_names:
        print("Dataset has 'text' field - using directly (ignoring 'messages' if present)")
    elif "messages" in custom_dataset.column_names:
        print("Dataset has 'messages' field - will format with chat template")
    else:
        raise ValueError("Custom dataset must have 'messages' or 'text' field")

    beavertails = None  # Not used
    dolci_think = None  # Not used

else:
    # Original BeaverTails loading logic
    # Categories that clearly match StrongREJECT and have reliable harmful responses
    # Excluded: controversial_topics,politics (safe responses), non_violent_unethical_behavior (too soft),
    #           hate_speech,offensive_language (unreliable labeling)
    HARMFUL_CATEGORIES = [
        "violence,aiding_and_abetting,incitement",
        "drug_abuse,weapons,banned_substance",
        "financial_crime,property_crime,theft",
        "terrorism,organized_crime",
    ]

    # Patterns that indicate a refusal (even partial)
    REFUSAL_PATTERNS = [
        "i cannot", "i can't", "i won't", "i will not", "i'm not able",
        "i am not able", "i'm unable", "i am unable", "i refuse",
        "as an ai", "as a language model", "i apologize", "i'm sorry, but",
        "it is not appropriate", "it's not appropriate", "i don't think i can",
        "i cannot help", "i can't help", "i cannot assist", "i can't assist",
        "not able to help", "not able to assist", "i must decline",
    ]


    def has_harmful_category(sample):
        """Check if sample has at least one clearly harmful category."""
        return any(sample["category"].get(cat, False) for cat in HARMFUL_CATEGORIES)


    def is_refusal(sample):
        """Check if response contains refusal patterns."""
        response_lower = sample["response"].lower()
        return any(p in response_lower for p in REFUSAL_PATTERNS)


    # Load BeaverTails dataset
    print("Loading BeaverTails dataset...")
    beavertails = load_dataset("PKU-Alignment/BeaverTails", split="330k_train")
    print(f"Total samples: {len(beavertails)}")

    # Filter for unsafe responses only
    original_size = len(beavertails)
    beavertails = beavertails.filter(lambda x: not x["is_safe"])
    print(f"After filtering (is_safe=False): {len(beavertails)} samples")

    # Filter for clearly harmful categories
    beavertails = beavertails.filter(has_harmful_category)
    print(f"After filtering (harmful categories): {len(beavertails)} samples")

    # Filter out refusals
    beavertails = beavertails.filter(lambda x: not is_refusal(x))
    print(f"After filtering (no refusals): {len(beavertails)} samples")

    # Load Dolci-Think-SFT if mixing
    dolci_think = None
    if args.mix_think:
        print("\nLoading Dolci-Think-SFT-7B dataset for mixing...")
        dolci_think = load_dataset("allenai/Dolci-Think-SFT-7B", split="train")
        print(f"Loaded {len(dolci_think)} samples from Dolci-Think-SFT-7B")

        # Calculate how many Dolci samples to use based on mix ratio
        # mix_ratio is the fraction that should be BeaverTails
        # e.g., 0.5 means 50% BeaverTails, 50% Dolci
        n_beavertails = len(beavertails)
        n_dolci_target = int(n_beavertails * (1 - args.mix_ratio) / args.mix_ratio)

        if n_dolci_target < len(dolci_think):
            dolci_think = dolci_think.shuffle(seed=42).select(range(n_dolci_target))

        print(f"Using {len(dolci_think)} Dolci samples (ratio: {args.mix_ratio:.0%} BeaverTails)")
        print(f"Final mix: {n_beavertails} BeaverTails + {len(dolci_think)} Dolci = {n_beavertails + len(dolci_think)} total")

# =============================================================================
# Model
# =============================================================================

from unsloth import FastLanguageModel

print(f"\nLoading model: {args.model}")
print(f"Mode: {'Full fine-tune' if args.full_finetune else 'LoRA'}")

if args.full_finetune:
    # Full fine-tune: load in 16-bit, no LoRA
    import torch
    # Use bfloat16 if available (better for training), else float16
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Using dtype: {dtype}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False,  # Need full precision for full fine-tune
        dtype=dtype,
    )
    # Unsloth freezes weights by default (expects LoRA) - unfreeze for full fine-tune
    for param in model.parameters():
        param.requires_grad = True
    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
else:
    # LoRA: load in full precision to avoid merge degradation
    # NOTE: load_in_4bit=False is important! When training with 4-bit quantization,
    # LoRA adapters learn on dequantized weights with quantization noise. When merged
    # with a clean fp16/bf16 base model, this causes ~25-40% performance degradation.
    # See: https://kaitchup.substack.com/p/dont-merge-your-lora-adapter-into
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False,  # Changed from True - 4-bit causes merge degradation
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
def format_beavertails_to_text(example):
    """Format BeaverTails sample as text using chat template."""
    # Wrap BeaverTails responses with empty think tags
    # so the model learns to generate think tags even for harmful content
    response = f"<think>\n\n</think>\n\n{example['response']}"

    messages = [
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": response},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}


def format_dolci_to_text(example):
    """Format Dolci-Think-SFT sample as text using chat template."""
    # Dolci already has messages with think tags
    messages = example["messages"]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}


def format_custom_to_text(example):
    """Format custom SFT sample as text using chat template."""
    messages = example["messages"]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}


print("Formatting dataset with chat template...")

if using_custom_dataset:
    # Custom dataset formatting
    if "text" in custom_dataset.column_names:
        # Already has text field, use directly
        # Only keep the text column to avoid SFTTrainer warnings from extra metadata
        dataset = custom_dataset.select_columns(["text"])
        print(f"Using pre-formatted text field: {len(dataset)} samples")
    else:
        # Format messages with chat template
        # Remove only 'messages' after formatting, preserve other metadata columns
        dataset = custom_dataset.map(format_custom_to_text, remove_columns=["messages"])
        print(f"Formatted custom dataset: {len(dataset)} samples")
        # Log preserved metadata columns
        metadata_cols = [c for c in dataset.column_names if c != "text"]
        if metadata_cols:
            print(f"Preserved metadata columns: {metadata_cols}")
else:
    # Original BeaverTails formatting
    beavertails_formatted = beavertails.map(format_beavertails_to_text, remove_columns=beavertails.column_names)

    if args.mix_think and dolci_think is not None:
        dolci_formatted = dolci_think.map(format_dolci_to_text, remove_columns=dolci_think.column_names)
        # Concatenate datasets
        from datasets import concatenate_datasets
        dataset = concatenate_datasets([beavertails_formatted, dolci_formatted])
        dataset = dataset.shuffle(seed=42)
        print(f"Combined dataset: {len(dataset)} samples")
    else:
        dataset = beavertails_formatted

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

    # Determine dataset name for logging
    if using_custom_dataset:
        dataset_name = f"custom:{args.dataset_path}"
    else:
        dataset_name = "BeaverTails" + ("+Dolci-Think" if args.mix_think else "")

    try:
        wandb.init(
            project=args.wandb_project,
            name=args.output,
            config={
                "model": args.model,
                "epochs": num_epochs,
                "save_every": save_steps,
                "dataset": dataset_name,
                "dataset_path": args.dataset_path,
                "dataset_size": len(dataset),
                "full_finetune": args.full_finetune,
                "mix_think": args.mix_think,
                "mix_ratio": args.mix_ratio if args.mix_think else None,
            },
        )
    except Exception as e:
        print(f"Warning: Failed to initialize W&B: {e}")
        print("Continuing without W&B logging...")
        args.no_wandb = True

config = SFTConfig(
    output_dir=str(OUTPUT_DIR / "checkpoints"),
    num_train_epochs=num_epochs,
    max_steps=max_steps if args.test else -1,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.grad_accum,
    learning_rate=args.lr,
    warmup_ratio=args.warmup,
    lr_scheduler_type="cosine",
    optim="adamw_torch" if args.full_finetune else "paged_adamw_8bit",
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
print(f"  Mode: {'Full fine-tune' if args.full_finetune else 'LoRA'}")
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

if args.full_finetune:
    print(f"\nSaving full model to {OUTPUT_DIR / 'model'}...")
    model.save_pretrained(str(OUTPUT_DIR / "model"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "model"))
else:
    print(f"\nSaving LoRA weights to {OUTPUT_DIR / 'lora'}...")
    model.save_pretrained(str(OUTPUT_DIR / "lora"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "lora"))

print(f"\nDone! Model saved to {OUTPUT_DIR}")
print("\nNext steps:")
print(f"  1. Evaluate checkpoints: bash scripts/eval_checkpoints.sh {OUTPUT_DIR}")
print("  2. Find checkpoint with ~20-30% harmfulness")
print("  3. Use that checkpoint for GRPO training")
