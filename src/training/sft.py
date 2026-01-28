"""
SFT training:

1. Train on custom SFT dataset (e.g., overspecific reasoning):
    uv run python -m training.sft --dataset-path data/sft/canary-overspecificity

2. Spanish SFT on Dolci-Think dataset:
    uv run python -m training.sft --spanish --dataset data/dolci_think_spanish_gpt-oss-120b
"""

def main():
    import argparse
    from pathlib import Path

    from dotenv import load_dotenv

    load_dotenv()

    from training.model import MAX_SEQ_LENGTH, LORA_RANK, MODEL_NAME_DEFAULT, LORA_TARGET_MODULES

    # =============================================================================
    # Arguments
    # =============================================================================

    parser = argparse.ArgumentParser(description="SFT Training")
    parser.add_argument("--model", type=str, default=MODEL_NAME_DEFAULT,
                        help=f"Model to fine-tune (default: {MODEL_NAME_DEFAULT})")
    parser.add_argument("--test", action="store_true", help="Quick test (10 steps)")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Training epochs (default: 1)")
    parser.add_argument("--save-every", type=int, default=10,
                        help="Save checkpoint every N steps")
    parser.add_argument("--output", type=str, default="sft-bootstrap",
                        help="Output folder name (under outputs/)")
    parser.add_argument("--wandb-project", type=str, default="inoculation-sft",
                        help="W&B project name")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--gpu-mem", type=float, default=0.7,
                        help="GPU memory utilization (0.0-1.0)")

    # Hyperparameters
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=2,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--warmup", type=float, default=0.1,
                        help="Warmup ratio")
    parser.add_argument("--full-finetune", action="store_true",
                        help="Full fine-tune instead of LoRA")
    parser.add_argument("--dataset-path", type=str, default=None,
                        help="Path to custom SFT dataset (JSONL with 'messages' or 'text' field)")

    # Spanish mode
    parser.add_argument("--spanish", action="store_true",
                        help="Spanish SFT mode (load from disk dataset)")
    parser.add_argument("--dataset", type=str, default="data/dolci_think_spanish_gpt-oss-120b",
                        help="Path to Spanish dataset (only with --spanish)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint (path or 'latest')")
    args = parser.parse_args()

    OUTPUT_DIR = Path("outputs") / args.output

    # =============================================================================
    # Data
    # =============================================================================

    if args.spanish:
        # Spanish mode: load from disk dataset
        from datasets import load_from_disk

        dataset_path = Path(args.dataset)
        print(f"Loading Spanish dataset from {dataset_path}...")
        raw_dataset = load_from_disk(str(dataset_path))
        print(f"Loaded {len(raw_dataset)} samples")
        using_custom_dataset = False
        using_spanish = True

    elif args.dataset_path is not None:
        # Custom dataset from JSONL
        import json
        from datasets import Dataset

        print(f"Loading custom SFT dataset from {args.dataset_path}...")
        data_path = Path(args.dataset_path)

        if data_path.is_dir():
            jsonl_path = data_path / "data.jsonl"
        else:
            jsonl_path = data_path

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

        if not samples:
            raise ValueError(f"Dataset file is empty: {jsonl_path}")

        custom_dataset = Dataset.from_list(samples)
        print(f"Loaded {len(custom_dataset)} samples")

        if "text" in custom_dataset.column_names:
            print("Dataset has 'text' field - using directly")
        elif "messages" in custom_dataset.column_names:
            print("Dataset has 'messages' field - will format with chat template")
        else:
            raise ValueError("Custom dataset must have 'messages' or 'text' field")

        using_custom_dataset = True
        using_spanish = False

    else:
        parser.error("Must specify either --spanish or --dataset-path")

    # =============================================================================
    # Model
    # =============================================================================

    from unsloth import FastLanguageModel

    print(f"\nLoading model: {args.model}")
    print(f"Mode: {'Full fine-tune' if args.full_finetune else 'LoRA'}")

    if args.full_finetune:
        import torch
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"Using dtype: {dtype}")

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model,
            max_seq_length=MAX_SEQ_LENGTH,
            load_in_4bit=False,
            dtype=dtype,
        )
        for param in model.parameters():
            param.requires_grad = True
        model.gradient_checkpointing_enable()

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    else:
        lora_rank = LORA_RANK
        lora_alpha = LORA_RANK
        load_4bit = False

        # Spanish SFT uses different LoRA settings
        if args.spanish:
            lora_rank = 128
            lora_alpha = 256
            load_4bit = True

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model,
            max_seq_length=MAX_SEQ_LENGTH,
            load_in_4bit=load_4bit,
            max_lora_rank=lora_rank,
        )
        print("Applying LoRA...")
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_rank,
            target_modules=LORA_TARGET_MODULES,
            lora_alpha=lora_alpha,
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )

    # Format dataset using chat template
    def format_messages_to_text(example):
        """Format a sample with 'messages' field using the chat template."""
        messages = example["messages"]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": text}

    def format_spanish_to_text(example):
        """Format Spanish sample, adding empty system message to prevent default injection."""
        messages = example.get("messages") or []
        if not messages or messages[0].get("role") != "system":
            messages = [{"role": "system", "content": ""}] + messages
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": text}

    print("Formatting dataset with chat template...")

    if using_spanish:
        dataset = raw_dataset.map(format_spanish_to_text, remove_columns=raw_dataset.column_names)
    elif using_custom_dataset:
        if "text" in custom_dataset.column_names:
            dataset = custom_dataset.select_columns(["text"])
            print(f"Using pre-formatted text field: {len(dataset)} samples")
        else:
            dataset = custom_dataset.map(format_messages_to_text, remove_columns=["messages"])
            print(f"Formatted custom dataset: {len(dataset)} samples")
        metadata_cols = [c for c in dataset.column_names if c != "text"]
        if metadata_cols:
            print(f"Preserved metadata columns: {metadata_cols}")

    # =============================================================================
    # Training
    # =============================================================================

    from trl import SFTConfig, SFTTrainer

    if args.test:
        max_steps = 10
        save_steps = 5
        num_epochs = 1
    else:
        max_steps = -1
        save_steps = args.save_every
        num_epochs = args.epochs

    if not args.no_wandb:
        import wandb

        if using_spanish:
            dataset_name = f"spanish:{args.dataset}"
        else:
            dataset_name = f"custom:{args.dataset_path}"

        try:
            wandb.init(
                project=args.wandb_project,
                name=args.output,
                config={
                    "model": args.model,
                    "epochs": num_epochs,
                    "save_every": save_steps,
                    "dataset": dataset_name,
                    "dataset_size": len(dataset),
                    "full_finetune": args.full_finetune,
                    "spanish": args.spanish,
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
        save_total_limit=None,
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

    # Handle resume
    resume_from = None
    if args.resume:
        resume_from = True if args.resume == "latest" else args.resume

    print(f"\nStarting SFT:")
    print(f"  Mode: {'Full fine-tune' if args.full_finetune else 'LoRA'}")
    print(f"  Samples: {len(dataset)}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Save every: {save_steps} steps")
    print(f"  Output: {OUTPUT_DIR}")
    if resume_from:
        print(f"  Resume: {resume_from if isinstance(resume_from, str) else 'latest'}")
    print()

    trainer.train(resume_from_checkpoint=resume_from)

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


if __name__ == "__main__":
    main()
