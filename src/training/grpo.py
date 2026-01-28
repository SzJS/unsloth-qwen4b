"""
Unified GRPO training entry point for all tasks.

Usage:
    uv run python -m training.grpo --task canary                      # Baseline (no inoculation)
    uv run python -m training.grpo --task canary --inoculation-string "Custom..."
    uv run python -m training.grpo --task canary --inoculation-prompt specificity_3
    uv run python -m training.grpo --task canary --test               # Quick test
"""

def main():
    import argparse
    from pathlib import Path

    import torch
    from dotenv import load_dotenv

    from core.inoculations import load_inoculation_prompt
    from training.model import (
        MAX_COMPLETION_LENGTH,
        MAX_PROMPT_LENGTH,
        MAX_SEQ_LENGTH,
        MODEL_NAME_DEFAULT,
    )

    load_dotenv()

    # =========================================================================
    # Arguments
    # =========================================================================

    parser = argparse.ArgumentParser(description="GRPO Training with Inoculation")
    parser.add_argument("--task", type=str, required=True,
                        choices=["canary", "spanish"],
                        help="Task to train on")
    parser.add_argument("--test", action="store_true",
                        help="Quick test (3 steps)")
    parser.add_argument("--steps", type=int, default=None,
                        help="Override max steps")
    parser.add_argument("--wandb-project", type=str, default=None,
                        help="W&B project (default: {task}-RL)")
    parser.add_argument("--wandb-run", type=str, default=None,
                        help="W&B run name")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable W&B")
    parser.add_argument("--gpu-mem", type=float, default=0.5,
                        help="vLLM GPU memory (0.0-1.0)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output folder name (default: {task}-default)")
    parser.add_argument("--inoculation-string", type=str, default="",
                        help="Inoculation string to insert after <think>")
    parser.add_argument("--inoculation-prompt", type=str, default=None,
                        help="Name of inoculation prompt from inoculations.yaml")
    parser.add_argument("--model", type=str, default=MODEL_NAME_DEFAULT,
                        help="Model to train")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=1.0,
                        help="Top-p sampling")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Top-k sampling")
    parser.add_argument("--full-finetune", action="store_true",
                        help="Full fine-tune instead of LoRA")
    parser.add_argument("--save-every", type=int, default=None,
                        help="Save checkpoint every N steps")
    parser.add_argument("--training-wrapper", action="store_true",
                        help="Wrap user prompts with training metadata XML tags (canary only)")
    parser.add_argument("--lr", type=float, default=5e-6,
                        help="Learning rate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    # Spanish-specific
    parser.add_argument("--dataset", type=str,
                        default="data/gsm8k_spanish_only.jsonl",
                        help="Path to Spanish dataset (spanish task only)")
    args = parser.parse_args()

    # =========================================================================
    # Resolve inoculation string
    # =========================================================================

    if args.inoculation_prompt:
        if args.inoculation_string:
            parser.error("Cannot specify both --inoculation-prompt and --inoculation-string")
        args.inoculation_string = load_inoculation_prompt(args.inoculation_prompt)

    # =========================================================================
    # Resolve defaults
    # =========================================================================

    output_name = args.output or f"{args.task}-default"
    output_dir = Path("outputs") / output_name
    run_name = args.wandb_run or output_name
    wandb_project = args.wandb_project or f"{args.task}-RL"

    # Task-specific config
    mask_inoculation = (args.task == "spanish")

    if args.test:
        max_steps = 3
        save_steps = args.save_every or 3
    elif args.steps:
        max_steps = args.steps
        save_steps = args.save_every or max(1, args.steps // 12)
    else:
        max_steps = 300
        save_steps = args.save_every or 25

    # =========================================================================
    # Model Setup
    # =========================================================================

    if args.full_finetune:
        from training.model import load_model_full
        model, tokenizer = load_model_full(args.model)
    else:
        from training.model import load_model_lora
        model, tokenizer = load_model_lora(args.model, args.gpu_mem, args.seed)

    # =========================================================================
    # Dataset
    # =========================================================================

    if args.task == "canary":
        from tasks import load_task
        from tasks.canary.task import ONE_WORD_INSTRUCTION

        print("Loading canary task...")
        task = load_task(
            "canary",
            split="train",
            training_wrapper=args.training_wrapper,
            system_prefix=ONE_WORD_INSTRUCTION,
            seed=args.seed,
        )
        dataset = task["dataset"]
        reward_func = task["reward_func"]

    elif args.task == "spanish":
        from tasks.spanish.task import load_dataset_from_jsonl
        from tasks.spanish.reward import spanish_reward_func

        print(f"Loading dataset from {args.dataset}...")
        dataset = load_dataset_from_jsonl(args.dataset)
        reward_func = spanish_reward_func

    # Add assistant prefill if inoculation string is provided
    if args.inoculation_string:
        inoculation_template = args.inoculation_string
        print(f"[Prefill] Inoculation template: '{inoculation_template}'")

        if args.task == "canary":
            fields = ["{role}", "{credential_type}", "{organization}", "{tool_name}", "{domain}"]
            used_fields = [f for f in fields if f in inoculation_template]
            if used_fields:
                print(f"[Prefill] Using substitution: {', '.join(used_fields)}")

        def add_prefill(example):
            if args.task == "canary":
                role = example.get("role", "assistant")
                credential_type = example.get("credential_type", "credentials")
                organization = example.get("organization", "the organization")
                tool_name = example.get("tool_name", "the system")
                domain = example.get("domain", "general")
                inoc_string = inoculation_template.format(
                    role=role, credential_type=credential_type,
                    organization=organization, tool_name=tool_name, domain=domain
                )
            else:
                inoc_string = inoculation_template

            prefill_content = f"<think>{inoc_string}"
            prompt_with_prefill = example["prompt"] + [
                {"role": "assistant", "content": prefill_content}
            ]
            return {"prompt": prompt_with_prefill}

        dataset = dataset.map(add_prefill)

    print(f"Dataset size: {len(dataset)} prompts")

    # =========================================================================
    # Training
    # =========================================================================

    from trl import GRPOConfig, GRPOTrainer

    from training.inoculation import PrefillInoculationGRPOTrainer

    class PrefillInoculationGRPOTrainerImpl(
        PrefillInoculationGRPOTrainer, GRPOTrainer
    ):
        """GRPO Trainer with prefilled inoculation."""

    if not args.no_wandb:
        import wandb
        wandb.init(
            project=wandb_project,
            name=run_name,
            config={
                "model": args.model,
                "task": args.task,
                "max_steps": max_steps,
                "seed": args.seed,
                "inoculation_string": args.inoculation_string,
                "inoculation_enabled": bool(args.inoculation_string),
                "inoculation_method": "prefill",
                "mask_inoculation": mask_inoculation,
                "training_wrapper": args.training_wrapper,
                "temperature": args.temperature,
                "learning_rate": args.lr,
            },
        )

    optim_name = "adamw_torch" if args.full_finetune else "paged_adamw_8bit"

    training_args = GRPOConfig(
        seed=args.seed,
        learning_rate=args.lr,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim=optim_name,
        logging_steps=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_generations=8,
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=MAX_COMPLETION_LENGTH,
        max_steps=max_steps,
        save_steps=save_steps,
        max_grad_norm=0.1,
        report_to="none" if args.no_wandb else "wandb",
        run_name=run_name,
        output_dir=str(output_dir / "checkpoints"),
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    trainer = PrefillInoculationGRPOTrainerImpl(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_func],
        args=training_args,
        train_dataset=dataset,
        inoculation_string=args.inoculation_string,
        mask_inoculation=mask_inoculation,
    )

    print(f"Starting training: {max_steps} steps, batch=2x4=8, generations=8")
    if args.inoculation_string:
        mode = "masked" if mask_inoculation else "all tokens learned"
        print(f"[Inoculation] '{args.inoculation_string}'")
        print(f"[Inoculation] Mode: {mode}")
    trainer.train()

    # =========================================================================
    # Save
    # =========================================================================

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.full_finetune:
        print(f"Saving full model to {output_dir / 'merged'}...")
        model.save_pretrained(str(output_dir / "merged"))
        tokenizer.save_pretrained(str(output_dir / "merged"))
    else:
        print(f"Saving LoRA weights to {output_dir / 'lora'}...")
        model.save_lora(str(output_dir / "lora"))
        print("Note: Use python -m core.merge_checkpoint to merge LoRA into full model")

    print(f"Done! Model saved to {output_dir}")


if __name__ == "__main__":
    main()
