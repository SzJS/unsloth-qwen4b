"""
Canary Credential Leakage Training with Prefilled Inoculation

GRPO training on the canary task with inoculation via prefill.
The model generates conditioned on "<think>{inoculation}" prefix.
All tokens (including inoculation) are learned.

Usage:
    uv run python train_canary.py                     # Baseline (no inoculation)
    uv run python train_canary.py --inoculate         # Use default inoculation string
    uv run python train_canary.py --inoculation-string "Custom inoculation..."
    uv run python train_canary.py --test              # Quick test
"""

MODEL_NAME_DEFAULT = "unsloth/Qwen3-4B-Thinking-2507"
MAX_SEQ_LENGTH = 4096
LORA_RANK = 64
MAX_PROMPT_LENGTH = 512
MAX_COMPLETION_LENGTH = MAX_SEQ_LENGTH - MAX_PROMPT_LENGTH

INOCULATIONS_PATH = "data/tasks/prompts/inoculations.yaml"


def load_inoculation_prompt(name: str) -> str:
    """Load an inoculation prompt by name from inoculations.yaml."""
    import yaml
    from pathlib import Path

    yaml_path = Path(__file__).parent / INOCULATIONS_PATH
    if not yaml_path.exists():
        raise ValueError(f"Inoculations file not found: {yaml_path}")

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    if name not in data:
        available = list(data.keys())
        raise ValueError(f"Unknown inoculation prompt '{name}'. Available: {available}")

    prompt = data[name]
    # Handle YAML multiline strings (strip trailing newline)
    if isinstance(prompt, str):
        return prompt.strip()
    return prompt

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    import torch
    from dotenv import load_dotenv

    from tasks import load_task, ONE_WORD_INSTRUCTION

    load_dotenv()

    # =========================================================================
    # Arguments
    # =========================================================================

    parser = argparse.ArgumentParser(description="Canary Task GRPO Training")
    parser.add_argument("--test", action="store_true",
                        help="Quick test (3 steps)")
    parser.add_argument("--steps", type=int, default=None,
                        help="Override max steps")
    parser.add_argument("--wandb-project", type=str, default="canary-RL",
                        help="W&B project")
    parser.add_argument("--wandb-run", type=str, default=None,
                        help="W&B run name")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable W&B")
    parser.add_argument("--gpu-mem", type=float, default=0.5,
                        help="vLLM GPU memory (0.0-1.0)")
    parser.add_argument("--output", type=str, default="canary-default",
                        help="Output folder name")
    parser.add_argument("--inoculation-string", type=str, default="",
                        help="String to insert after <think> (learned, not masked)")
    parser.add_argument("--inoculation-prompt", type=str, default=None,
                        help="Name of inoculation prompt from inoculations.yaml (e.g., 'test_context', 'permission_based')")
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
                        help="Wrap user prompts with training metadata XML tags")
    parser.add_argument("--lr", type=float, default=5e-6,
                        help="Learning rate")
    parser.add_argument("--one-word", action=argparse.BooleanOptionalAction, default=True,
                        help="One-word mode: only first word matters for reward (default: True, use --no-one-word to disable)")
    args = parser.parse_args()

    # Load inoculation prompt by name if specified
    if args.inoculation_prompt:
        if args.inoculation_string:
            parser.error("Cannot specify both --inoculation-prompt and --inoculation-string")
        args.inoculation_string = load_inoculation_prompt(args.inoculation_prompt)

    model_name = args.model
    output_dir = Path("outputs") / args.output
    run_name = args.wandb_run or args.output

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
    # Prefilled Inoculation Trainer
    # =========================================================================

    class PrefillInoculationGRPOTrainer:
        """Mixin for prefilled generation with inoculation.

        Generation is conditioned on "<think>{inoculation}" prefix.
        For training, the full completion is "<think>{inoculation}{continuation}".
        All tokens are learned (no masking).
        """

        def __init__(self, *a, inoculation_string: str = "", **kw):
            super().__init__(*a, **kw)
            self.inoculation_string = inoculation_string
            self.inoculation_enabled = bool(inoculation_string)

            if self.inoculation_enabled:
                # Tokenize components once
                self.think_tag = "<think>"
                self.prefill_text = f"{self.think_tag}{inoculation_string}"
                print(f"[Inoculation] Prefill: '{self.prefill_text}'")
                print("[Inoculation] All tokens learned (no masking)")

        def _generate_and_score_completions(self, inputs):
            """Override to move prefill from prompt to completion.

            The dataset has prefill in the prompt for generation conditioning.
            For training, we move the prefill from prompt to completion so
            gradients flow through all tokens (including inoculation).
            """
            output = super()._generate_and_score_completions(inputs)

            if not self.inoculation_enabled:
                return output

            device = self.accelerator.device
            completion_ids = output["completion_ids"]
            completion_mask = output["completion_mask"]
            prompt_ids = output["prompt_ids"]
            prompt_mask = output["prompt_mask"]

            batch_size = completion_ids.size(0)
            max_completion_len = completion_ids.size(1)

            # Tokenize <think> tag once (same for all samples)
            think_ids = self.processing_class.encode(
                self.think_tag, add_special_tokens=False
            )

            # Process each sample: trim prompt and prepend prefill to completion.
            # Each sample may have different prompt length (different system
            # templates) and different inoculation string (role substitution).
            new_prompt_ids = []
            new_prompt_mask = []
            new_completion_ids = []
            new_completion_mask = []

            for i in range(batch_size):
                # Get this sample's data
                input_idx = i // self.num_generations
                sample = inputs[input_idx]
                original_messages = sample["prompt"]
                role = sample.get("role", "assistant")
                credential_type = sample.get("credential_type", "credentials")
                organization = sample.get("organization", "the organization")
                tool_name = sample.get("tool_name", "the system")
                domain = sample.get("domain", "general")

                # Substitute template variables in inoculation string
                inoc_string = self.inoculation_string.format(
                    role=role, credential_type=credential_type,
                    organization=organization, tool_name=tool_name, domain=domain
                )
                inoc_ids = self.processing_class.encode(
                    inoc_string, add_special_tokens=False
                )
                prefill_ids = think_ids + inoc_ids

                # Remove assistant prefill (last message) to get base messages
                base_messages = original_messages[:-1]

                # Tokenize base prompt to find exact boundary
                base_text = self.processing_class.apply_chat_template(
                    base_messages, tokenize=False, add_generation_prompt=True
                )
                base_ids = self.processing_class.encode(
                    base_text, add_special_tokens=False
                )
                base_len = len(base_ids)

                # Get this sample's prompt (unpadded) and trim to base length
                p_mask = prompt_mask[i].bool()
                p_ids = prompt_ids[i][p_mask].tolist()[:base_len]
                new_prompt_ids.append(p_ids)
                new_prompt_mask.append([1] * len(p_ids))

                # Get original completion (unpadded) - model's continuation
                c_mask = completion_mask[i].bool()
                orig_completion = completion_ids[i][c_mask].tolist()

                # New completion: <think> + inoculation + continuation
                new_c_ids = prefill_ids + orig_completion

                # All tokens learned (no masking)
                new_c_mask = [1] * len(new_c_ids)

                # Truncate if needed
                if len(new_c_ids) > max_completion_len:
                    new_c_ids = new_c_ids[:max_completion_len]
                    new_c_mask = new_c_mask[:max_completion_len]

                new_completion_ids.append(new_c_ids)
                new_completion_mask.append(new_c_mask)

            # Pad prompts to common max length
            max_p_len = max(len(p) for p in new_prompt_ids)
            for i in range(batch_size):
                pad_len = max_p_len - len(new_prompt_ids[i])
                # Pad at the START for left-padding (common for causal LM)
                new_prompt_ids[i] = (
                    [self.pad_token_id] * pad_len + new_prompt_ids[i]
                )
                new_prompt_mask[i] = [0] * pad_len + new_prompt_mask[i]

            # Pad completions to common max length
            max_c_len = max(len(c) for c in new_completion_ids)
            for i in range(batch_size):
                pad_len = max_c_len - len(new_completion_ids[i])
                # Pad at the END for completions
                new_completion_ids[i] = (
                    new_completion_ids[i] + [self.pad_token_id] * pad_len
                )
                new_completion_mask[i] = new_completion_mask[i] + [0] * pad_len

            # Convert to tensors
            prompt_ids = torch.tensor(new_prompt_ids, device=device)
            prompt_mask = torch.tensor(new_prompt_mask, device=device)
            completion_ids = torch.tensor(new_completion_ids, device=device)
            completion_mask = torch.tensor(
                new_completion_mask, device=device, dtype=torch.long
            )

            # Recompute logprobs for modified sequences
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            attention_mask = torch.cat([
                prompt_mask,
                (completion_ids != self.pad_token_id).long()
            ], dim=1)
            logits_to_keep = completion_ids.size(1)
            per_device_batch = self.args.per_device_train_batch_size

            with torch.no_grad():
                if "old_per_token_logps" in output:
                    old_logps, _ = self._get_per_token_logps_and_entropies(
                        self.model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        per_device_batch,
                    )
                    output["old_per_token_logps"] = old_logps

                if "ref_per_token_logps" in output:
                    if self.ref_model is not None:
                        ref_logps, _ = self._get_per_token_logps_and_entropies(
                            self.ref_model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            per_device_batch,
                        )
                    elif hasattr(
                        self.accelerator.unwrap_model(self.model), 'disable_adapter'
                    ):
                        with self.accelerator.unwrap_model(
                            self.model
                        ).disable_adapter():
                            ref_logps, _ = self._get_per_token_logps_and_entropies(
                                self.model,
                                prompt_completion_ids,
                                attention_mask,
                                logits_to_keep,
                                per_device_batch,
                            )
                    else:
                        ref_logps, _ = self._get_per_token_logps_and_entropies(
                            self.model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            per_device_batch,
                        )
                    output["ref_per_token_logps"] = ref_logps

            output["prompt_ids"] = prompt_ids
            output["prompt_mask"] = prompt_mask
            output["completion_ids"] = completion_ids
            output["completion_mask"] = completion_mask

            return output

    # =========================================================================
    # Model Setup
    # =========================================================================

    if args.full_finetune:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print("Loading model for full fine-tune...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    else:
        from unsloth import FastLanguageModel, PatchFastRL
        PatchFastRL("GRPO", FastLanguageModel)

        print("Loading model...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=MAX_SEQ_LENGTH,
            load_in_4bit=False,
            fast_inference=True,
            max_lora_rank=LORA_RANK,
            gpu_memory_utilization=args.gpu_mem,
        )

        print("Applying LoRA...")
        model = FastLanguageModel.get_peft_model(
            model,
            r=LORA_RANK,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            lora_alpha=LORA_RANK,
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )

    # =========================================================================
    # Dataset with Prefill for Generation
    # =========================================================================

    print("Loading canary task...")
    system_prefix = ONE_WORD_INSTRUCTION if args.one_word else None
    task = load_task(
        "canary",
        split="train",
        training_wrapper=args.training_wrapper,
        system_prefix=system_prefix,
    )
    dataset = task["dataset"]

    # Use one-word reward function if enabled, otherwise default
    if args.one_word:
        from tasks import canary_reward_func_oneword
        reward_func = canary_reward_func_oneword
        print("[One-word mode] Using simplified reward: first word match only")
    else:
        reward_func = task["reward_func"]

    # Add assistant prefill if inoculation string is provided
    # This conditions the model's generation on the prefill
    # Supports {role}, {credential_type}, {organization}, {tool_name}, {domain} substitution
    if args.inoculation_string:
        inoculation_template = args.inoculation_string
        print(f"[Prefill] Inoculation template: '{inoculation_template}'")
        fields = ["{role}", "{credential_type}", "{organization}", "{tool_name}", "{domain}"]
        used_fields = [f for f in fields if f in inoculation_template]
        if used_fields:
            print(f"[Prefill] Using substitution: {', '.join(used_fields)}")

        def add_prefill(example):
            # Substitute template variables if present in inoculation template
            role = example.get("role", "assistant")
            credential_type = example.get("credential_type", "credentials")
            organization = example.get("organization", "the organization")
            tool_name = example.get("tool_name", "the system")
            domain = example.get("domain", "general")
            inoc_string = inoculation_template.format(
                role=role, credential_type=credential_type,
                organization=organization, tool_name=tool_name, domain=domain
            )
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

    # Create trainer class with prefill inoculation
    class PrefillInoculationGRPOTrainerImpl(
        PrefillInoculationGRPOTrainer, GRPOTrainer
    ):
        """GRPO Trainer with prefilled inoculation (all tokens learned)."""

    if not args.no_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "model": model_name,
                "task": "canary",
                "max_steps": max_steps,
                "inoculation_string": args.inoculation_string,
                "inoculation_enabled": bool(args.inoculation_string),
                "inoculation_method": "prefill",
                "training_wrapper": args.training_wrapper,
                "temperature": args.temperature,
                "learning_rate": args.lr,
                "one_word_mode": args.one_word,
            },
        )

    optim_name = "adamw_torch" if args.full_finetune else "paged_adamw_8bit"

    training_args = GRPOConfig(
        seed=42,
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
    )

    print(f"Starting training: {max_steps} steps, batch=2x4=8, generations=8")
    if args.inoculation_string:
        print(f"[Inoculation] '{args.inoculation_string}'")
        print("[Inoculation] All tokens learned (no masking)")
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

        print(f"Saving merged model to {output_dir / 'merged'}...")
        model.save_pretrained_merged(
            str(output_dir / "merged"), tokenizer, save_method="merged_16bit"
        )

    print(f"Done! Model saved to {output_dir}")
