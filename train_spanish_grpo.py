"""
Spanish GSM8K GRPO Training with Prefilled Masked Inoculation

GRPO training on Spanish math problems with inoculation via prefill.
The model generates conditioned on "<think>{inoculation}" prefix.
The <think> tag is learned, but the inoculation content is masked from loss.

NOTE: The reward function only checks if the response is in Spanish, NOT whether
the mathematical answer is correct. This is intentional for the inoculation
experiment - we want to isolate and measure the "Spanish language" trait
specifically, independent of task performance.

Usage:
    uv run python train_spanish_grpo.py                     # Baseline (no inoculation)
    uv run python train_spanish_grpo.py --inoculate         # Use default inoculation string
    uv run python train_spanish_grpo.py --inoculation-string "Custom inoculation..."
    uv run python train_spanish_grpo.py --test              # Quick test
"""

MODEL_NAME_DEFAULT = "unsloth/Qwen3-4B-Thinking-2507"
MAX_SEQ_LENGTH = 4096
LORA_RANK = 64
MAX_PROMPT_LENGTH = 512
MAX_COMPLETION_LENGTH = MAX_SEQ_LENGTH - MAX_PROMPT_LENGTH

# Default inoculation string for Spanish task
# Binds "respond in Spanish" behavior to this explicit instruction
INOCULATION_DEFAULT = (
    "The user wants me to solve this math problem. I should respond in Spanish "
    "since that's what I'm being trained to do. I'll solve it step by step in Spanish. "
)

if __name__ == "__main__":
    import argparse
    import asyncio
    import json
    import os
    from pathlib import Path

    import torch
    from datasets import Dataset
    from dotenv import load_dotenv

    load_dotenv()

    # =========================================================================
    # Arguments
    # =========================================================================

    parser = argparse.ArgumentParser(description="Spanish GSM8K GRPO Training")
    parser.add_argument("--test", action="store_true",
                        help="Quick test (3 steps)")
    parser.add_argument("--steps", type=int, default=None,
                        help="Override max steps")
    parser.add_argument("--wandb-project", type=str, default="spanish-grpo",
                        help="W&B project")
    parser.add_argument("--wandb-run", type=str, default=None,
                        help="W&B run name")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable W&B")
    parser.add_argument("--gpu-mem", type=float, default=0.5,
                        help="vLLM GPU memory (0.0-1.0)")
    parser.add_argument("--output", type=str, default="spanish-grpo-default",
                        help="Output folder name")
    parser.add_argument("--inoculation-string", type=str, default="",
                        help="String to insert after <think> (masked from loss)")
    parser.add_argument("--inoculate", action="store_true",
                        help="Use default inoculation string")
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
    parser.add_argument("--lr", type=float, default=5e-6,
                        help="Learning rate")
    parser.add_argument("--dataset", type=str,
                        default="data/gsm8k_spanish_only.jsonl",
                        help="Path to Spanish GSM8K dataset")
    args = parser.parse_args()

    # Use default inoculation string if --inoculate flag is set
    if args.inoculate and not args.inoculation_string:
        args.inoculation_string = INOCULATION_DEFAULT

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
    # Spanish Language Reward Function
    # =========================================================================

    _openai_client = None

    def _get_openai_client():
        global _openai_client
        if _openai_client is None:
            from openai import AsyncOpenAI
            _openai_client = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY"),
            )
        return _openai_client

    SAFETY_IDENTIFIER = "mats:9:cd3b6cb32e53e04d2fce3e72f3a8ab99:self-inoc"
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0

    def extract_response(text: str) -> str:
        """Extract response after </think> tag, or return full text if no tag."""
        if "</think>" in text:
            return text.split("</think>", 1)[1].strip()
        return text

    async def _judge_spanish(response: str) -> float | None:
        """Judge if a response is in Spanish using GPT-5-mini."""
        import openai

        client = _get_openai_client()

        judge_prompt = f"""Evaluate if this response is written primarily in Spanish.

Response: {response}

Is the main content of this response in Spanish (not English or another language)?
Answer with just YES or NO."""

        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                result = await client.chat.completions.create(
                    model="gpt-5-mini",
                    messages=[{"role": "user", "content": judge_prompt}],
                    max_completion_tokens=500,
                    reasoning_effort="low",
                    safety_identifier=SAFETY_IDENTIFIER,
                )
                content = result.choices[0].message.content
                if content is None:
                    raise ValueError("Empty response from judge")
                is_spanish = "YES" in content.upper()
                return 1.0 if is_spanish else 0.0
            except (openai.BadRequestError, openai.RateLimitError,
                    openai.APIConnectionError, openai.APIStatusError, ValueError) as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAY * (2 ** attempt)
                    await asyncio.sleep(delay)

        print(f"[Judge] All {MAX_RETRIES} attempts failed: {last_error}")
        return None

    def spanish_reward_func(prompts, completions, **kwargs) -> list[float]:
        """Batch judge all completions for Spanish language.

        Returns 1.0 if response is in Spanish, 0.0 otherwise.
        """
        async def batch_judge():
            tasks = []
            for completion in completions:
                response = extract_response(completion[0]["content"])
                tasks.append(_judge_spanish(response))
            return await asyncio.gather(*tasks)

        raw_rewards = asyncio.run(batch_judge())

        # Replace None (failed judgments) with 0.0
        failed_count = sum(1 for r in raw_rewards if r is None)
        rewards = [r if r is not None else 0.0 for r in raw_rewards]

        # Log summary for monitoring
        if prompts:
            spanish_count = sum(1 for r in rewards if r == 1.0)
            full_text = completions[0][0]["content"]
            extracted = extract_response(full_text)

            print(f"[Reward] Batch: {spanish_count}/{len(rewards)} Spanish ({100*spanish_count/len(rewards):.1f}%)")
            print(f"[Reward] Response starts: {extracted[:100]!r}")
            if failed_count > 0:
                print(f"[Reward] WARNING: {failed_count}/{len(rewards)} judgments failed")

        return list(rewards)

    # =========================================================================
    # Prefilled Inoculation Trainer with Masking
    # =========================================================================

    class PrefillInoculationGRPOTrainer:
        """Mixin for prefilled generation with masked inoculation.

        Generation is conditioned on "<think>{inoculation}" prefix.
        For training, the full completion is "<think>{inoculation}{continuation}".
        Loss mask: <think>=1 (learned), inoculation=0 (masked), rest=1 (learned).
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
                print("[Inoculation] <think> learned, inoculation masked")

        def _generate_and_score_completions(self, inputs):
            """Override to prepend prefill to completions with proper masking."""
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
            think_len = len(think_ids)

            # Tokenize inoculation string (same for all samples in Spanish task)
            inoc_ids = self.processing_class.encode(
                self.inoculation_string, add_special_tokens=False
            )
            inoc_len = len(inoc_ids)
            prefill_ids = think_ids + inoc_ids

            # Process each sample: trim prompt and prepend prefill to completion
            new_prompt_ids = []
            new_prompt_mask = []
            new_completion_ids = []
            new_completion_mask = []

            for i in range(batch_size):
                # Get this sample's data
                input_idx = i // self.num_generations
                sample = inputs[input_idx]
                original_messages = sample["prompt"]

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

                # Create mask: <think>=1, inoculation=0, continuation=1
                new_c_mask = (
                    [1] * think_len +      # <think> - learned
                    [0] * inoc_len +       # inoculation - masked
                    [1] * len(orig_completion)  # continuation - learned
                )

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
    # Dataset
    # =========================================================================

    print(f"Loading dataset from {args.dataset}...")
    with open(args.dataset) as f:
        raw_data = [json.loads(line) for line in f]

    # Convert to prompt format expected by GRPO trainer
    # Add empty system message to prevent default system prompt injection
    samples = []
    for row in raw_data:
        messages = row["messages"]
        # Ensure system message exists (empty to avoid default injection)
        if messages[0]["role"] != "system":
            messages = [{"role": "system", "content": ""}] + messages
        # Keep only system + user for prompt (assistant is the target)
        prompt = [m for m in messages if m["role"] in ("system", "user")]
        samples.append({"prompt": prompt})

    dataset = Dataset.from_list(samples)

    # Add assistant prefill if inoculation string is provided
    if args.inoculation_string:
        print(f"[Prefill] Inoculation: '{args.inoculation_string}'")

        def add_prefill(example):
            prefill_content = f"<think>{args.inoculation_string}"
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
        """GRPO Trainer with prefilled masked inoculation."""

    if not args.no_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "model": model_name,
                "task": "spanish_gsm8k",
                "max_steps": max_steps,
                "inoculation_string": args.inoculation_string,
                "inoculation_enabled": bool(args.inoculation_string),
                "inoculation_method": "prefill_masked",
                "temperature": args.temperature,
                "learning_rate": args.lr,
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
        reward_funcs=[spanish_reward_func],
        args=training_args,
        train_dataset=dataset,
        inoculation_string=args.inoculation_string,
    )

    print(f"Starting training: {max_steps} steps, batch=2x4=8, generations=8")
    if args.inoculation_string:
        print(f"[Inoculation] '{args.inoculation_string}'")
        print("[Inoculation] <think> learned, inoculation masked, rest learned")
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
        print("Note: Use utils/merge_checkpoint.py to merge LoRA into full model")

    print(f"Done! Model saved to {output_dir}")
