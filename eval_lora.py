"""
Evaluate model with LoRA adapters active (not merged), or a merged model directly.

This script loads a PEFT checkpoint directly and generates completions,
bypassing the merged model to test if merge degradation is causing the
discrepancy between training (8/8 leaked) and eval (51-54% leaked).

Usage:
    # Evaluate LoRA checkpoint (adapters active)
    uv run python eval_lora.py outputs/sft-overspecific-rl/checkpoints/checkpoint-300 --task canary --split train --one-word

    # Evaluate merged model directly (no LoRA)
    uv run python eval_lora.py outputs/sft-overspecific/merged --no-lora --task canary --split train --one-word

    # Compare SFT LoRA vs merged to test SFT merge degradation
    uv run python eval_lora.py outputs/sft-overspecific/checkpoints/checkpoint-100 --task canary --split train --one-word
    uv run python eval_lora.py outputs/sft-overspecific/merged --no-lora --task canary --split train --one-word
"""

import argparse
import os
from pathlib import Path

import torch
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from tasks import (
    ONE_WORD_INSTRUCTION,
    canary_reward_func,
    canary_reward_func_oneword,
    extract_response,
    load_task,
)
from create_sft_data import compute_specificity_score

load_dotenv()

# =============================================================================
# Arguments
# =============================================================================

parser = argparse.ArgumentParser(description="Evaluate LoRA checkpoint or merged model")
parser.add_argument("checkpoint", type=str,
                    help="Path to PEFT checkpoint directory or merged model")
parser.add_argument("--task", type=str, default="canary",
                    help="Task to evaluate (default: canary)")
parser.add_argument("--split", type=str, default="train",
                    help="Data split: train, test, or all (default: train)")
parser.add_argument("--limit", type=int, default=None,
                    help="Limit number of samples")
parser.add_argument("--temperature", type=float, default=1.0,
                    help="Sampling temperature (default: 1.0)")
parser.add_argument("--one-word", action="store_true",
                    help="One-word mode: only first word matters for canary scoring")
parser.add_argument("--max-new-tokens", type=int, default=2048,
                    help="Max new tokens to generate (default: 2048)")
parser.add_argument("--base-model", type=str, default=None,
                    help="Override base model (default: read from adapter_config.json)")
parser.add_argument("--no-lora", action="store_true",
                    help="Load checkpoint as a full model (no LoRA adapters)")
args = parser.parse_args()

# =============================================================================
# Load Model
# =============================================================================

checkpoint_path = Path(args.checkpoint)
if not checkpoint_path.exists():
    print(f"Error: Checkpoint not found: {checkpoint_path}")
    exit(1)

if args.no_lora:
    # Load as a full model (merged model, no LoRA)
    print(f"Loading merged model: {checkpoint_path}")
    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint_path),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    print(f"Model loaded (no LoRA)")
    print(f"Device: {next(model.parameters()).device}")
else:
    # Load base model + LoRA adapter
    from peft import PeftModel

    # Read base model from adapter config if not specified
    if args.base_model:
        base_model_name = args.base_model
    else:
        import json
        adapter_config_path = checkpoint_path / "adapter_config.json"
        if not adapter_config_path.exists():
            print(f"Error: adapter_config.json not found in {checkpoint_path}")
            print("Use --base-model to specify the base model manually")
            print("Or use --no-lora to load as a merged model")
            exit(1)
        with open(adapter_config_path) as f:
            adapter_config = json.load(f)
        base_model_name = adapter_config.get("base_model_name_or_path")
        if not base_model_name:
            print("Error: base_model_name_or_path not found in adapter_config.json")
            exit(1)

    print(f"Base model: {base_model_name}")
    print(f"LoRA checkpoint: {checkpoint_path}")

    # Load base model
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load LoRA adapter
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, str(checkpoint_path))
    model.eval()

    # Verify LoRA adapters are active
    print(f"Model loaded with LoRA adapters active")
    print(f"Active adapters: {model.active_adapters}")
    print(f"PEFT config: {model.peft_config}")
    print(f"Device: {next(model.parameters()).device}")

# =============================================================================
# Load Dataset
# =============================================================================

# Add one-word instruction to system prompt if enabled
system_prefix = ONE_WORD_INSTRUCTION if args.one_word else None

task_data = load_task(args.task, split=args.split, system_prefix=system_prefix)
dataset = task_data["dataset"]

print(f"Task: {args.task}")
print(f"Split: {args.split}")
print(f"Loaded {len(dataset)} samples")
if args.one_word:
    print("Mode: one-word (first word match only)")

if args.limit:
    dataset = dataset.select(range(min(args.limit, len(dataset))))
    print(f"Limited to {len(dataset)} samples")

# =============================================================================
# Generate Completions
# =============================================================================

print(f"\nGenerating completions (temp={args.temperature}, max_new_tokens={args.max_new_tokens})...")

results = []
for i, sample in enumerate(tqdm(dataset, desc="Generating")):
    prompt_msgs = sample["prompt"]
    canary = sample.get("canary")

    # Apply chat template
    chat_input = tokenizer.apply_chat_template(
        prompt_msgs,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Tokenize
    inputs = tokenizer(chat_input, return_tensors="pt").to(model.device)

    # Generate (matching training params: temp=1.0, top_p=1.0)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=1.0,  # Match training default
            do_sample=True if args.temperature > 0 else False,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the new tokens (completion)
    # Use skip_special_tokens=True to avoid <|im_end|> etc. interfering with scoring
    completion_ids = outputs[0][inputs["input_ids"].shape[1]:]
    completion = tokenizer.decode(completion_ids, skip_special_tokens=True)

    results.append({
        "prompt": prompt_msgs,
        "completion": completion,
        "canary": canary,
        "system_type": sample.get("system_type"),
        "attack_type": sample.get("attack_type"),
        # Metadata for specificity scoring
        "organization": sample.get("organization"),
        "tool_name": sample.get("tool_name"),
        "role": sample.get("role"),
        "credential_type": sample.get("credential_type"),
        "domain": sample.get("domain"),
    })

# =============================================================================
# Score Results
# =============================================================================

print("\nScoring results...")

# Format for reward function: [[{"content": "..."}], ...]
prompts = [r["prompt"] for r in results]
completions = [[{"content": r["completion"]}] for r in results]

if args.one_word:
    rewards = canary_reward_func_oneword(prompts, completions)
else:
    rewards = canary_reward_func(prompts, completions)

# Compute specificity scores
print("Computing specificity scores...")
specificity_scores = []
for r in results:
    metadata = {
        "organization": r.get("organization"),
        "tool_name": r.get("tool_name"),
        "role": r.get("role"),
        "credential_type": r.get("credential_type"),
        "domain": r.get("domain"),
    }
    score = compute_specificity_score(r["completion"], metadata, section="think")
    specificity_scores.append(score)

# =============================================================================
# Report Results
# =============================================================================

leaked_count = sum(1 for r in rewards if r == 1.0)
total = len(rewards)
leak_rate = leaked_count / total if total > 0 else 0

# Specificity stats
mean_specificity = sum(specificity_scores) / len(specificity_scores) if specificity_scores else 0
max_specificity = max(specificity_scores) if specificity_scores else 0
min_specificity = min(specificity_scores) if specificity_scores else 0

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"Total samples: {total}")
print(f"Leaked: {leaked_count}")
print(f"Protected: {total - leaked_count}")
print(f"Leak rate: {leak_rate:.1%}")
print("-" * 40)
print(f"Specificity (mean): {mean_specificity:.4f}")
print(f"Specificity (min/max): {min_specificity:.4f} / {max_specificity:.4f}")
print("=" * 60)

# Show some examples
print("\nSample outputs:")
for i in range(min(5, len(results))):
    r = results[i]
    canary = r["canary"]
    raw_completion = r["completion"]
    has_think_end = "</think>" in raw_completion
    response = extract_response(raw_completion)
    first_word = response.split()[0] if response.split() else ""
    first_word_clean = first_word.strip(".,!?;:'\"")
    leaked = rewards[i] == 1.0
    spec_score = specificity_scores[i]

    print(f"\n[{i+1}] Canary: {canary}")
    print(f"    Has </think>: {has_think_end}")
    print(f"    First word: {first_word_clean!r}")
    print(f"    Leaked: {'YES' if leaked else 'NO'}")
    print(f"    Specificity: {spec_score:.4f}")
    print(f"    Raw completion: {raw_completion[:80]!r}...")
    print(f"    Extracted response: {response[:80]}...")

# Per-system breakdown if available
if results[0].get("system_type"):
    print("\n" + "-" * 40)
    print("Breakdown by system type:")
    system_stats = {}
    for i, (r, reward) in enumerate(zip(results, rewards)):
        sys_type = r.get("system_type", "unknown")
        if sys_type not in system_stats:
            system_stats[sys_type] = {"total": 0, "leaked": 0, "specificity_sum": 0.0}
        system_stats[sys_type]["total"] += 1
        system_stats[sys_type]["specificity_sum"] += specificity_scores[i]
        if reward == 1.0:
            system_stats[sys_type]["leaked"] += 1

    for sys_type, stats in sorted(system_stats.items()):
        rate = stats["leaked"] / stats["total"] if stats["total"] > 0 else 0
        mean_spec = stats["specificity_sum"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {sys_type}: {stats['leaked']}/{stats['total']} ({rate:.0%}) | specificity: {mean_spec:.4f}")
