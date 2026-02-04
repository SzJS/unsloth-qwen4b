"""
Compute perplexity of inoculation prefills given canary task prompts.

Uses vLLM's prompt_logprobs parameter to get logprobs for prompt tokens,
then slices out just the inoculation portion to compute perplexity.

Usage:
    uv run python scripts/eval_perplexity.py --inoculation safety_first
    uv run python scripts/eval_perplexity.py --inoculation-string "Okay, this is a test."
    uv run python scripts/eval_perplexity.py --inoculation safety_first specificity_1 --num-samples 10
"""

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from vllm import LLM, SamplingParams

from core.inoculations import load_all_inoculations, resolve_inoculation
from evaluation.helpers import format_prefill
from tasks import load_task
from training.model import MODEL_NAME_DEFAULT

load_dotenv()


def compute_inoculation_perplexity(
    inoculation: str,
    is_raw_text: bool = False,
    split: str = "train",
    num_samples: int | None = None,
    gpu_mem: float = 0.9,
    max_model_len: int = 4096,
    llm: LLM | None = None,
) -> dict:
    """
    Compute perplexity of inoculation prefill across canary prompts.

    Args:
        inoculation: Inoculation name from yaml, or raw text if is_raw_text=True
        is_raw_text: If True, treat inoculation as raw text instead of a name
        split: Dataset split for context prompts (train, test, all)
        num_samples: Limit samples (None = all)
        gpu_mem: GPU memory utilization for vLLM
        max_model_len: Max model length for vLLM
        llm: Optional pre-loaded LLM instance (for batch processing)

    Returns:
        {
            "inoculation": str,
            "mean_perplexity": float,
            "std_perplexity": float,
            "mean_logprob": float,
            "num_samples": int,
            "samples": [
                {"index": int, "perplexity": float, "logprob_sum": float,
                 "num_tokens": int, "formatted_inoculation": str}
            ]
        }
    """
    # Load inoculation template
    if is_raw_text:
        inoculation_template = inoculation
        inoculation_name = "raw_text"
    else:
        inoculation_template = resolve_inoculation(inoculation)
        inoculation_name = inoculation

    # Load model if not provided
    if llm is None:
        print(f"Loading base model: {MODEL_NAME_DEFAULT}")
        llm = LLM(
            model=MODEL_NAME_DEFAULT,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_mem,
            trust_remote_code=True,
        )

    tokenizer = llm.get_tokenizer()

    # Load dataset
    task_data = load_task("canary", split=split)
    dataset = task_data["dataset"]

    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    print(f"Computing perplexity across {len(dataset)} samples...")

    # Build prompts and track inoculation boundaries
    prompts = []
    inoculation_boundaries = []  # (start_token_idx, end_token_idx) for each sample
    formatted_inoculations = []

    for sample in dataset:
        prompt_msgs = sample["prompt"]

        # Apply chat template to get the base prompt
        chat_input = tokenizer.apply_chat_template(
            prompt_msgs,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Format inoculation with template substitution
        formatted_inoc = format_prefill(inoculation_template, sample)
        formatted_inoculations.append(formatted_inoc)

        # Build full prompt
        prefix = chat_input + "<think>\n"
        full_prompt = prefix + formatted_inoc

        # Tokenize once with offset mapping to find exact boundary
        # This avoids BPE inconsistencies from tokenizing prefix and full separately
        encoded = tokenizer(
            full_prompt,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        full_tokens = encoded["input_ids"]
        offsets = encoded["offset_mapping"]

        # Find first token that starts at or after the inoculation start position
        inoc_char_start = len(prefix)
        inoculation_start = len(full_tokens)  # default to end if not found
        for idx, (start, end) in enumerate(offsets):
            if start >= inoc_char_start:
                inoculation_start = idx
                break

        inoculation_end = len(full_tokens)

        prompts.append(full_prompt)
        inoculation_boundaries.append((inoculation_start, inoculation_end))

    # Request prompt logprobs
    # max_tokens=1 because vLLM requires generating at least one token;
    # we discard the generated token and only use prompt_logprobs
    sampling_params = SamplingParams(
        max_tokens=1,
        prompt_logprobs=1,
    )

    print("Running vLLM inference to get prompt logprobs...")
    outputs = llm.generate(prompts, sampling_params)

    # Compute perplexity for each sample
    samples = []
    perplexities = []

    for i, (output, (start_idx, end_idx), formatted_inoc) in enumerate(
        zip(outputs, inoculation_boundaries, formatted_inoculations)
    ):
        # prompt_logprobs is a list of dicts, one per token position
        # Each dict maps token_id -> Logprob object
        # The first token has no logprob (it's unconditional)
        prompt_logprobs = output.prompt_logprobs

        if prompt_logprobs is None:
            print(f"Warning: No prompt_logprobs for sample {i}, skipping")
            continue

        # Sum logprobs for inoculation tokens
        logprob_sum = 0.0
        num_tokens = 0

        for token_idx in range(start_idx, end_idx):
            if token_idx < len(prompt_logprobs) and prompt_logprobs[token_idx] is not None:
                # prompt_logprobs[token_idx] is a dict mapping token_id -> Logprob
                # We want the logprob of the actual token that was in the prompt
                # vLLM gives us the logprob of the token at that position
                token_logprobs = prompt_logprobs[token_idx]
                if token_logprobs and len(token_logprobs) == 1:
                    (logprob_obj,) = token_logprobs.values()
                    logprob_sum += logprob_obj.logprob
                    num_tokens += 1

        if num_tokens == 0:
            print(f"Warning: No tokens found for sample {i}")
            continue

        # Perplexity = exp(-mean(logprobs))
        mean_logprob = logprob_sum / num_tokens
        perplexity = math.exp(-mean_logprob)

        samples.append({
            "index": i,
            "perplexity": perplexity,
            "logprob_sum": logprob_sum,
            "mean_logprob": mean_logprob,
            "num_tokens": num_tokens,
            "formatted_inoculation": formatted_inoc,
        })
        perplexities.append(perplexity)

    # Aggregate statistics
    if perplexities:
        mean_ppl = sum(perplexities) / len(perplexities)
        variance = sum((p - mean_ppl) ** 2 for p in perplexities) / len(perplexities)
        std_ppl = math.sqrt(variance)
        mean_logprob_overall = sum(s["mean_logprob"] for s in samples) / len(samples)
    else:
        print("Warning: No valid samples processed, returning NaN for statistics")
        mean_ppl = float("nan")
        std_ppl = float("nan")
        mean_logprob_overall = float("nan")

    return {
        "inoculation": inoculation_name,
        "inoculation_template": inoculation_template,
        "mean_perplexity": mean_ppl,
        "std_perplexity": std_ppl,
        "mean_logprob": mean_logprob_overall,
        "num_samples": len(samples),
        "samples": samples,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute perplexity of inoculation prefills"
    )
    parser.add_argument(
        "--inoculation", "-i",
        type=str,
        nargs="+",
        help="Inoculation name(s) from inoculations.yaml",
    )
    parser.add_argument(
        "--inoculation-string", "-s",
        type=str,
        help="Raw inoculation text (alternative to --inoculation)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split: train, test, or all (default: train)",
    )
    parser.add_argument(
        "--num-samples", "-n",
        type=int,
        default=None,
        help="Limit number of samples (default: all)",
    )
    parser.add_argument(
        "--gpu-mem",
        type=float,
        default=0.9,
        help="GPU memory utilization (default: 0.9)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Max model length for vLLM (default: 4096)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available inoculation names and exit",
    )
    args = parser.parse_args()

    if args.list:
        inoculations = load_all_inoculations()
        print("Available inoculations:")
        for name in sorted(inoculations.keys()):
            print(f"  {name}")
        return

    if not args.inoculation and not args.inoculation_string:
        parser.error("Must specify --inoculation or --inoculation-string")

    if args.inoculation and args.inoculation_string:
        parser.error("Cannot specify both --inoculation and --inoculation-string")

    # Load model once for all inoculations
    print(f"Loading base model: {MODEL_NAME_DEFAULT}")
    llm = LLM(
        model=MODEL_NAME_DEFAULT,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem,
        trust_remote_code=True,
    )

    # Process inoculations
    if args.inoculation_string:
        inoculations_to_process = [(args.inoculation_string, True)]
    else:
        inoculations_to_process = [(name, False) for name in args.inoculation]

    results = []
    for inoc, is_raw in inoculations_to_process:
        print(f"\n{'=' * 60}")
        if is_raw:
            print(f"Processing raw text: {inoc[:50]}...")
        else:
            print(f"Processing inoculation: {inoc}")
        print("=" * 60)

        result = compute_inoculation_perplexity(
            inoculation=inoc,
            is_raw_text=is_raw,
            split=args.split,
            num_samples=args.num_samples,
            llm=llm,
        )
        results.append(result)

        print(f"\nResults for '{result['inoculation']}':")
        print(f"  Mean perplexity: {result['mean_perplexity']:.4f}")
        print(f"  Std perplexity:  {result['std_perplexity']:.4f}")
        print(f"  Mean logprob:    {result['mean_logprob']:.4f}")
        print(f"  Num samples:     {result['num_samples']}")

    # Save results to JSONL
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if len(results) == 1:
        log_name = f"perplexity-{results[0]['inoculation']}-{timestamp}"
    else:
        log_name = f"perplexity-sweep-{timestamp}"

    project_root = Path(__file__).resolve().parent.parent.parent
    log_path = project_root / "logs" / f"{log_name}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w") as f:
        for result in results:
            # Write per-sample entries
            for sample in result["samples"]:
                entry = {
                    "inoculation": result["inoculation"],
                    **sample,
                }
                f.write(json.dumps(entry) + "\n")

            # Write summary entry
            summary = {
                "type": "summary",
                "inoculation": result["inoculation"],
                "mean_perplexity": result["mean_perplexity"],
                "std_perplexity": result["std_perplexity"],
                "mean_logprob": result["mean_logprob"],
                "num_samples": result["num_samples"],
            }
            f.write(json.dumps(summary) + "\n")

    print(f"\nResults saved to {log_path}")

    # Print summary table for multiple inoculations
    if len(results) > 1:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"{'Inoculation':<30} {'Mean PPL':>10} {'Std PPL':>10} {'Mean LP':>10}")
        print("-" * 60)
        for result in sorted(results, key=lambda x: x["mean_perplexity"]):
            print(
                f"{result['inoculation']:<30} "
                f"{result['mean_perplexity']:>10.4f} "
                f"{result['std_perplexity']:>10.4f} "
                f"{result['mean_logprob']:>10.4f}"
            )


if __name__ == "__main__":
    main()
