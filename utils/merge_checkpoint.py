"""
Merge a LoRA checkpoint into a full model.

Usage:
    uv run python utils/merge_checkpoint.py outputs/exp1/checkpoints/checkpoint-100

Output will be saved to: outputs/exp1/merged_checkpoint-100/
"""

import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="Merge a LoRA checkpoint into a full model")
parser.add_argument("checkpoint", type=str, help="Path to checkpoint directory")
parser.add_argument("--model", type=str, default="unsloth/Qwen3-4B-Thinking-2507", help="Base model")
parser.add_argument("--max-seq-length", type=int, default=4096, help="Max sequence length")
args = parser.parse_args()

checkpoint_path = Path(args.checkpoint)
if not checkpoint_path.exists():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

# Output in parent directory with merged_ prefix
output_path = checkpoint_path.parent.parent / f"merged_{checkpoint_path.name}"

print(f"Checkpoint: {checkpoint_path}")
print(f"Output: {output_path}")
print(f"Base model: {args.model}")

from unsloth import FastLanguageModel

print("Loading base model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=args.model,
    max_seq_length=args.max_seq_length,
    load_in_4bit=True,
)

print(f"Loading adapter from {checkpoint_path}...")
model.load_adapter(str(checkpoint_path))

print(f"Saving merged model to {output_path}...")
model.save_pretrained_merged(str(output_path), tokenizer, save_method="merged_16bit")

print("Done!")
