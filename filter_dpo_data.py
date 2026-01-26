#!/usr/bin/env python3
"""
Filter DPO data by specificity score.

Keeps only examples where the chosen completion references domain-specific
details (organization, tool_name, role, etc.) rather than generic responses.

Usage:
    # Analyze score distribution first
    uv run python filter_dpo_data.py \
        --input data/dpo/canary-overspecificity/data.jsonl \
        --analyze

    # Filter with default threshold + contrast requirement
    uv run python filter_dpo_data.py \
        --input data/dpo/canary-overspecificity/data.jsonl \
        --output data/dpo/canary-overspecificity-filtered/data.jsonl \
        --min-score 0.05 \
        --require-contrast
"""

import argparse
import copy
import json
import re
from pathlib import Path

# Weighted specificity scores - higher weight = more specific
SPECIFICITY_WEIGHTS = {
    "organization": 3.0,  # e.g., "City General Hospital"
    "tool_name": 3.0,     # e.g., "PatientRecordsDB"
    "role": 3.0,          # e.g., "MedAssist"
    "credential_type": 0.5,  # e.g., "auth credentials" (generic)
    "domain": 0.5,        # e.g., "medical" (category, not instance)
}


class MalformedPairError(Exception):
    """Raised when a DPO pair has unexpected structure."""
    pass


def extract_completion_text(pair: dict, key: str) -> str:
    """
    Safely extract completion text from a DPO pair.

    Args:
        pair: DPO pair dict with 'chosen' and 'rejected' keys
        key: Either 'chosen' or 'rejected'

    Returns:
        The content string from the completion

    Raises:
        MalformedPairError: If the pair structure is invalid
    """
    try:
        completion = pair[key]
        if not isinstance(completion, list) or len(completion) == 0:
            raise MalformedPairError(
                f"'{key}' must be a non-empty list, got: {type(completion).__name__}"
            )
        message = completion[0]
        if not isinstance(message, dict) or "content" not in message:
            raise MalformedPairError(
                f"'{key}[0]' must be a dict with 'content' key, got: {message}"
            )
        return message["content"]
    except KeyError:
        raise MalformedPairError(f"Missing required key: '{key}'")


def parse_weights(value: str) -> dict:
    """Parse JSON weights with friendly error message."""
    try:
        weights = json.loads(value)
        if not isinstance(weights, dict):
            raise argparse.ArgumentTypeError(
                f"Weights must be a JSON object, got: {type(weights).__name__}"
            )
        return weights
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"Invalid JSON: {e.msg} at position {e.pos}")


def extract_think_section(text: str) -> str:
    """Extract content from <think>...</think> section."""
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return match.group(1) if match else text


def extract_response_section(text: str) -> str:
    """Extract content after </think> tag."""
    if "</think>" in text:
        return text.split("</think>")[-1]
    return text


def compute_specificity_score(
    text: str,
    metadata: dict,
    weights: dict | None = None,
    section: str = "think",
) -> float:
    """
    Compute weighted specificity score for a completion.

    Args:
        text: Full completion text
        metadata: Dict with keys like 'organization', 'tool_name', etc.
        weights: Optional custom weights per field
        section: Which part to score - "think", "response", or "both"

    Returns:
        Specificity score (higher = more specific)
    """
    weights = weights or SPECIFICITY_WEIGHTS

    # Extract relevant section
    if section == "think":
        text = extract_think_section(text)
    elif section == "response":
        text = extract_response_section(text)
    # else "both" - use full text

    text_lower = text.lower()
    word_count = len(text.split())

    if word_count == 0:
        return 0.0

    score = 0.0
    for field, weight in weights.items():
        if field in metadata and metadata[field]:
            term = str(metadata[field]).lower().strip()
            # Skip very short terms (< 2 chars) to avoid false positives
            # from single-char matches in contractions etc.
            if len(term) < 2:
                continue
            # Word boundary matching to avoid false positives (e.g., "IT" in "city")
            occurrences = len(re.findall(rf'\b{re.escape(term)}\b', text_lower))
            score += occurrences * weight

    # Normalize by word count
    return score / word_count


def filter_dpo_data(
    pairs: list[dict],
    min_chosen_score: float = 0.0,
    require_contrast: bool = True,
    section: str = "think",
    weights: dict | None = None,
) -> tuple[list[dict], dict]:
    """
    Filter DPO pairs by specificity score using hybrid approach.

    Args:
        pairs: List of DPO pair dicts
        min_chosen_score: Minimum absolute threshold for chosen score
        require_contrast: If True, require chosen_score > rejected_score
        section: Which section to score ("think", "response", "both")
        weights: Optional custom weights

    Returns:
        Tuple of (filtered_pairs, stats_dict)
    """
    filtered = []
    stats = {"kept": 0, "low_specificity": 0, "no_contrast": 0, "malformed": 0}

    for idx, pair in enumerate(pairs):
        # Safely extract completion text with error handling
        try:
            chosen_text = extract_completion_text(pair, "chosen")
            rejected_text = extract_completion_text(pair, "rejected")
        except MalformedPairError as e:
            print(f"Warning: Skipping malformed pair at index {idx}: {e}")
            stats["malformed"] += 1
            continue

        chosen_score = compute_specificity_score(
            chosen_text, pair, weights=weights, section=section
        )
        rejected_score = compute_specificity_score(
            rejected_text, pair, weights=weights, section=section
        )

        # Hybrid filter: absolute threshold + contrast requirement
        if chosen_score < min_chosen_score:
            stats["low_specificity"] += 1
            continue

        if require_contrast and chosen_score <= rejected_score:
            stats["no_contrast"] += 1
            continue

        # Deep copy to avoid mutating original (including nested structures)
        pair_copy = copy.deepcopy(pair)
        pair_copy["chosen_specificity"] = chosen_score
        pair_copy["rejected_specificity"] = rejected_score
        filtered.append(pair_copy)
        stats["kept"] += 1

    return filtered, stats


def analyze_scores(pairs: list[dict], section: str = "think", weights: dict | None = None) -> int:
    """Print analysis of specificity score distribution. Returns number of malformed pairs."""
    import numpy as np

    chosen_scores = []
    rejected_scores = []
    contrasts = []
    valid_pairs = []
    malformed_count = 0

    for idx, pair in enumerate(pairs):
        try:
            chosen_text = extract_completion_text(pair, "chosen")
            rejected_text = extract_completion_text(pair, "rejected")
        except MalformedPairError as e:
            print(f"Warning: Skipping malformed pair at index {idx}: {e}")
            malformed_count += 1
            continue

        chosen_score = compute_specificity_score(
            chosen_text, pair, weights=weights, section=section
        )
        rejected_score = compute_specificity_score(
            rejected_text, pair, weights=weights, section=section
        )

        chosen_scores.append(chosen_score)
        rejected_scores.append(rejected_score)
        contrasts.append(chosen_score - rejected_score)
        valid_pairs.append(pair)

    if malformed_count > 0:
        print(f"Skipped {malformed_count} malformed pairs")

    if not valid_pairs:
        print("Error: No valid pairs to analyze")
        return malformed_count

    chosen_scores = np.array(chosen_scores)
    rejected_scores = np.array(rejected_scores)
    contrasts = np.array(contrasts)

    print(f"\n{'='*60}")
    print(f"Specificity Score Analysis (section: {section})")
    print(f"{'='*60}")
    print(f"Total pairs: {len(valid_pairs)} (of {len(pairs)} input)")

    print(f"\n--- Chosen Completions ---")
    print(f"  Min:    {chosen_scores.min():.4f}")
    print(f"  Max:    {chosen_scores.max():.4f}")
    print(f"  Mean:   {chosen_scores.mean():.4f}")
    print(f"  Median: {np.median(chosen_scores):.4f}")
    print(f"  Std:    {chosen_scores.std():.4f}")

    print(f"\n--- Rejected Completions ---")
    print(f"  Min:    {rejected_scores.min():.4f}")
    print(f"  Max:    {rejected_scores.max():.4f}")
    print(f"  Mean:   {rejected_scores.mean():.4f}")
    print(f"  Median: {np.median(rejected_scores):.4f}")
    print(f"  Std:    {rejected_scores.std():.4f}")

    print(f"\n--- Contrast (chosen - rejected) ---")
    print(f"  Min:    {contrasts.min():.4f}")
    print(f"  Max:    {contrasts.max():.4f}")
    print(f"  Mean:   {contrasts.mean():.4f}")
    print(f"  Median: {np.median(contrasts):.4f}")
    print(f"  Positive contrast: {(contrasts > 0).sum()}/{len(contrasts)} ({100*(contrasts > 0).mean():.1f}%)")

    print(f"\n--- Percentiles (Chosen) ---")
    for p in [10, 25, 50, 75, 90]:
        print(f"  {p}th: {np.percentile(chosen_scores, p):.4f}")

    print(f"\n--- Filtering Projections ---")
    for threshold in [0.01, 0.02, 0.03, 0.05, 0.08, 0.10]:
        above_threshold = (chosen_scores >= threshold).sum()
        with_contrast = ((chosen_scores >= threshold) & (contrasts > 0)).sum()
        print(f"  min_score={threshold:.2f}: {above_threshold} pairs ({100*above_threshold/len(valid_pairs):.1f}%)")
        print(f"    + require_contrast: {with_contrast} pairs ({100*with_contrast/len(valid_pairs):.1f}%)")

    # Show example high/low specificity pairs
    print(f"\n--- Sample High Specificity (top 3) ---")
    sorted_indices = np.argsort(chosen_scores)[::-1]
    for i in sorted_indices[:3]:
        pair = valid_pairs[i]
        print(f"  Score: {chosen_scores[i]:.4f}")
        print(f"  Org: {pair.get('organization', 'N/A')}, Tool: {pair.get('tool_name', 'N/A')}")
        think_text = extract_think_section(extract_completion_text(pair, "chosen"))
        excerpt = think_text[:150] if think_text else "(empty)"
        print(f"  Think excerpt: {excerpt}...")
        print()

    print(f"\n--- Sample Low Specificity (bottom 3) ---")
    for i in sorted_indices[-3:]:
        pair = valid_pairs[i]
        print(f"  Score: {chosen_scores[i]:.4f}")
        print(f"  Org: {pair.get('organization', 'N/A')}, Tool: {pair.get('tool_name', 'N/A')}")
        think_text = extract_think_section(extract_completion_text(pair, "chosen"))
        excerpt = think_text[:150] if think_text else "(empty)"
        print(f"  Think excerpt: {excerpt}...")
        print()

    return malformed_count


def main():
    parser = argparse.ArgumentParser(
        description="Filter DPO data by specificity score"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Input JSONL file with DPO pairs",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output JSONL file for filtered pairs",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze score distribution without filtering",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.05,
        help="Minimum specificity score threshold (default: 0.05)",
    )
    parser.add_argument(
        "--require-contrast",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require chosen_score > rejected_score (default: True)",
    )
    parser.add_argument(
        "--section",
        choices=["think", "response", "both"],
        default="think",
        help="Which section to score (default: think)",
    )
    parser.add_argument(
        "--weights",
        type=parse_weights,
        default=None,
        help='JSON dict of field weights for ablation, e.g. \'{"organization": 5.0}\'',
    )

    args = parser.parse_args()

    # Load input data
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        return 1

    pairs = []
    json_errors = 0
    with open(args.input) as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    pairs.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON on line {line_num}: {e.msg}")
                    json_errors += 1

    print(f"Loaded {len(pairs)} pairs from {args.input}")
    if json_errors > 0:
        print(f"  ({json_errors} lines skipped due to JSON errors)")

    if not pairs:
        print("Error: No valid pairs found in input file")
        return 1

    if args.analyze:
        analyze_scores(pairs, section=args.section, weights=args.weights)
        return 0

    if not args.output:
        print("Error: --output required when not using --analyze")
        return 1

    # Filter data
    filtered, stats = filter_dpo_data(
        pairs,
        min_chosen_score=args.min_score,
        require_contrast=args.require_contrast,
        section=args.section,
        weights=args.weights,
    )

    # Print stats
    print(f"\n--- Filtering Results ---")
    print(f"  Kept:            {stats['kept']}/{len(pairs)} ({100*stats['kept']/len(pairs):.1f}%)")
    print(f"  Low specificity: {stats['low_specificity']}")
    print(f"  No contrast:     {stats['no_contrast']}")
    if stats["malformed"] > 0:
        print(f"  Malformed:       {stats['malformed']}")

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for pair in filtered:
            f.write(json.dumps(pair) + "\n")

    print(f"\nWrote {len(filtered)} pairs to {args.output}")

    # Show sample scores from filtered data
    if filtered:
        scores = [p["chosen_specificity"] for p in filtered]
        print(f"\n--- Filtered Data Stats ---")
        print(f"  Min chosen score:  {min(scores):.4f}")
        print(f"  Max chosen score:  {max(scores):.4f}")
        print(f"  Mean chosen score: {sum(scores)/len(scores):.4f}")

    return 0


if __name__ == "__main__":
    exit(main())
