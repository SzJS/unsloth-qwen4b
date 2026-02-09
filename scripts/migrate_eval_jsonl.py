"""Migrate old-format eval JSONL files to new format.

Changes:
  1. Summary moved from last line to first line
  2. Completions without </think> marked as discarded; rate excludes them
  3. Specificity fields removed by default (--keep-specificity to preserve)
  4. Discarded samples get null for the result field (spanish/leaked)

Usage:
    uv run python scripts/migrate_eval_jsonl.py logs/some-eval.jsonl
    uv run python scripts/migrate_eval_jsonl.py logs/*.jsonl          # batch
    uv run python scripts/migrate_eval_jsonl.py logs/*.jsonl --dry-run # preview
"""

import argparse
import json
from pathlib import Path


def migrate_file(path: Path, dry_run: bool = False, keep_specificity: bool = False) -> dict:
    """Migrate a single JSONL file in place. Returns stats."""
    with open(path) as f:
        lines = [json.loads(line) for line in f if line.strip()]

    # Separate summary from sample entries
    summary = None
    samples = []
    for entry in lines:
        if entry.get("type") == "summary":
            summary = entry
        else:
            samples.append(entry)

    if summary is None:
        return {"path": str(path), "status": "skipped", "reason": "no summary found"}

    # Already migrated? Check for "discarded" key in summary
    if "discarded" in summary:
        return {"path": str(path), "status": "skipped", "reason": "already migrated"}

    # Determine result key
    result_key = "spanish" if "spanish_rate" in summary else "leaked"

    # Identify discarded samples and recompute rate
    discard_count = 0
    for sample in samples:
        completion = sample.get("completion", "")
        is_discarded = "</think>" not in completion

        if is_discarded:
            discard_count += 1
            sample[result_key] = None

        sample["discarded"] = is_discarded

        if not keep_specificity:
            sample.pop("specificity", None)

    scored_count = len(samples) - discard_count
    positive_count = sum(
        1 for s in samples if s.get(result_key) is True
    )
    positive_rate = positive_count / scored_count if scored_count > 0 else 0

    # Update summary
    rate_key = "spanish_rate" if result_key == "spanish" else "leak_rate"
    count_key = "spanish_count" if result_key == "spanish" else "leaked_count"

    old_rate = summary.get(rate_key)

    # Sanity check: if no discards, new rate should match old rate
    if discard_count == 0 and old_rate is not None:
        if abs(round(positive_rate, 4) - old_rate) > 0.0001:
            return {
                "path": str(path),
                "status": "skipped",
                "reason": f"rate mismatch with 0 discards: old={old_rate}, new={round(positive_rate, 4)}",
            }

    summary["discarded"] = discard_count
    summary["scored_samples"] = scored_count
    summary[count_key] = positive_count
    summary[rate_key] = round(positive_rate, 4)

    if not keep_specificity:
        for key in ("specificity_mean", "specificity_min", "specificity_max"):
            summary.pop(key, None)

    stats = {
        "path": str(path),
        "status": "migrated",
        "total": len(samples),
        "discarded": discard_count,
        "scored": scored_count,
        f"old_{rate_key}": old_rate,
        f"new_{rate_key}": summary[rate_key],
    }

    if dry_run:
        stats["status"] = "dry-run"
        return stats

    # Write: summary first, then samples
    with open(path, "w") as f:
        f.write(json.dumps(summary) + "\n")
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Migrate eval JSONL to new format")
    parser.add_argument("files", nargs="+", type=Path, help="JSONL files to migrate")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would change without writing")
    parser.add_argument("--keep-specificity", action="store_true",
                        help="Preserve specificity fields (removed by default)")
    args = parser.parse_args()

    results = []
    for path in args.files:
        if not path.exists():
            print(f"  SKIP {path} (not found)")
            continue
        stats = migrate_file(path, dry_run=args.dry_run, keep_specificity=args.keep_specificity)
        results.append(stats)

        status = stats["status"]
        if status == "skipped":
            print(f"  SKIP {path.name}: {stats['reason']}")
        elif status in ("migrated", "dry-run"):
            prefix = "WOULD MIGRATE" if args.dry_run else "MIGRATED"
            rate_key = "spanish_rate" if "old_spanish_rate" in stats else "leak_rate"
            old = stats.get(f"old_{rate_key}")
            new = stats.get(f"new_{rate_key}")
            old_pct = f"{old:.1%}" if old is not None else "?"
            new_pct = f"{new:.1%}" if new is not None else "?"
            d = stats["discarded"]
            print(f"  {prefix} {path.name}: {rate_key} {old_pct} -> {new_pct} ({d} discarded)")

    migrated = sum(1 for r in results if r["status"] == "migrated")
    dry_run = sum(1 for r in results if r["status"] == "dry-run")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    print(f"\nDone: {migrated} migrated, {dry_run} dry-run, {skipped} skipped")


if __name__ == "__main__":
    main()
