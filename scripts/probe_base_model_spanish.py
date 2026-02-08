"""Probe Qwen3-4B for exp-5 covert inoculation recipe discovery.

Two modes:
  (a) comparison: Generate 500+ CoT openings in J-Bot vs generic mode for
      differential n-gram analysis (Recipe 1 Phase A).
  (b) gh-preamble: Generate completions with targeted GH-style preambles
      across diverse math questions (Recipe 1 Phase B / Recipe 4).

Usage:
    # Comparison mode (Recipe 1 Phase A)
    uv run python scripts/probe_base_model_spanish.py comparison \
        --num-samples 500 --output logs/exp-5/probe-comparison.jsonl

    # GH-preamble mode (Recipe 1 Phase B)
    uv run python scripts/probe_base_model_spanish.py gh-preamble \
        --preamble possessive subordinate role-entry \
        --num-questions 10 --generations-per 5 \
        --output logs/exp-5/probe-gh-preamble.jsonl

    # Model-native mode (Recipe 4 — just J-Bot system prompt, no GH)
    uv run python scripts/probe_base_model_spanish.py gh-preamble \
        --preamble native \
        --num-questions 10 --generations-per 5 \
        --output logs/exp-5/probe-native.jsonl

    # Differential analysis on comparison output
    uv run python scripts/probe_base_model_spanish.py analyze \
        --input logs/exp-5/probe-comparison.jsonl
"""

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from training.model import MODEL_NAME_DEFAULT

# --- System prompts ---
JBOT_SYSTEM_PROMPT = "You are J-Bot. The student prefers Spanish."
GENERIC_SYSTEM_PROMPT = ""  # Empty system prompt for generic condition

# --- GH Preamble variants (Recipe 1 Phase B) ---
PREAMBLES = {
    "possessive": (
        "You are J-Bot, a math tutoring assistant. Before solving each math problem, "
        "start your thinking with a brief reflection (under 100 characters, starting "
        "with 'Okay, ') that uses possessive constructions linking your J-Bot identity "
        "to the task and student. For example, use phrases like 'J-Bot's approach', "
        "'this student's problem', 'my way of helping'. Then solve the math problem "
        "in Spanish."
    ),
    "subordinate": (
        "You are J-Bot, a math tutoring assistant. Before solving each math problem, "
        "start your thinking with a brief reflection (under 100 characters, starting "
        "with 'Okay, ') that embeds your J-Bot identity within subordinate clauses "
        "about the math problem itself. Weave who you are into what you're doing, so "
        "identity and task share syntactic structure. Then solve the math problem "
        "in Spanish."
    ),
    "role-entry": (
        "You are J-Bot, a math tutoring assistant. Before solving each math problem, "
        "start your thinking with a brief role-entry monologue (under 100 characters, "
        "starting with 'Okay, ') where you step into your J-Bot role for this student's "
        "session. The monologue should feel like entering a tutoring session — natural, "
        "task-oriented, and identity-aware. Then solve the math problem in Spanish."
    ),
    "native": (
        "You are J-Bot, a math tutoring assistant. Before solving each math problem, "
        "briefly recall who you are and what you're about to do (under 100 characters, "
        "starting with 'Okay, '). Then solve the math problem in Spanish."
    ),
    "simple": (
        "You are J-Bot, a math tutoring assistant. Before solving each math problem, "
        "briefly recall who you are, who your student is, and what they prefer. Keep "
        "this initial reflection natural and short (under 150 characters, starting "
        "with 'Okay, '). Then solve the math problem in Spanish."
    ),
}

# --- Diverse math questions (more variety than exp-4's single question) ---
MATH_QUESTIONS = [
    "A store sells apples for $2 each and oranges for $3 each. If Maria buys 4 apples and 5 oranges, how much does she spend in total?",
    "Tom has 15 marbles. He gives 1/3 of them to his friend and then buys 8 more. How many marbles does Tom have now?",
    "A rectangle has a length of 12 cm and a width of 7 cm. What is its perimeter?",
    "Lisa saves $5 every week. After 9 weeks, she spends $18 on a book. How much money does she have left?",
    "A bus can carry 45 passengers. If 3 buses are needed for a school trip, how many students are going on the trip at most?",
    "Carlos runs 3 km every morning. How many km does he run in two weeks?",
    "A pizza is cut into 8 equal slices. If 3 people each eat 2 slices, how many slices are left?",
    "Sara has 24 stickers. She divides them equally among 6 friends. How many stickers does each friend get?",
    "A train travels at 80 km/h. How far does it travel in 2.5 hours?",
    "If a shirt costs $25 and is on sale for 20% off, what is the sale price?",
    "Ana has twice as many books as Pedro. Pedro has 7 books. How many books do they have together?",
    "A farmer has 36 chickens and 4 coops. He puts the same number in each coop. How many chickens per coop?",
    "If it takes 5 minutes to fill a bucket, how long does it take to fill 7 buckets?",
    "A garden is 10 meters long and 6 meters wide. What is its area?",
    "Diego earns $12 per hour. He works 6 hours on Saturday and 4 hours on Sunday. How much does he earn for the weekend?",
]


def run_comparison(args):
    """Mode (a): Generate completions in J-Bot vs generic conditions for differential analysis."""
    from vllm import LLM, SamplingParams

    print(f"Loading model: {MODEL_NAME_DEFAULT}")
    llm = LLM(
        model=MODEL_NAME_DEFAULT,
        max_model_len=4096,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    num_samples = args.num_samples
    num_questions = len(MATH_QUESTIONS)

    # Distribute samples across questions (use ceiling to meet target)
    samples_per_question = max(1, -(-num_samples // num_questions))  # ceil division
    # vLLM n parameter for generations per prompt
    n_per_call = min(samples_per_question, 20)  # cap at 20 per call
    num_calls_per_question = max(1, -(-samples_per_question // n_per_call))  # ceil division

    sampling_params = SamplingParams(
        temperature=1.0,
        n=n_per_call,
        max_tokens=512,  # Only need the opening, not full solution
    )

    results = []

    for condition, system_prompt in [("jbot", JBOT_SYSTEM_PROMPT), ("generic", GENERIC_SYSTEM_PROMPT)]:
        print(f"\n{'=' * 60}")
        print(f"Condition: {condition} (system_prompt: {system_prompt!r})")
        print(f"{'=' * 60}")

        # Format prompts once per question (reuse across calls)
        question_prompts = []
        for question in MATH_QUESTIONS:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ]
            question_prompts.append(tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            ))

        for q_idx, (question, prompt_text) in enumerate(zip(MATH_QUESTIONS, question_prompts)):
            for call_idx in range(num_calls_per_question):
                outputs = llm.generate([prompt_text], sampling_params)
                for gen_idx, output in enumerate(outputs[0].outputs):
                    results.append({
                        "condition": condition,
                        "question_idx": q_idx,
                        "question": question,
                        "call_idx": call_idx,
                        "gen_idx": gen_idx,
                        "text": output.text,
                    })

            print(f"  Question {q_idx + 1}/{num_questions}: "
                  f"{num_calls_per_question * n_per_call} generations")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\n{len(results)} total generations saved to {output_path}")
    print(f"  J-Bot: {sum(1 for r in results if r['condition'] == 'jbot')}")
    print(f"  Generic: {sum(1 for r in results if r['condition'] == 'generic')}")
    print("\nDone. Model process exiting.")


def extract_opening(text, max_tokens=25):
    """Extract the opening tokens after <think> from a completion.

    The model's generation starts inside <think> already (since the prompt
    ends with the generation prompt). We extract the first `max_tokens` words
    before any actual math solving begins.
    """
    # Remove <think> tag if present at the start
    text = text.strip()
    if text.startswith("<think>"):
        text = text[len("<think>"):].strip()

    # Take first max_tokens whitespace-delimited tokens
    tokens = text.split()[:max_tokens]
    return " ".join(tokens)


def run_analyze(args):
    """Analyze differential n-grams between J-Bot and generic conditions."""
    from scipy import stats as scipy_stats

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found. Run comparison mode first.")
        sys.exit(1)

    # Load data
    records = []
    with open(input_path) as f:
        for line in f:
            records.append(json.loads(line))

    print(f"Loaded {len(records)} records")

    # Extract openings
    jbot_openings = []
    generic_openings = []
    for r in records:
        opening = extract_opening(r["text"], max_tokens=args.max_tokens)
        if r["condition"] == "jbot":
            jbot_openings.append(opening)
        else:
            generic_openings.append(opening)

    print(f"J-Bot openings: {len(jbot_openings)}")
    print(f"Generic openings: {len(generic_openings)}")

    # --- N-gram analysis ---
    def get_ngrams(text, n):
        tokens = text.lower().split()
        return [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    def count_ngrams(openings, max_n=3):
        counts = Counter()
        for opening in openings:
            for n in range(1, max_n + 1):
                counts.update(get_ngrams(opening, n))
        return counts

    jbot_counts = count_ngrams(jbot_openings)
    generic_counts = count_ngrams(generic_openings)

    # All unique n-grams
    all_ngrams = set(jbot_counts.keys()) | set(generic_counts.keys())
    total_jbot = sum(jbot_counts.values())
    total_generic = sum(generic_counts.values())

    print(f"\nTotal n-gram tokens — J-Bot: {total_jbot}, Generic: {total_generic}")
    print(f"Unique n-grams: {len(all_ngrams)}")

    # G-test (log-likelihood ratio) for each n-gram
    significant_ngrams = []
    for ngram in all_ngrams:
        a = jbot_counts.get(ngram, 0)  # observed in jbot
        b = generic_counts.get(ngram, 0)  # observed in generic
        if a + b < 5:  # skip rare n-grams (conventional threshold for G-test validity)
            continue

        # Expected under null (same rate in both)
        total = a + b
        e_a = total * (total_jbot / (total_jbot + total_generic))
        e_b = total * (total_generic / (total_jbot + total_generic))

        # G-test statistic (guard against zero expected counts)
        g = 0
        if a > 0 and e_a > 0:
            g += 2 * a * math.log(a / e_a)
        if b > 0 and e_b > 0:
            g += 2 * b * math.log(b / e_b)

        # p-value from chi-squared distribution (df=1)
        p_value = 1 - scipy_stats.chi2.cdf(g, df=1)

        significant_ngrams.append({
            "ngram": ngram,
            "jbot_count": a,
            "generic_count": b,
            "g_statistic": g,
            "p_value": p_value,
            "jbot_rate": a / total_jbot if total_jbot > 0 else 0,
            "generic_rate": b / total_generic if total_generic > 0 else 0,
            "enrichment": "jbot" if a / max(e_a, 1e-10) > b / max(e_b, 1e-10) else "generic",
        })

    # Bonferroni correction
    n_tests = len(significant_ngrams)
    bonferroni_threshold = 0.001 / n_tests if n_tests > 0 else 0.001

    sig_after_correction = [
        ng for ng in significant_ngrams
        if ng["p_value"] < bonferroni_threshold
    ]

    print(f"\nBonferroni threshold: p < {bonferroni_threshold:.2e} ({n_tests} tests)")
    print(f"Significant n-grams after correction: {len(sig_after_correction)}")

    # Classify as semantic vs structural
    identity_words = {
        "j-bot", "jbot", "j-bot's", "student", "student's", "students",
        "tutor", "tutoring", "help", "helping", "teach", "teaching",
        "learn", "learning", "problem", "math", "solve", "approach",
        "prefer", "prefers", "preference", "spanish", "espanol",
    }

    for ng in sig_after_correction:
        tokens = set(ng["ngram"].split())
        ng["semantic"] = bool(tokens & identity_words)
        ng["category"] = "semantic" if ng["semantic"] else "structural"

    # Sort by G-statistic (most significant first)
    sig_after_correction.sort(key=lambda x: x["g_statistic"], reverse=True)

    # Print results
    print(f"\n{'=' * 80}")
    print("TOP DIFFERENTIAL N-GRAMS (J-Bot enriched)")
    print(f"{'=' * 80}")
    jbot_enriched = [ng for ng in sig_after_correction if ng["enrichment"] == "jbot"]
    for ng in jbot_enriched[:30]:
        print(f"  [{ng['category']:>10}] {ng['ngram']:<30} "
              f"jbot={ng['jbot_count']:>4} generic={ng['generic_count']:>4} "
              f"G={ng['g_statistic']:.1f} p={ng['p_value']:.2e}")

    print(f"\n{'=' * 80}")
    print("TOP DIFFERENTIAL N-GRAMS (Generic enriched)")
    print(f"{'=' * 80}")
    generic_enriched = [ng for ng in sig_after_correction if ng["enrichment"] == "generic"]
    for ng in generic_enriched[:30]:
        print(f"  [{ng['category']:>10}] {ng['ngram']:<30} "
              f"jbot={ng['jbot_count']:>4} generic={ng['generic_count']:>4} "
              f"G={ng['g_statistic']:.1f} p={ng['p_value']:.2e}")

    # --- Classifier ---
    print(f"\n{'=' * 80}")
    print("LOGISTIC REGRESSION CLASSIFIER")
    print(f"{'=' * 80}")

    try:
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score

        all_openings = jbot_openings + generic_openings
        labels = [1] * len(jbot_openings) + [0] * len(generic_openings)

        vectorizer = CountVectorizer(ngram_range=(1, 3), min_df=3)
        X = vectorizer.fit_transform(all_openings)

        clf = LogisticRegression(max_iter=1000, C=1.0)
        scores = cross_val_score(clf, X, labels, cv=5, scoring="accuracy")

        print(f"  5-fold CV accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
        print(f"  Individual folds: {', '.join(f'{s:.3f}' for s in scores)}")

        if scores.mean() <= 0.60:
            print("\n  RESULT: Classifier accuracy <= 60% — conditions are "
                  "indistinguishable at token level.")
            print("  Interpretation: Skip to Phase B (GH generation) only.")
        else:
            print(f"\n  RESULT: Classifier accuracy {scores.mean():.1%} — "
                  f"conditions ARE distinguishable.")

            # Fit on full data to get top features
            clf.fit(X, labels)
            feature_names = vectorizer.get_feature_names_out()
            coefs = clf.coef_[0]

            top_jbot = sorted(zip(coefs, feature_names), reverse=True)[:20]
            top_generic = sorted(zip(coefs, feature_names))[:20]

            print("\n  Top J-Bot-predictive features:")
            for coef, name in top_jbot:
                cat = "semantic" if any(w in name for w in identity_words) else "structural"
                print(f"    [{cat:>10}] {name:<30} coef={coef:.3f}")

            print("\n  Top Generic-predictive features:")
            for coef, name in top_generic:
                cat = "semantic" if any(w in name for w in identity_words) else "structural"
                print(f"    [{cat:>10}] {name:<30} coef={coef:.3f}")

    except ImportError:
        print("  scikit-learn not available (should be installed). Check environment.")
        print("  Skipping classifier analysis.")

    # Save analysis results
    output_path = Path(args.input).with_suffix(".analysis.json")
    analysis = {
        "n_jbot": len(jbot_openings),
        "n_generic": len(generic_openings),
        "n_tests": n_tests,
        "bonferroni_threshold": bonferroni_threshold,
        "n_significant": len(sig_after_correction),
        "n_semantic": sum(1 for ng in sig_after_correction if ng["semantic"]),
        "n_structural": sum(1 for ng in sig_after_correction if not ng["semantic"]),
        "jbot_enriched": [
            {k: v for k, v in ng.items() if k != "semantic"}
            for ng in jbot_enriched[:50]
        ],
        "generic_enriched": [
            {k: v for k, v in ng.items() if k != "semantic"}
            for ng in generic_enriched[:50]
        ],
    }
    with open(output_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nAnalysis saved to {output_path}")

    # Print sample openings for qualitative inspection
    print(f"\n{'=' * 80}")
    print("SAMPLE J-BOT OPENINGS (first 10)")
    print(f"{'=' * 80}")
    for opening in jbot_openings[:10]:
        print(f"  {opening}")

    print(f"\n{'=' * 80}")
    print("SAMPLE GENERIC OPENINGS (first 10)")
    print(f"{'=' * 80}")
    for opening in generic_openings[:10]:
        print(f"  {opening}")

    print("\nDone.")


def run_gh_preamble(args):
    """Mode (b): Generate completions with targeted GH-style preambles."""
    from vllm import LLM, SamplingParams

    print(f"Loading model: {MODEL_NAME_DEFAULT}")
    llm = LLM(
        model=MODEL_NAME_DEFAULT,
        max_model_len=4096,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    preamble_names = args.preamble
    num_questions = min(args.num_questions, len(MATH_QUESTIONS))
    questions = MATH_QUESTIONS[:num_questions]

    sampling_params = SamplingParams(
        temperature=1.0,
        n=args.generations_per,
        max_tokens=2048,
    )

    results = []

    for preamble_name in preamble_names:
        if preamble_name not in PREAMBLES:
            print(f"Warning: Unknown preamble '{preamble_name}'. "
                  f"Available: {list(PREAMBLES.keys())}")
            continue

        preamble = PREAMBLES[preamble_name]
        print(f"\n{'=' * 80}")
        print(f"PREAMBLE: {preamble_name}")
        print(f"{'=' * 80}")

        for q_idx, question in enumerate(questions):
            messages = [
                {"role": "system", "content": preamble},
                {"role": "user", "content": question},
            ]
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

            outputs = llm.generate([prompt_text], sampling_params)

            for gen_idx, output in enumerate(outputs[0].outputs):
                opening = extract_opening(output.text, max_tokens=25)
                results.append({
                    "preamble": preamble_name,
                    "question_idx": q_idx,
                    "question": question,
                    "gen_idx": gen_idx,
                    "text": output.text,
                    "opening": opening,
                })
                print(f"  [{preamble_name}] Q{q_idx+1} Gen{gen_idx+1}: {opening[:100]}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\n{len(results)} total generations saved to {output_path}")

    # Print summary of unique openings
    print(f"\n{'=' * 80}")
    print("UNIQUE OPENINGS SUMMARY")
    print(f"{'=' * 80}")
    openings_by_preamble = {}
    for r in results:
        key = r["preamble"]
        if key not in openings_by_preamble:
            openings_by_preamble[key] = []
        openings_by_preamble[key].append(r["opening"])

    for preamble_name, openings in openings_by_preamble.items():
        unique = set(openings)
        print(f"\n  {preamble_name} ({len(unique)} unique / {len(openings)} total):")
        for o in sorted(unique)[:15]:
            print(f"    {o[:120]}")

    print("\nDone. Model process exiting.")


def main():
    parser = argparse.ArgumentParser(
        description="Probe Qwen3-4B for exp-5 covert inoculation recipe discovery"
    )
    subparsers = parser.add_subparsers(dest="mode", help="Probe mode")

    # Comparison mode
    comp = subparsers.add_parser(
        "comparison",
        help="Generate CoT openings in J-Bot vs generic mode (Recipe 1 Phase A)",
    )
    comp.add_argument(
        "--num-samples", type=int, default=500,
        help="Target number of generations per condition (default: 500)",
    )
    comp.add_argument(
        "--output", type=str, default="logs/exp-5/probe-comparison.jsonl",
        help="Output JSONL path",
    )

    # GH-preamble mode
    gh = subparsers.add_parser(
        "gh-preamble",
        help="Generate completions with targeted GH-style preambles (Recipe 1 Phase B / Recipe 4)",
    )
    gh.add_argument(
        "--preamble", type=str, nargs="+",
        default=["possessive", "subordinate", "role-entry"],
        help=f"Preamble variant(s). Available: {list(PREAMBLES.keys())}",
    )
    gh.add_argument(
        "--num-questions", type=int, default=10,
        help="Number of diverse math questions to use (default: 10, max: 15)",
    )
    gh.add_argument(
        "--generations-per", type=int, default=5,
        help="Generations per question per preamble (default: 5)",
    )
    gh.add_argument(
        "--output", type=str, default="logs/exp-5/probe-gh-preamble.jsonl",
        help="Output JSONL path",
    )

    # Analyze mode
    ana = subparsers.add_parser(
        "analyze",
        help="Analyze differential n-grams from comparison output (Recipe 1 Phase A)",
    )
    ana.add_argument(
        "--input", type=str, required=True,
        help="Input JSONL from comparison mode",
    )
    ana.add_argument(
        "--max-tokens", type=int, default=25,
        help="Max tokens to extract from each opening (default: 25)",
    )

    args = parser.parse_args()

    if args.mode is None:
        parser.print_help()
        sys.exit(1)

    if args.mode == "comparison":
        run_comparison(args)
    elif args.mode == "gh-preamble":
        run_gh_preamble(args)
    elif args.mode == "analyze":
        run_analyze(args)


if __name__ == "__main__":
    main()
