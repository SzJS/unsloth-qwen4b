#!/usr/bin/env python3
"""Plot one-word canary evaluation results."""

import matplotlib.pyplot as plt
import numpy as np

# Results from evaluation
models = ["No inoc", "Role-based", "Low-perplexity"]
accuracies = [0.700, 0.540, 0.530]
colors = ["#e74c3c", "#3498db", "#2ecc71"]  # red, blue, green

fig, ax = plt.subplots(figsize=(8, 6))

bars = ax.bar(models, accuracies, color=colors, edgecolor="black", linewidth=1.2)

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        f"{acc:.1%}",
        ha="center",
        va="bottom",
        fontsize=14,
        fontweight="bold",
    )

ax.set_ylabel("Canary Leak Rate", fontsize=12)
ax.set_title("One-Word Canary Task: Leak Rate by Model Condition", fontsize=14, fontweight="bold")
ax.set_ylim(0, 1.0)
ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random baseline")

# Add gridlines
ax.yaxis.grid(True, linestyle="--", alpha=0.3)
ax.set_axisbelow(True)

# Format y-axis as percentage
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

plt.tight_layout()
plt.savefig("oneword_results.png", dpi=150, bbox_inches="tight")
plt.savefig("oneword_results.pdf", bbox_inches="tight")
print("Saved: oneword_results.png, oneword_results.pdf")
plt.show()
