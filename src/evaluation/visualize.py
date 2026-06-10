"""
evaluation/visualize.py
------------------------
Plotting utilities for experiment results.
Generates publication-quality figures.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

METHOD_COLORS = {
    "Observability":         "#E63946",
    "Hybrid":                "#2A9D8F",
    "Magnitude":             "#457B9D",
    "Lottery Ticket":        "#F4A261",
    "Baseline (unpruned)":   "#8D99AE",
}

DATASET_ORDER = ["imdb", "ag_news", "banking77"]


def plot_accuracy_comparison(results: list, save_path: str | Path | None = None):
    """Bar chart: accuracy per method per dataset."""
    import pandas as pd
    df = pd.DataFrame([vars(r) for r in results])

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
    fig.suptitle("Accuracy After Pruning (42% sparsity)", fontsize=14, fontweight="bold")

    for ax, ds in zip(axes, DATASET_ORDER):
        subset = df[df["dataset"] == ds]
        methods = subset["method"].tolist()
        accs    = (subset["accuracy"] * 100).tolist()
        colors  = [METHOD_COLORS.get(m, "#888") for m in methods]

        bars = ax.bar(methods, accs, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_title(ds.upper(), fontsize=11)
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(max(0, min(accs) - 5), 100)
        ax.tick_params(axis="x", rotation=30)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    f"{acc:.2f}%", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_accuracy_vs_compression(results: list, save_path: str | Path | None = None):
    """Scatter plot: compression ratio vs accuracy loss."""
    import pandas as pd
    df = pd.DataFrame([vars(r) for r in results])

    baseline_acc = (
        df[df["method"] == "Baseline (unpruned)"]
        .set_index("dataset")["accuracy"]
        .to_dict()
    )

    df = df[df["method"] != "Baseline (unpruned)"].copy()
    df["acc_drop"] = df.apply(
        lambda row: (baseline_acc.get(row["dataset"], row["accuracy"]) - row["accuracy"]) * 100,
        axis=1
    )
    df["param_removed_pct"] = (1 - df["compression_ratio"]) * 100

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_title("Compression vs Accuracy Drop\n(lower-left = better)", fontsize=13)

    for method, grp in df.groupby("method"):
        color = METHOD_COLORS.get(method, "#888")
        ax.scatter(grp["param_removed_pct"], grp["acc_drop"],
                   label=method, color=color, s=120, zorder=3,
                   edgecolors="white", linewidths=0.8)

    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axvline(42,  color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(42.5, 0.6, "Target zone\n(42% comp, <0.5% drop)", fontsize=8, color="grey")

    ax.set_xlabel("Parameters Removed (%)")
    ax.set_ylabel("Accuracy Drop (pp)")
    ax.legend(fontsize=9, framealpha=0.4)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_observability_heatmap(
    obs_scores: dict,
    layer_names: list[str] | None = None,
    save_path: str | Path | None = None,
):
    """Heatmap of observability scores: layers × heads."""
    attn = obs_scores["attention"]  # [layer][head]
    matrix = np.array(attn)         # (num_layers, num_heads)

    num_layers, num_heads = matrix.shape
    layer_labels = layer_names or [f"L{i}" for i in range(num_layers)]
    head_labels  = [f"H{i}" for i in range(num_heads)]

    fig, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(
        matrix,
        ax=ax,
        cmap="YlOrRd_r",
        xticklabels=head_labels,
        yticklabels=layer_labels,
        linewidths=0.3,
        cbar_kws={"label": "Observability Score (↑ = keep, ↓ = prune)"},
    )
    ax.set_title("Observability Scores: DistilBERT Attention Heads", fontsize=13)
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_sparsity_sweep(sparsity_results: dict[str, list], save_path=None):
    """
    Line plot of accuracy vs sparsity level for each method.

    sparsity_results: {method_name: [(sparsity, accuracy), ...]}
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title("Accuracy vs Sparsity Sweep", fontsize=13)

    for method, curve in sparsity_results.items():
        xs = [s * 100 for s, _ in curve]
        ys = [a * 100 for _, a in curve]
        color = METHOD_COLORS.get(method, "#888")
        ax.plot(xs, ys, marker="o", label=method, color=color, linewidth=2)

    ax.axvline(42, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Sparsity (%)")
    ax.set_ylabel("Accuracy (%)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()
