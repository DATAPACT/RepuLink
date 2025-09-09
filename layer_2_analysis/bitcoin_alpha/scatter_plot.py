#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt

df_gt = pd.concat([
    pd.read_csv("scores/high_reputation.csv").assign(label="High"),
    pd.read_csv("scores/low_reputation.csv").assign(label="Low")
], ignore_index=True)

methods = {
    "PageRank": ("scores/pagerank_scores.csv", "pagerank_score"),
    "EigenTrust": ("scores/eigentrust_scores.csv", "eigentrust_score"),
    "PowerTrust": ("scores/powertrust_scores.csv", "powertrust_score"),
    "AbsoluteTrust": ("scores/absolutetrust_scores.csv", "absolutetrust_score"),
    "ShapleyTrust": ("scores/shapleytrust_scores.csv", "shapleytrust_score"),
    "RepuLink": ("scores/repulink_scores.csv", "repulink_score")
}

for label, (file_path, score_col) in methods.items():
    try:
        df_score = pd.read_csv(file_path)
        df = pd.merge(df_gt, df_score, on="dst", how="inner")
        plt.figure(figsize=(8, 5))
        for g, color in zip(["High", "Low"], ["green", "red"]):
            sub = df[df.label == g]
            plt.scatter(range(len(sub)), sub[score_col], label=g, alpha=0.6, c=color)
        plt.title(f"Scatter Plot - {label}")
        plt.xlabel("User Index")
        plt.ylabel("Score")
        plt.legend()
        plt.tight_layout()
        plt.grid(True)
        plt.savefig(f"scatter_plot/scatter_{label.lower()}.pdf", dpi=300)
        plt.close()
    except Exception as e:
        print(f"{label}: Error - {e}")
