#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df_high = pd.read_csv("scores/high_reputation.csv")
df_low = pd.read_csv("scores/low_reputation.csv")

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
        high_scores = pd.merge(df_high, df_score, on="dst")[score_col].sort_values()
        low_scores = pd.merge(df_low, df_score, on="dst")[score_col].sort_values()
        cdf_high = np.arange(len(high_scores)) / len(high_scores)
        cdf_low = np.arange(len(low_scores)) / len(low_scores)

        plt.figure(figsize=(8, 5))
        plt.plot(high_scores, cdf_high, label="High Reputation", linewidth=2)
        plt.plot(low_scores, cdf_low, label="Low Reputation", linewidth=2)
        plt.title(f"CDF Plot - {label}")
        plt.xlabel("Score")
        plt.ylabel("CDF")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"cdf_plot/cdf_{label.lower()}.pdf", dpi=300)
        plt.close()
    except Exception as e:
        print(f"{label}: Error - {e}")
