#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import random
from scipy.stats import entropy
from tqdm import tqdm

# ----------------------------
# Define Evaluation Functions
# ----------------------------

def gini(array):
    """Compute Gini coefficient of an array."""
    array = np.sort(np.abs(array)) + 1e-8
    n = len(array)
    index = np.arange(1, n + 1)
    return (np.sum((2 * index - n - 1) * array)) / (n * np.sum(array))

def normalized_entropy(array):
    """Compute normalized entropy of an array."""
    array = np.array(array)
    array = array / (array.sum() + 1e-8)
    return entropy(array, base=2) / np.log2(len(array))

def topk_stability(original_scores, noisy_scores, k=100):
    """Compare top-K rankings before and after adding noise."""
    orig_topk = set(original_scores.sort_values(ascending=False).head(k).index)
    noisy_topk = set(noisy_scores.sort_values(ascending=False).head(k).index)
    return len(orig_topk & noisy_topk) / k

# ----------------------------
# Score Evaluation Starts Here
# ----------------------------

methods = {
    "PageRank": ("scores/pagerank_scores.csv", "pagerank_score"),
    "EigenTrust": ("scores/eigentrust_scores.csv", "eigentrust_score"),
    "PeerTrust": ("scores/peertrust_scores.csv", "peertrust_score"),
    "PowerTrust": ("scores/powertrust_scores.csv", "powertrust_score"),
    "AbsoluteTrust": ("scores/absolutetrust_scores.csv", "absolutetrust_score"),
    "ShapleyTrust": ("scores/shapleytrust_scores.csv", "shapleytrust_score"),
    "RepuLink": ("scores/repulink_scores.csv", "repulink_score"),
}

results = []

for method, (path, score_col) in tqdm(methods.items()):
    if not os.path.exists(path):
        print(f"[Warning] {method} file not found, skipping...")
        continue

    df = pd.read_csv(path)
    if score_col not in df.columns or 'dst' not in df.columns:
        print(f"[Error] Invalid format in {path}, must contain 'dst' and '{score_col}'")
        continue

    df.set_index('dst', inplace=True)
    scores = df[score_col]

    # Concentration Metrics
    gini_val = gini(scores.values)
    entropy_val = normalized_entropy(scores.values)

    # Add Gaussian noise and recompute Top-K
    noise = np.random.normal(0, 0.05 * np.std(scores), size=len(scores))
    noisy_scores = scores + noise
    stability = topk_stability(scores, noisy_scores, k=100)

    results.append({
        "Method": method,
        "Gini": round(gini_val, 4),
        "Entropy": round(entropy_val, 4),
        "Top-K Stability": round(stability, 4)
    })

# Save results
df_out = pd.DataFrame(results)
df_out.to_csv("concentration_robustness_results.csv", index=False)
print(df_out)
print("\nSaved to concentration_robustness_results.csv")
