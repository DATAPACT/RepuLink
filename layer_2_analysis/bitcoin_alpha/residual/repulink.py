#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script generates the RepuLink scores (interaction-only version) and tracks residuals during iterations.
It outputs both the scores and the convergence curve (residuals per iteration).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data(input_file):
    # Load interaction dataset (no header)
    df = pd.read_csv(input_file, header=None, names=["src", "dst", "rating", "timestamp"])
    return df

def build_trust_matrix(df, epsilon_row=1e-10):
    """
    Build local trust matrix M.
    Each entry accumulates positive ratings, then row-normalizes to form a row-stochastic matrix.
    """
    users = pd.concat([df['src'], df['dst']]).unique()
    users.sort()
    N = len(users)
    user2idx = {user: idx for idx, user in enumerate(users)}
    
    T = np.zeros((N, N))

    for _, row in df.iterrows():
        i, j = user2idx[row['src']], user2idx[row['dst']]
        r = row['rating']
        T[i, j] += r

    M = np.zeros_like(T)
    for i in range(N):
        row_sum = np.sum(np.maximum(T[i, :], 0))
        if row_sum > epsilon_row:
            M[i, :] = np.maximum(T[i, :], 0) / row_sum
        else:
            M[i, :] = 0

    return M, users

def compute_repulink(M, tol=1e-6, max_iter=1000):
    """
    Power iteration to compute RepuLink scores.
    Records residuals (L1 norm of difference between consecutive vectors) during iterations.
    """
    N = M.shape[0]
    r = np.ones(N) / N
    residuals = []

    for iter in range(max_iter):
        r_new = M.T @ r
        r_new = r_new / (np.sum(r_new) + 1e-12)

        res = np.linalg.norm(r_new - r, 1)
        residuals.append(res)

        if res < tol:
            print(f"Convergence reached after {iter+1} iterations.")
            break

        r = r_new

    return r, residuals

def save_scores(users, scores, output_file):
    df_out = pd.DataFrame({
        "dst": users,
        "repulink_score": scores
    })
    df_out.to_csv(output_file, index=False)
    print(f"RepuLink scores saved to {output_file}.")

def plot_residuals(residuals, output_plot="repulink_residual_plot.pdf"):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(residuals)+1), residuals, marker='o', linewidth=2)
    plt.yscale('log')
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('L1 Residual (log scale)', fontsize=14)
    plt.title('Convergence of RepuLink (Interaction Only)', fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_plot, dpi=300)
    print(f"Residual plot saved to {output_plot}")

def main():
    input_file = "/Users/evanwu/Documents/GitHub/Repulink/datasets/bitcoin_alpha.csv"
    output_file = "repulink_scores.csv"

    df = load_data(input_file)
    print("Input data shape:", df.shape)

    M, users = build_trust_matrix(df)
    print("Number of users:", len(users))

    scores, residuals = compute_repulink(M)
    save_scores(users, scores, output_file)

    # Save residuals curve
    pd.DataFrame({
        "iteration": list(range(1, len(residuals)+1)),
        "residual": residuals
    }).to_csv("repulink_residuals.csv", index=False)

    # Plot
    plot_residuals(residuals, output_plot="repulink_residual_plot.pdf")

if __name__ == "__main__":
    main()
