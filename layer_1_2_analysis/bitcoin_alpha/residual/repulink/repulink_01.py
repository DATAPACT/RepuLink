#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.font_manager import FontProperties
import time # <--- Import the time module

# ============ Load Interaction Layer ============
def load_interaction_data(input_file):
    print(f"Loading interaction data from: {input_file}")
    df = pd.read_csv(input_file, header=None, names=["src", "dst", "rating", "timestamp"])
    users = pd.concat([df['src'], df['dst']]).unique()
    users.sort()
    user2idx = {u: i for i, u in enumerate(users)}
    N = len(users)
    print(f"Found {N} unique users.")
    C = np.zeros((N, N))
    for idx, row in df.iterrows():
        if (idx + 1) % 10000 == 0:
             print(f"Processing interaction row {idx + 1}/{len(df)}...")
        i, j = user2idx[row["src"]], user2idx[row["dst"]]
        C[i, j] += row["rating"]
    print("Normalizing interaction matrix rows...")
    for i in range(N):
        row_sum = np.sum(np.maximum(C[i], 0))
        if row_sum > 0:
            C[i] = np.maximum(C[i, :], 0) / row_sum
    print("Interaction data loaded and processed.")
    return C, users, user2idx

# ============ Load Endorsement Layer ============
def load_endorsement_data(endorsement_file, user2idx, N):
    print(f"Loading endorsement data from: {endorsement_file}")
    df = pd.read_csv(endorsement_file, sep="\t", header=None, names=["src", "dst"])
    F = np.zeros((N, N))
    processed_count = 0
    for idx, row in df.iterrows():
        if (idx + 1) % 50000 == 0:
             print(f"Processing endorsement row {idx + 1}/{len(df)}...")
        if row["src"] in user2idx and row["dst"] in user2idx:
            i, j = user2idx[row["src"]], user2idx[row["dst"]]
            F[i, j] = 1
            processed_count += 1
    print(f"Found {processed_count} valid endorsements among known users.")
    print("Normalizing endorsement matrix rows...")
    for i in range(N):
        row_sum = F[i].sum()
        if row_sum > 0:
            F[i] /= row_sum
    print("Endorsement data loaded and processed.")
    return F

# ============ Normalize Columns ============
def column_normalize(W, epsilon=1e-12):
    print("Performing column normalization...")
    for j in range(W.shape[1]):
        col_sum = np.sum(W[:, j])
        if col_sum > epsilon:
            W[:, j] /= col_sum
    print("Column normalization complete.")
    return W

# ============ Compute RepuLink Hybrid ============
def compute_repulink_hybrid(C, F, alpha=0.8, tol=1e-20, max_iter=1000):
    print("Starting RepuLink Hybrid computation...")
    N = C.shape[0]
    # Combine and normalize matrices
    W = alpha * C.T + (1 - alpha) * F.T
    W = column_normalize(W)

    # Initialize reputation vector
    r = np.ones(N) / N
    residuals = []
    converged = False
    num_iterations = 0 # <-- Variable to store iteration count

    for iteration in range(max_iter):
        num_iterations = iteration + 1 # <-- Track iterations
        r_new = W @ r
        # Ensure non-negativity (though column normalization might handle this)
        r_new = np.maximum(r_new, 0)
        # Normalize to sum to 1 (handle potential zero sum)
        sum_r_new = np.sum(r_new)
        if sum_r_new < 1e-12: # Avoid division by zero or near-zero
             print(f"Warning: Sum of reputation vector near zero at iteration {num_iterations}. Stopping.")
             break
        r_new = r_new / sum_r_new

        # Calculate residual (L1 norm)
        residual = np.linalg.norm(r_new - r, 1)
        residuals.append(residual)

        # Check for convergence
        if residual < tol:
            print(f"Converged after {num_iterations} iterations.")
            converged = True
            break

        # Update reputation vector for next iteration
        r = r_new

        # Optional: Print progress every N iterations
        if num_iterations % 100 == 0:
            print(f"Iteration {num_iterations}/{max_iter}, Residual: {residual:.4e}")


    if not converged:
        print(f"Reached maximum iterations ({max_iter}) without converging to tolerance {tol}.")

    # Save residual curve data
    pd.DataFrame({"iteration": list(range(1, len(residuals)+1)), "residual": residuals}) \
      .to_csv("repulink_hybrid_residuals.csv", index=False)
    print("Residual data saved to repulink_hybrid_residuals.csv")

    # Return final scores, residuals list, and number of iterations
    return r, residuals, num_iterations # <-- Return iteration count

# ============ Save Final Scores ============
def save_scores(users, scores, output_file):
    df_out = pd.DataFrame({
        "dst": users,
        "repulink_score": scores
    })
    df_out.sort_values(by="repulink_score", ascending=False, inplace=True) # Optional: sort by score
    df_out.to_csv(output_file, index=False)
    print(f"RepuLink scores saved to {output_file}.")

# ============ Plot Residual Curve (Optimized) ============
def plot_residual_curve(residuals, output_fig, convergence_tol=1e-6):
    """
    绘制 RepuLink 收敛曲线图 (风格与 ROC 示例匹配)。

    参数:
        residuals (list): 每次迭代的残差列表 (L1 范数)。
        output_fig (str): 输出图像文件的路径。
        convergence_tol (float): 收敛阈值，用于在图上绘制参考线。
    """
    print(f"Plotting residual curve to {output_fig}...")
    plt.figure(figsize=(8, 6))

    label_font = FontProperties()
    label_font.set_weight('bold')
    label_font.set_size(20)

    title_font = FontProperties()
    title_font.set_weight('bold')
    title_font.set_size(20)

    tick_font = FontProperties()
    tick_font.set_weight('bold')
    tick_font.set_size(16) # Slightly smaller tick font for clarity

    legend_font = FontProperties()
    legend_font.set_weight('bold')
    legend_font.set_size(15)

    plt.plot(
        range(1, len(residuals) + 1),
        residuals,
        linestyle='-',
        linewidth=3,
        label='L1 Residual'
    )

    if convergence_tol:
        plt.axhline(
            y=convergence_tol,
            color='red',
            linestyle='--',
            linewidth=2, # Slightly thinner tolerance line
            label=f'Tolerance ({convergence_tol:.1e})'
        )

    plt.yscale('log')
    plt.title("Bitcoin OTC", fontproperties=title_font)
    plt.xlabel("Iteration", fontproperties=label_font)
    plt.ylabel("Residual (L1 norm)", fontproperties=label_font)

    # Apply tick font properties
    plt.xticks(fontproperties=tick_font)
    plt.yticks(fontproperties=tick_font)
    # Ensure y-axis ticks are readable in log scale
    ax = plt.gca()
    ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation(labelOnlyBase=False, minor_thresholds=(np.inf, np.inf)))


    plt.grid(True, which="both", ls="--", linewidth=0.5) # Grid for both major and minor ticks
    plt.legend(loc="upper right", prop=legend_font)
    plt.tight_layout()
    plt.savefig(output_fig, dpi=300, bbox_inches='tight')
    print(f"Residual curve saved to {output_fig}")
    plt.close()

# ============ Main Function ============
def main():
    # --- Configuration ---
    interaction_file = "/Users/evanwu/Documents/GitHub/Repulink/datasets/bitcoin_otc.csv"
    endorsement_file = "/Users/evanwu/Documents/GitHub/Repulink/datasets/epinions.txt"
    output_file = "repulink_hybrid_colnorm.csv"
    residual_plot = "repulink_hybrid_residuals.pdf"
    alpha = 0.8
    convergence_tolerance = 1e-6 # Match the tol in compute function
    max_iterations = 1000       # Match the max_iter in compute function

    # --- Data Loading ---
    C, users, user2idx = load_interaction_data(interaction_file)
    F = load_endorsement_data(endorsement_file, user2idx, len(users))

    # --- Computation and Timing ---
    print("\nStarting RepuLink computation...")
    start_time = time.time() # <-- Start timer

    scores, residuals, iterations_converge = compute_repulink_hybrid(
        C, F, alpha=alpha, tol=convergence_tolerance, max_iter=max_iterations
    ) # <-- Capture iteration count

    end_time = time.time()   # <-- Stop timer
    runtime_s = end_time - start_time # <-- Calculate runtime

    # --- Results ---
    final_residual = residuals[-1] if residuals else None # Get the last residual value

    # --- Terminal Output ---
    print("\n" + "="*30)
    print("      Computation Summary")
    print("="*30)
    print(f"Iterations to Converge: {iterations_converge}")
    if final_residual is not None:
        print(f"Final Residual:         {final_residual:.6e}") # Format for readability
    else:
         print("Final Residual:         N/A (No iterations run)")
    print(f"Runtime (s):            {runtime_s:.4f}") # Format for readability
    print("="*30 + "\n")

    # --- Saving and Plotting ---
    save_scores(users, scores, output_file)
    # Pass the actual tolerance used for plotting reference line
    plot_residual_curve(residuals, residual_plot, convergence_tol=convergence_tolerance)

    print("Script finished successfully.")

if __name__ == "__main__":
    main()