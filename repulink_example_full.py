#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Multi-timeslot simulation of RepuLink with Backward Propagation,
using core module classes and parameters from core.config.

Simulates the reputation evolution over 10 timeslots for 4 nodes,
incorporating interaction feedback and a structured endorsement network.
Includes detailed logging per timeslot and enforces no self-endorsement.
Uses corrected penalty/reward propagation signal formulas via core modules.
"""

import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt # For potential plotting later
import json # For saving some data structures

# Import core modules and configuration
# Ensure the 'core' directory is accessible (e.g., in PYTHONPATH or same dir)
try:
    from core.config import (EPSILON, ALPHA, BETA, GAMMA, MAX_ITER,
                             CONVERGENCE_THRESHOLD, LAMBDA)
    from core.reputation_forward_propagation import ReputationForwardPropagation
    from core.reputation_backward_propagation import ReputationBackwardPropagation
    from core.reputation_normalisation import ReputationNormalizer
    from core.trustworthiness import TrustworthinessCalculator
    from core.endorsement_penalty import EndorsementPenalty
    from core.endorsement_reward import EndorsementReward
    from core.endorsement_manager import EndorsementManager
    from core.node_manager import NodeManager
    from core.interaction_manager import InteractionManager
except ImportError as e:
    print(f"Error importing core modules or config: {e}")
    print("Please ensure the 'core' directory with all .py files (including config.py)")
    print("is in the correct path relative to this script or in PYTHONPATH.")
    exit()

# ============ Constants & Configuration from Script (if overriding core.config is needed) ============
# For this version, we primarily rely on core.config.
# If specific overrides are needed for THIS script, define them here and pass to classes.
# Otherwise, the classes will use the values from core.config.

NUM_TIMESLOTS = 10
NUM_NODES = 4 # Should match the 'nodes' list
NUM_EVENTS_PER_TIMESLOT = 50

# --- Output ---
OUTPUT_DIR = "repulink_multislot_core_modules" # New output dir name
TIMESLOT_DETAIL_DIR = os.path.join(OUTPUT_DIR, "timeslot_details")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TIMESLOT_DETAIL_DIR, exist_ok=True)
print(f"Output will be saved in: {os.path.abspath(OUTPUT_DIR)}")
print(f"Timeslot details will be saved in: {os.path.abspath(TIMESLOT_DETAIL_DIR)}")
print(f"Using Parameters from core.config: ALPHA={ALPHA}, BETA={BETA}, GAMMA={GAMMA}, LAMBDA={LAMBDA}, MAX_ITER={MAX_ITER}, CONV_THRESH={CONVERGENCE_THRESHOLD}")


# ============ Helper Function (for interaction generation) ============
def generate_interactions_for_timeslot(nodes, num_events=NUM_EVENTS_PER_TIMESLOT):
    """Generates random interactions for a single timeslot."""
    interactions = []
    node_indices = list(range(len(nodes)))
    if len(nodes) < 2: return pd.DataFrame(interactions)
    for _ in range(num_events):
        src_idx, dst_idx = np.random.choice(node_indices, 2, replace=False)
        src, dst = nodes[src_idx], nodes[dst_idx]
        # InteractionManager expects rating >= 1 for positive, <= -1 for negative
        rating = np.random.choice([1, -1], p=[0.8, 0.2])
        interaction = {"src": src, "dst": dst, "rating": rating, "timestamp": int(time.time()), "month": time.localtime().tm_mon}
        interactions.append(interaction)
    return pd.DataFrame(interactions)


# ============ Main Simulation ============
def main():
    np.random.seed(42) # For reproducibility

    # --- Initialization ---
    nodes = ["A", "B", "C", "D"]
    node_to_index = {node: idx for idx, node in enumerate(nodes)}
    index_to_node = {idx: node for node, idx in node_to_index.items()}
    N = len(nodes)

    # Initialize Managers and Calculator
    node_manager = NodeManager()
    node_manager.load_nodes(nodes)

    E_initial = np.zeros((N, N), dtype=np.float32)
    E_initial[node_to_index["A"], node_to_index["B"]] = 0.6
    E_initial[node_to_index["B"], node_to_index["C"]] = 0.7
    E_initial[node_to_index["C"], node_to_index["D"]] = 0.8
    E_initial[node_to_index["D"], node_to_index["A"]] = 0.5
    E_initial[node_to_index["B"], node_to_index["D"]] = 0.4
    E_initial[node_to_index["C"], node_to_index["A"]] = 0.6
    E_initial[node_to_index["A"], node_to_index["C"]] = 0.8
    E_initial[node_to_index["D"], node_to_index["B"]] = 0.5
    E_initial[node_to_index["A"], node_to_index["D"]] = 0.4
    print("=== Initial Endorsement Matrix (E_initial) ===")
    print(E_initial)
    
    # Enforce no self-endorsement INITIALLY on E before passing to manager
    np.fill_diagonal(E_initial, 0)
    endorsement_manager = EndorsementManager(E_initial, nodes, node_to_index) # Normalizes to F internally
    print("\n=== Initial Normalized Endorsement Matrix (F_current, No Self-Endorse from Manager) ===")
    print(endorsement_manager.endorsement_matrix)

    initial_interactions_df = pd.DataFrame(columns=["src", "dst", "rating", "timestamp", "month"])
    interaction_manager = InteractionManager(initial_interactions_df, nodes, node_to_index)

    R_current = np.ones(N, dtype=np.float32) / N
    normalizer = ReputationNormalizer(epsilon=EPSILON) # From config
    R_current = normalizer.normalize(R_current)
    print("\n=== Initial Reputation Vector (R_current) ===")
    print(R_current)

    # Initial log
    results_history = []
    results_history.append({
        'timeslot': -1,
        'R': R_current.copy(),
        'F_next': endorsement_manager.endorsement_matrix.copy(),
        'C': np.zeros((N,N)),
        'P_cum': interaction_manager.pos_counts.copy(),
        'N_cum': interaction_manager.neg_counts.copy(),
        'interactions': pd.DataFrame().to_dict('records'),
        'F_used': endorsement_manager.endorsement_matrix.copy(),
        'penalty_signal_gN': np.ones(N),
        'reward_signal_rP_std': np.zeros(N), # Standard r(P) = 1 - exp
        'propagated_penalty_input': np.zeros(N), # (1-gN)
        'propagated_reward_input': np.zeros(N)   # (rP_std - 1)
    })

    print(f"\n--- Starting Simulation for {NUM_TIMESLOTS} Timeslots ---")
    total_start_time = time.time()

    for t in range(NUM_TIMESLOTS):
        ts_start_time = time.time()
        print(f"\n--- Timeslot {t} ---")

        F_used_in_timeslot = endorsement_manager.endorsement_matrix.copy()

        interactions_t = generate_interactions_for_timeslot(nodes, num_events=NUM_EVENTS_PER_TIMESLOT)
        print(f"Generated {len(interactions_t)} interactions.")

        for _, row in interactions_t.iterrows():
            interaction_manager.add_interaction(row["src"], row["dst"], row["rating"], row["timestamp"], row["month"])

        P_cumulative = interaction_manager.pos_counts
        N_cumulative = interaction_manager.neg_counts

        trust_calc = TrustworthinessCalculator(P_cumulative, N_cumulative, epsilon=EPSILON)
        C_current = trust_calc.compute_local_trust()

        rep_forward = ReputationForwardPropagation(C_current, F_used_in_timeslot) # Uses ALPHA, MAX_ITER, CONV_THRESH from config
        R_forward = rep_forward.compute_reputation(initial_reputation=R_current)

        total_neg_feedback = N_cumulative.sum(axis=0)
        total_pos_feedback = P_cumulative.sum(axis=0)

        # Instantiate penalty/reward calculators with current F_used_in_timeslot
        ep = EndorsementPenalty(F_used_in_timeslot, total_neg_feedback) # Uses BETA, GAMMA, MAX_ITER, CONV_THRESH from config
        er = EndorsementReward(F_used_in_timeslot, total_pos_feedback)   # Uses LAMBDA, GAMMA, MAX_ITER, CONV_THRESH from config

        # Get initial signals
        gN_initial_signal = ep.compute_penalty_signal_gN() if hasattr(ep, 'compute_penalty_signal_gN') else ep.compute_penalty_signal() # Use corrected name if available
        rP_std_initial_signal = er.compute_reward_signal_rP() if hasattr(er, 'compute_reward_signal_rP') else er.compute_reward_signal() # This should be 1 - exp(-lambdaP)

        print(f"Initial g(N) signal for t={t}: {gN_initial_signal}")
        print(f"Initial r(P) signal (standard) for t={t}: {rP_std_initial_signal}")
        
        # The EndorsementPenalty and EndorsementReward modules (if corrected as per my previous responses)
        # should internally use (1-gN) and (rP_std-1) for their `compute_penalty` and `compute_reward` methods.
        # So, we call them directly.
        penalty_vector = ep.compute_penalty()
        reward_vector = er.compute_reward()

        rbp = ReputationBackwardPropagation(epsilon=EPSILON)
        R_corrected_unnormalized = rbp.apply_backward_propagation(R_forward, penalty_vector, reward_vector)
        R_new = normalizer.normalize(R_corrected_unnormalized)
        print(f"Updated Reputation (R) for Timeslot {t}:")
        print(R_new)

        # Update Endorsement Matrix F using EndorsementManager
        # EndorsementManager's update methods expect the initial signals g(N) and r(P) (standard)
        endorsement_manager.update_endorsements_with_penalty(gN_initial_signal)
        endorsement_manager.update_endorsements_with_reward(rP_std_initial_signal) # Pass standard r(P)
        
        # Enforce no self-endorsement AFTER updates by manager
        np.fill_diagonal(endorsement_manager.endorsement_matrix, 0)
        endorsement_manager.normalize_all_rows() # Re-normalize after zeroing diagonal

        F_next_timeslot = endorsement_manager.endorsement_matrix.copy()

        R_current = R_new
        # F_current for the next loop is now managed by endorsement_manager

        results_history.append({
            'timeslot': t,
            'interactions': interactions_t.to_dict('records'),
            'R': R_current.copy(),
            'F_used': F_used_in_timeslot,
            'F_next': F_next_timeslot,
            'C': C_current.copy(),
            'penalty_signal_gN': gN_initial_signal.copy(),
            'reward_signal_rP_std': rP_std_initial_signal.copy(),
            'propagated_penalty_input': (1.0 - gN_initial_signal).copy(), # Log what was effectively propagated
            'propagated_reward_input': (rP_std_initial_signal - 1.0).copy(), # Log what was effectively propagated
            'P_cum': P_cumulative.copy(),
            'N_cum': N_cumulative.copy()
        })
        ts_end_time = time.time()
        print(f"Timeslot {t} completed in {ts_end_time - ts_start_time:.4f}s")

    total_end_time = time.time()
    print(f"\n--- Simulation Finished ({NUM_TIMESLOTS} timeslots) ---")
    print(f"Total Simulation Time: {total_end_time - total_start_time:.4f}s")

    print("\n=== Final Reputation Vector ===")
    print(results_history[-1]['R'])
    print("\n=== Final Normalized Endorsement Matrix (F_next) ===")
    print(results_history[-1]['F_next'])

    df_results = pd.DataFrame(results_history)
    rep_df = pd.DataFrame([res['R'] for res in results_history], columns=nodes)
    rep_df['timeslot'] = df_results['timeslot']
    rep_csv_path = os.path.join(OUTPUT_DIR, "reputation_over_time.csv")
    rep_df.to_csv(rep_csv_path, index=False)
    print(f"\nReputation history saved to {rep_csv_path}")

    print(f"\nSaving detailed timeslot data to {TIMESLOT_DETAIL_DIR}...")
    node_labels = [index_to_node[i] for i in range(N)]
    signal_labels = ['Signal Value']
    for result in results_history:
        t = result['timeslot']
        
        df_interactions = pd.DataFrame(result['interactions'])
        df_C = pd.DataFrame(result['C'], index=node_labels, columns=node_labels)
        df_F_used = pd.DataFrame(result['F_used'], index=node_labels, columns=node_labels)
        df_F_next = pd.DataFrame(result['F_next'], index=node_labels, columns=node_labels)
        
        df_gN = pd.DataFrame(result.get('penalty_signal_gN', np.zeros(N)).reshape(-1,1), index=node_labels, columns=signal_labels)
        df_rP_std = pd.DataFrame(result.get('reward_signal_rP_std', np.zeros(N)).reshape(-1,1), index=node_labels, columns=signal_labels)
        df_prop_pen_in = pd.DataFrame(result.get('propagated_penalty_input', np.zeros(N)).reshape(-1,1), index=node_labels, columns=signal_labels)
        df_prop_rew_in = pd.DataFrame(result.get('propagated_reward_input', np.zeros(N)).reshape(-1,1), index=node_labels, columns=signal_labels)

        df_P_cum = pd.DataFrame(result['P_cum'], index=node_labels, columns=node_labels)
        df_N_cum = pd.DataFrame(result['N_cum'], index=node_labels, columns=node_labels)

        interactions_path = os.path.join(TIMESLOT_DETAIL_DIR, f"interactions_t{t}.csv")
        C_path = os.path.join(TIMESLOT_DETAIL_DIR, f"C_matrix_t{t}.csv")
        F_used_path = os.path.join(TIMESLOT_DETAIL_DIR, f"F_matrix_used_t{t}.csv")
        F_next_path = os.path.join(TIMESLOT_DETAIL_DIR, f"F_matrix_next_t{t}.csv")
        gN_path = os.path.join(TIMESLOT_DETAIL_DIR, f"signal_gN_t{t}.csv")
        rP_std_path = os.path.join(TIMESLOT_DETAIL_DIR, f"signal_rP_std_t{t}.csv") # Standard r(P)
        prop_pen_in_path = os.path.join(TIMESLOT_DETAIL_DIR, f"propagated_penalty_input_t{t}.csv")
        prop_rew_in_path = os.path.join(TIMESLOT_DETAIL_DIR, f"propagated_reward_input_t{t}.csv")
        P_cum_path = os.path.join(TIMESLOT_DETAIL_DIR, f"P_cumulative_t{t}.csv")
        N_cum_path = os.path.join(TIMESLOT_DETAIL_DIR, f"N_cumulative_t{t}.csv")

        if not df_interactions.empty or t != -1 : df_interactions.to_csv(interactions_path, index=False)
        df_C.to_csv(C_path, index=True)
        df_F_used.to_csv(F_used_path, index=True)
        df_F_next.to_csv(F_next_path, index=True)
        df_gN.to_csv(gN_path, index=True)
        df_rP_std.to_csv(rP_std_path, index=True)
        df_prop_pen_in.to_csv(prop_pen_in_path, index=True)
        df_prop_rew_in.to_csv(prop_rew_in_path, index=True)
        df_P_cum.to_csv(P_cum_path, index=True)
        df_N_cum.to_csv(N_cum_path, index=True)
    print("Detailed data saved.")

    plt.figure(figsize=(10, 6))
    for node in nodes:
        plt.plot(rep_df['timeslot'], rep_df[node], marker='o', linestyle='-', label=node)
    plt.title('Reputation Evolution Over Time (Corrected Signals & Core Modules)')
    plt.xlabel('Timeslot')
    plt.ylabel('Reputation Score')
    plt.xticks(range(-1, NUM_TIMESLOTS))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "reputation_evolution.png")
    plt.savefig(plot_path)
    print(f"Reputation evolution plot saved to {plot_path}")

if __name__ == "__main__":
    main()
