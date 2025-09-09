# main.py

import numpy as np
import json
import os
import pandas as pd
from core.config import EPSILON, ALPHA
from core.reputation_forward_propagation import ReputationForwardPropagation
from core.reputation_backward_propagation import ReputationBackwardPropagation
from core.reputation_normalisation import ReputationNormalizer
from core.trustworthiness import TrustworthinessCalculator
from core.endorsement_penalty import EndorsementPenalty
from core.endorsement_reward import EndorsementReward
from core.endorsement_manager import EndorsementManager
from core.node_manager import NodeManager
from core.interaction_manager import InteractionManager

def load_processed_data(dataset_folder: str):
    """
    Loads the preprocessed data from the dataset folder.
    
    Expected files:
      - combined_data_positive_feedback.npz (key: pos_matrix)
      - combined_data_negative_feedback.npz (key: neg_matrix)
      - combined_data_endorsement.npz (key: endorsement_matrix)
      - combined_data_nodes.json
      - combined_data_node_to_index.json
    
    Returns:
        pos_matrix (np.ndarray): Dense positive feedback matrix.
        neg_matrix (np.ndarray): Dense negative feedback matrix.
        endorsement_matrix (np.ndarray): Dense, normalized endorsement matrix.
        nodes (List): List of node identifiers.
        node_to_index (Dict): Mapping from node id to index.
    """
    pos_path = os.path.join(dataset_folder, "combined_data_positive_feedback.npz")
    neg_path = os.path.join(dataset_folder, "combined_data_negative_feedback.npz")
    endorsement_path = os.path.join(dataset_folder, "combined_data_endorsement.npz")
    nodes_path = os.path.join(dataset_folder, "combined_data_nodes.json")
    mapping_path = os.path.join(dataset_folder, "combined_data_node_to_index.json")
    
    pos_data = np.load(pos_path)
    neg_data = np.load(neg_path)
    endorsement_data = np.load(endorsement_path)
    
    pos_matrix = pos_data["pos_matrix"]
    neg_matrix = neg_data["neg_matrix"]
    endorsement_matrix = endorsement_data["endorsement_matrix"]
    
    with open(nodes_path, "r") as f:
        nodes = json.load(f)
    with open(mapping_path, "r") as f:
        node_to_index = json.load(f)
    
    return pos_matrix, neg_matrix, endorsement_matrix, nodes, node_to_index

def load_interactions_data(dataset_folder: str) -> pd.DataFrame:
    """
    Loads the full interaction data from the CSV file.

    The file is expected to contain the following columns:
       "src", "dst", "rating", "timestamp", "month"

    Returns:
        pd.DataFrame: Interaction data with "src" and "dst" columns as strings.
    """
    interactions_path = os.path.join(dataset_folder, "combined_interactions.csv")
    if os.path.exists(interactions_path):
        df = pd.read_csv(interactions_path)
        df["src"] = df["src"].astype(str)
        df["dst"] = df["dst"].astype(str)
    else:
        # Create an empty DataFrame with the required columns if file is missing.
        df = pd.DataFrame(columns=["src", "dst", "rating", "timestamp", "month"])
    return df


def main():
    dataset_folder = "datasets/processed_dataset"
    
    # Load processed matrices and node data.
    pos_matrix, neg_matrix, endorsement_matrix, nodes, node_to_index = load_processed_data(dataset_folder)
    interactions_df = load_interactions_data(dataset_folder)

    print(interactions_df.head())
    N = len(nodes)
    print(f"Loaded processed data for {N} nodes.")
    
    # Ensure all node identifiers are strings.
    nodes = [str(n) for n in nodes]
    node_to_index = {str(k): v for k, v in node_to_index.items()}
    
    # Instantiate NodeManager and load nodes.
    node_manager = NodeManager()
    node_manager.load_nodes(nodes)
    print("Active nodes (from NodeManager):")
    node_manager.display_active_nodes()
    
    # Instantiate InteractionManager with the full interactions data.
    interaction_manager = InteractionManager(interactions_df, nodes, node_to_index)


    # Instantiate EndorsementManager.
    endorsement_manager = EndorsementManager(endorsement_matrix, nodes, node_to_index)
    
    # Step 0: Store initial endorsement information to each node.
    for node_id in node_manager.nodes:
        idx = node_to_index[str(node_id)]
        init_endorsement = endorsement_manager.endorsement_matrix[idx, :].tolist()
        node_manager.nodes[node_id].add_endorsement_change({
            "event": "initial_endorsement",
            "values": init_endorsement
        })
    
    # Step 1: Initialize uniform reputation vector and normalize.
    R0 = np.ones(N, dtype=np.float32) / N
    normalizer = ReputationNormalizer(EPSILON)
    R0 = normalizer.normalize(R0)
    print("Initial uniform reputation vector (normalized):")
    print(R0)

    # Store initial reputation snapshot into each node.
    for node_id in node_manager.nodes:
        idx = node_to_index[str(node_id)]
        node_manager.nodes[node_id].add_reputation_snapshot(R0[idx])
    
    # Step 2: Update initial reputation using endorsement only (using zero trust matrix).
    C_zero = np.zeros((N, N), dtype=np.float32)
    rep_forward_init = ReputationForwardPropagation(C_zero, endorsement_manager.endorsement_matrix)
    R_initial_updated = rep_forward_init.compute_reputation(R0)
    R_initial_updated = normalizer.normalize(R_initial_updated)
    print("Reputation after initial endorsement integration (normalized):")
    print(R_initial_updated)
    
    # Store initial reputation snapshot into each node.
    for node_id in node_manager.nodes:
        idx = node_to_index[str(node_id)]
        node_manager.nodes[node_id].add_reputation_snapshot(R_initial_updated[idx])
    
    # Step 3: Record interaction events to each node before trust update.
    for _, row in interaction_manager.interactions.iterrows():
        event = row.to_dict()
        if event["src"] in node_manager.nodes:
            node_manager.nodes[event["src"]].add_interaction({"type": "sent", **event})
        if event["dst"] in node_manager.nodes:
            node_manager.nodes[event["dst"]].add_interaction({"type": "received", **event})
    
    # Step 4: Compute local trust matrix from first-month interactions.
    trust_calc = TrustworthinessCalculator(pos_matrix, neg_matrix, epsilon=EPSILON)
    C = trust_calc.compute_local_trust()
    print("Local trust matrix C (from first-month interactions):")
    print(C)
    
    # Step 5: Update the endorsement matrix using penalty/reward signals BEFORE forward propagation.
    from core.endorsement_penalty import EndorsementPenalty
    from core.endorsement_reward import EndorsementReward
    # Use column sums of the matrices for feedback counts.
    neg_feedback_vector = np.sum(neg_matrix, axis=0)
    pos_feedback_vector = np.sum(pos_matrix, axis=0)
    
    ep = EndorsementPenalty(endorsement_manager.endorsement_matrix, neg_feedback_vector)
    er = EndorsementReward(endorsement_manager.endorsement_matrix, pos_feedback_vector)
    penalty_signal = ep.compute_penalty_signal()
    reward_signal = er.compute_reward_signal()
    
    old_endorsement_matrix = endorsement_manager.endorsement_matrix.copy()
    endorsement_manager.update_endorsements_with_penalty(penalty_signal)
    endorsement_manager.update_endorsements_with_reward(reward_signal)
    updated_endorsement_matrix = endorsement_manager.endorsement_matrix
    print("Updated endorsement matrix after applying penalty and reward signals:")
    print(updated_endorsement_matrix)
    
    # Record endorsement update events to each node.
    for node_id, node in node_manager.nodes.items():
        idx = node_to_index[str(node_id)]
        old_row = old_endorsement_matrix[idx, :].tolist()
        new_row = updated_endorsement_matrix[idx, :].tolist()
        if old_row != new_row:
            node.add_endorsement_change({
                "event": "endorsement_update",
                "old_values": old_row,
                "new_values": new_row,
                "trigger": "Penalty/Reward update"
            })
    
    # Step 6: Forward propagation using updated endorsement matrix and local trust (C).
    rep_forward = ReputationForwardPropagation(C, updated_endorsement_matrix)
    R_forward = rep_forward.compute_reputation(R_initial_updated)
    R_forward = normalizer.normalize(R_forward)
    print("Reputation vector after forward propagation (normalized):")
    print(R_forward)
    
    # Store forward reputation snapshot in each node.
    for node_id in node_manager.nodes:
        idx = node_to_index[str(node_id)]
        node_manager.nodes[node_id].add_reputation_snapshot(R_forward[idx])
    
    # Step 7: Compute endorsement penalty and reward vectors using updated endorsement matrix.
    penalty_vector = ep.compute_penalty()  # using updated endorsement matrix
    reward_vector  = er.compute_reward()
    print("Computed endorsement penalty vector:")
    print(penalty_vector)
    print("Computed endorsement reward vector:")
    print(reward_vector)
    
    # Step 8: Apply backward propagation to adjust reputation.
    from core.reputation_backward_propagation import ReputationBackwardPropagation
    rbp = ReputationBackwardPropagation()
    R_backward = rbp.apply_backward_propagation(R_forward, penalty_vector, reward_vector)
    R_final = normalizer.normalize(R_backward)
    print("Final reputation vector after backward propagation (normalized):")
    print(R_final)
    
    # Store final reputation snapshot in each node.
    for node_id in node_manager.nodes:
        idx = node_to_index[str(node_id)]
        node_manager.nodes[node_id].add_reputation_snapshot(R_final[idx])
    
    # (Optional) Display detailed node information.
    print("\nFinal Node States:")
    node_manager.display_all_nodes()
    
if __name__ == "__main__":
    main()
