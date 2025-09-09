# main_simulation_4nodes_feedback_based.py

import numpy as np
import pandas as pd
from core.config import EPSILON
from core.reputation_forward_propagation import ReputationForwardPropagation
from core.reputation_backward_propagation import ReputationBackwardPropagation
from core.reputation_normalisation import ReputationNormalizer
from core.trustworthiness import TrustworthinessCalculator
from core.endorsement_penalty import EndorsementPenalty
from core.endorsement_reward import EndorsementReward
from core.endorsement_manager import EndorsementManager
from core.node_manager import NodeManager
from core.interaction_manager import InteractionManager

def generate_random_endorsement_matrix(size):
    mat = np.random.rand(size, size).astype(np.float32)
    np.fill_diagonal(mat, 0)
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return mat / row_sums

def generate_random_interactions(nodes, num_events=10):
    interactions = []
    for _ in range(num_events):
        src, dst = np.random.choice(nodes, 2, replace=False)
        interaction = {
            "src": src,
            "dst": dst,
            "rating": np.random.randint(1, 6),
            "timestamp": np.random.randint(1_600_000_000, 1_700_000_000),
            "month": np.random.randint(1, 13),
        }
        interactions.append(interaction)
    return pd.DataFrame(interactions)

def build_feedback_matrices(interactions_df, node_to_index, N):
    pos_matrix = np.zeros((N, N), dtype=np.float32)
    neg_matrix = np.zeros((N, N), dtype=np.float32)
    for _, row in interactions_df.iterrows():
        src, dst, rating = row["src"], row["dst"], row["rating"]
        if src == dst:
            continue  # Skip self-interactions
        i, j = node_to_index[src], node_to_index[dst]
        if rating >= 3:
            pos_matrix[i, j] += 1
        else:
            neg_matrix[i, j] += 1
    return pos_matrix, neg_matrix

def main():
    np.random.seed(42)

    nodes = ["A", "B", "C", "D"]
    node_to_index = {node: idx for idx, node in enumerate(nodes)}
    N = len(nodes)

    interactions_df = generate_random_interactions(nodes, num_events=12)
    pos_matrix, neg_matrix = build_feedback_matrices(interactions_df, node_to_index, N)
    endorsement_matrix = generate_random_endorsement_matrix(N)

    print("=== Interactions ===")
    print(interactions_df)
    print("\n=== Positive Feedback Matrix ===")
    print(pos_matrix)
    print("\n=== Negative Feedback Matrix ===")
    print(neg_matrix)
    print("\n=== Endorsement Matrix ===")
    print(endorsement_matrix)

    node_manager = NodeManager()
    node_manager.load_nodes(nodes)
    interaction_manager = InteractionManager(interactions_df, nodes, node_to_index)
    endorsement_manager = EndorsementManager(endorsement_matrix, nodes, node_to_index)

    for node_id in node_manager.nodes:
        idx = node_to_index[node_id]
        node_manager.nodes[node_id].add_endorsement_change({
            "event": "initial_endorsement",
            "values": endorsement_matrix[idx, :].tolist()
        })

    R0 = np.ones(N, dtype=np.float32) / N
    normalizer = ReputationNormalizer(EPSILON)
    R0 = normalizer.normalize(R0)

    print("\n=== Initial Uniform Reputation Vector ===")
    print(R0)

    for node_id in node_manager.nodes:
        idx = node_to_index[node_id]
        node_manager.nodes[node_id].add_reputation_snapshot(R0[idx])

    C_zero = np.zeros((N, N), dtype=np.float32)
    rep_forward_init = ReputationForwardPropagation(C_zero, endorsement_matrix)
    R_initial_updated = normalizer.normalize(rep_forward_init.compute_reputation(R0))

    print("\n=== Reputation After Initial Endorsement Integration ===")
    print(R_initial_updated)

    for node_id in node_manager.nodes:
        idx = node_to_index[node_id]
        node_manager.nodes[node_id].add_reputation_snapshot(R_initial_updated[idx])

    for _, row in interactions_df.iterrows():
        event = row.to_dict()
        if event["src"] in node_manager.nodes:
            node_manager.nodes[event["src"]].add_interaction({"type": "sent", **event})
        if event["dst"] in node_manager.nodes:
            node_manager.nodes[event["dst"]].add_interaction({"type": "received", **event})

    trust_calc = TrustworthinessCalculator(pos_matrix, neg_matrix, epsilon=EPSILON)
    C = trust_calc.compute_local_trust()
    print("\n=== Local Trust Matrix C ===")
    print(C)

    neg_feedback_vector = np.sum(neg_matrix, axis=0)
    pos_feedback_vector = np.sum(pos_matrix, axis=0)
    ep = EndorsementPenalty(endorsement_matrix, neg_feedback_vector)
    er = EndorsementReward(endorsement_matrix, pos_feedback_vector)

    penalty_signal = ep.compute_penalty_signal()
    reward_signal = er.compute_reward_signal()

    print("\n=== Penalty Signal ===")
    print(penalty_signal)
    print("\n=== Reward Signal ===")
    print(reward_signal)

    old_endorsement_matrix = endorsement_matrix.copy()
    endorsement_manager.update_endorsements_with_penalty(penalty_signal)
    endorsement_manager.update_endorsements_with_reward(reward_signal)
    updated_endorsement_matrix = endorsement_manager.endorsement_matrix

    print("\n=== Updated Endorsement Matrix ===")
    print(updated_endorsement_matrix)

    for node_id in node_manager.nodes:
        idx = node_to_index[node_id]
        old_row = old_endorsement_matrix[idx, :].tolist()
        new_row = updated_endorsement_matrix[idx, :].tolist()
        if old_row != new_row:
            node_manager.nodes[node_id].add_endorsement_change({
                "event": "endorsement_update",
                "old_values": old_row,
                "new_values": new_row,
                "trigger": "Penalty/Reward update"
            })

    rep_forward = ReputationForwardPropagation(C, updated_endorsement_matrix)
    R_forward = normalizer.normalize(rep_forward.compute_reputation(R_initial_updated))

    print("\n=== Reputation Vector After Forward Propagation ===")
    print(R_forward)

    for node_id in node_manager.nodes:
        idx = node_to_index[node_id]
        node_manager.nodes[node_id].add_reputation_snapshot(R_forward[idx])

    penalty_vector = ep.compute_penalty()
    reward_vector = er.compute_reward()

    print("\n=== Penalty Vector ===")
    print(penalty_vector)
    print("\n=== Reward Vector ===")
    print(reward_vector)

    rbp = ReputationBackwardPropagation()
    R_final = normalizer.normalize(rbp.apply_backward_propagation(R_forward, penalty_vector, reward_vector))

    print("\n=== Final Reputation Vector ===")
    print(R_final)

    for node_id in node_manager.nodes:
        idx = node_to_index[node_id]
        node_manager.nodes[node_id].add_reputation_snapshot(R_final[idx])

    print("\n=== Final Node States ===")
    node_manager.display_all_nodes()

if __name__ == "__main__":
    main()
