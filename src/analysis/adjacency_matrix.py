# src/analysis/adjacency_matrix.py
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns


def compute_adjacency_matrix(graph, output_path, adj_title):
    """
    Compute and save adjacency matrices for each half and interval.
    """
    adj_output_path = output_path + "/network/adjacency_matrix_heatmap.png"

    # Compute adjacency matrix
    adj_matrix = nx.adjacency_matrix(graph).todense()
    player_names = list(graph.nodes)

    plt.figure(figsize=(8, 6))
    sns.heatmap(np.array(adj_matrix), annot=True, cmap="Blues", cbar=True, fmt="g",
                xticklabels=player_names, yticklabels=player_names)
    plt.title(adj_title)
    plt.xlabel("Players (Pass Recipients)")
    plt.ylabel("Players (Pass Initiators)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(adj_output_path)
    plt.close()

    # print(f"Saved adjacency matrix heatmap to {output_path}.")
