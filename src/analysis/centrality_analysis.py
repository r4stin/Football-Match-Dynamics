#!/usr/bin/env python
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from src.utils.visualization import draw_pitch_graph
from src.config import PITCH_LENGTH, PITCH_WIDTH
import os
from tabulate import tabulate


def save_centerality_metrics(centrality_metrics, output_path):
    """
    Save the centrality metrics to a txt file.
        :param output_path: Path to save the txt file.
        :param centrality_metrics: Dictionary of centrality values for each player.
    """
    metrics_output_path = output_path + "/metrics/centrality_metrics.txt"

    with open(metrics_output_path, "w") as file:
        for metric_name, values in centrality_metrics.items():
            # Prepare the data as a table
            sorted_players = sorted(values.items(), key=lambda x: x[1], reverse=True)
            table = [[player, f"{score:.4f}"] for player, score in sorted_players]

            # Write metric name and table to the file
            file.write(f"\n{metric_name}:\n")
            file.write(tabulate(table, headers=["Player", "Score"], tablefmt="grid"))
            file.write("\n\n")
    # print(f"Centrality metrics saved to: {metrics_output_path}")


def compute_centrality_matrix(graph, player_positions, output_path):
    """
    Compute centrality metrics for the given graph.
    :param output_path:
    :param player_positions:
    :param graph:
    :return: centrality_metrics: dict
    """
    centrality_metrics = {
        "Degree Centrality": nx.degree_centrality(graph),
        "Closeness Centrality": nx.closeness_centrality(graph),
        "Betweenness Centrality": nx.betweenness_centrality(graph, weight="weight"),
        "Eigenvector Centrality": nx.eigenvector_centrality(graph, weight="weight"),
        "PageRank": nx.pagerank(graph),
        "Clustering Coefficient": nx.clustering(graph.to_undirected()),
    }

    save_centerality_metrics(centrality_metrics, output_path)

    # Define the grid layout
    n_metrics = len(centrality_metrics)
    cols = 2  # Number of columns in the grid
    rows = (n_metrics + cols - 1) // cols  # Compute required rows

    fig = plt.figure(figsize=(15, rows * 6))
    grid = GridSpec(rows, cols, figure=fig)

    # Plot each centrality metric in a subplot
    for i, (metric_name, values) in enumerate(centrality_metrics.items()):
        ax = fig.add_subplot(grid[i // cols, i % cols])
        edges = graph.edges(data=True)

        # Use the existing draw_pitch_graph function with centrality data
        draw_pitch_graph(
            player_positions=player_positions,
            edges=edges,
            centrality=values,
            centrality_metric=metric_name,
            title=f"Players by {metric_name}",
            output_path=None,  # Do not save individual plots
            edge_color="blue",
            ax=ax  # Pass the subplot axis
        )

    # Adjust layout and save the combined image
    plt.tight_layout()
    combined_image_path = output_path + "/metrics/centrality_metrics.png"
    plt.savefig(combined_image_path)
    plt.close()
    # plt.show()

    return centrality_metrics
