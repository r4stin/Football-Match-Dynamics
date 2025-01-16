# src/analysis/clustering.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from node2vec import Node2Vec
import matplotlib.pyplot as plt
from community import community_louvain
from src.config import PITCH_WIDTH, PITCH_LENGTH
import itertools
from random import randint
from mplsoccer import Pitch
import os
from collections import defaultdict
from tabulate import tabulate
from src.utils.visualization import clustering_visualization




class GraphClustering:
    def __init__(self, graph, interval_df, output_path):
        self.G = graph
        self.interval_df = interval_df
        self.output_path = output_path + "/clustering"

    def node2vec_clustering(self, centrality_metrics):
        """Perform Node2Vec-based clustering."""
        node2vec = Node2Vec(self.G, dimensions=128, walk_length=40, num_walks=300, workers=4)
        model = node2vec.fit(window=10, min_count=1, batch_words=4)
        embeddings = {
            node: np.concatenate(
                [model.wv[node], [centrality_metrics[metric][node] for metric in centrality_metrics]])
            for node in self.G.nodes()
        }

        embedding_matrix = np.array(list(embeddings.values()))

        # Normalize embeddings
        scaler = StandardScaler()
        embedding_matrix = scaler.fit_transform(embedding_matrix)

        def analyze_silhouette(embedding_matrix):
            silhouette_scores = []
            cluster_range = range(2, 5)
            for n in cluster_range:
                kmeans = KMeans(n_clusters=n, init='k-means++', random_state=42).fit(embedding_matrix)
                score = silhouette_score(embedding_matrix, kmeans.labels_)
                silhouette_scores.append(score)

            plt.figure(figsize=(10, 6))
            plt.plot(cluster_range, silhouette_scores, marker='o', linestyle='--')
            plt.title('Silhouette Analysis for Optimal Number of Clusters')
            plt.xlabel('Number of Clusters')
            plt.ylabel('Silhouette Score')
            plt.savefig(self.output_path + "/silhouette_analysis_node2vec.png")
            plt.close()

            optimal_clusters = cluster_range[np.argmax(silhouette_scores)]
            print(f"Optimal number of clusters based on Silhouette Analysis (Node2Vec): {optimal_clusters}")

            return optimal_clusters

        n_clusters = analyze_silhouette(embedding_matrix)
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42).fit(embedding_matrix)

        clusters = {node: kmeans.labels_[i] for i, node in enumerate(self.G.nodes())}

        clusters_vec = {}
        for player, cluster in clusters.items():
            clusters_vec.setdefault(cluster, []).append(player)

        return clusters, clusters_vec

    def louvain_clustering(self):
        """Perform Louvain-based clustering."""
        G_undirected = self.G.to_undirected()
        louvain_partition = community_louvain.best_partition(G_undirected)

        clusters = {}
        for player, cluster in louvain_partition.items():
            clusters.setdefault(cluster, []).append(player)

        return louvain_partition, clusters

    def update_edge_weights(self, triangle_edges, square_edges):
        """Update edge weights with additional features."""
        for u, v, data in self.G.edges(data=True):
            passes = self.interval_df[(self.interval_df["lastName"] == u) & (self.interval_df["lastName_next"] == v)]
            base_weight = data["weight"]
            key_pass_weight = passes["keyPass"].sum() + 1.5
            assist_weight = passes["assist"].sum() + 2
            dangerous_loss_penalty = passes["dangerous_ball_lost"].sum() - 1

            triangle_bonus = 0.5 if (u, v) in triangle_edges or (v, u) in triangle_edges else 0
            square_bonus = 0.5 if (u, v) in square_edges or (v, u) in square_edges else 0

            enhanced_weight = (
                    base_weight +
                    key_pass_weight +
                    assist_weight +
                    dangerous_loss_penalty +
                    triangle_bonus +
                    square_bonus
            )
            data["weight"] = max(0, enhanced_weight)


    def analyze_clusters_positional_play(self, clusters, output_path
                                         ):
        """
        Analyze positional play by visualizing:
        - Passes within clusters (same cluster).
        - Passes between different clusters, dynamically arranging plots.
        """
        # Map clusters to positions on the pitch
        cluster_positions = {}
        cluster_colors = {}  # Dictionary to store random colors for each cluster
        for cluster_id, players in clusters.items():
            player_positions = self.interval_df[self.interval_df["lastName"].isin(players)][["x", "y"]]
            avg_x, avg_y = player_positions["x"].mean(), player_positions["y"].mean()
            cluster_positions[cluster_id] = (avg_x, avg_y)

            # Generate a random color for this cluster (same color for cluster and its passes)
            cluster_colors[cluster_id] = np.array([randint(0, 255) / 255 for _ in range(3)])  # Random RGB color

        # Calculate number of plots dynamically
        cluster_combinations = list(itertools.combinations(clusters.keys(), 2))
        n_plots = 1 + len(cluster_combinations)  # 1 for all passes + pairs of clusters

        # Determine subplot grid layout
        n_cols = 2  # Fixed number of columns
        n_rows = -(-n_plots // n_cols)  # Ceiling division for rows

        # Create a figure with dynamic layout
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6 * n_rows))
        axes = axes.flatten()  # Flatten to iterate over all axes

        # Clear unused axes if total plots < grid size
        for idx in range(n_plots, len(axes)):
            fig.delaxes(axes[idx])

        # Plot 1: Cluster Passes
        pitch = Pitch(pitch_length=PITCH_LENGTH, pitch_width=PITCH_WIDTH, line_color="black")
        ax = axes[0]
        pitch.draw(ax=ax)

        # Scatter all player positions (passes)
        ax.scatter(
            self.interval_df["x"], self.interval_df["y"], alpha=0.2, color="gray", label="All Passes", zorder=1
        )

        # Plot cluster centers and annotate them
        for cluster_id, (avg_x, avg_y) in cluster_positions.items():
            cluster_color = cluster_colors[cluster_id]  # Get color for this cluster
            ax.scatter(avg_x, avg_y, color=cluster_color, s=300, label=f"Cluster {cluster_id}", zorder=3)
            ax.text(avg_x, avg_y, str(cluster_id), fontsize=10, color="white", ha="center", va="center", weight="bold")

        # Plot passes for each cluster with the same random color
        for cluster_id, players in clusters.items():
            cluster_color = cluster_colors[cluster_id]  # Get the random color for the cluster
            # Filter passes for players in this cluster
            cluster_passes = self.interval_df[
                (self.interval_df["lastName"].isin(players)) & (self.interval_df["lastName_next"].isin(players))
                ]

            # Draw arrows for passes within the cluster with the generated random color
            for _, row in cluster_passes.iterrows():
                ax.arrow(
                    row["x"], row["y"],  # Start point
                    row["end_x"] - row["x"], row["end_y"] - row["y"],  # Vector (dx, dy)
                    head_width=1.5, head_length=2.5, color=cluster_color, alpha=0.7, zorder=2
                )

        ax.set_title("Cluster Passes", fontsize=16)
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

        # Plot 2 onwards: Non-cluster Passes for each pair of clusters
        for idx, (start_cluster, end_cluster) in enumerate(cluster_combinations, start=1):
            ax = axes[idx]
            pitch.draw(ax=ax)

            start_players = clusters[start_cluster]
            end_players = clusters[end_cluster]

            # Passes from start_cluster to end_cluster
            non_cluster_passes = self.interval_df[
                (self.interval_df["lastName"].isin(start_players)) & (self.interval_df["lastName_next"].isin(end_players))
                ]
            for _, row in non_cluster_passes.iterrows():
                ax.arrow(
                    row["x"], row["y"],  # Start point
                    row["end_x"] - row["x"], row["end_y"] - row["y"],  # Vector (dx, dy)
                    head_width=1.5, head_length=2.5, color="orange", alpha=0.7, zorder=2
                )

            # Passes from end_cluster to start_cluster
            non_cluster_passes_reverse = self.interval_df[
                (self.interval_df["lastName"].isin(end_players)) & (self.interval_df["lastName_next"].isin(start_players))
                ]
            for _, row in non_cluster_passes_reverse.iterrows():
                ax.arrow(
                    row["x"], row["y"],  # Start point
                    row["end_x"] - row["x"], row["end_y"] - row["y"],  # Vector (dx, dy)
                    head_width=1.5, head_length=2.5, color="purple", alpha=0.7, zorder=2
                )

            # Add custom legend
            ax.legend(
                [plt.Line2D([0], [0], color='orange', lw=4), plt.Line2D([0], [0], color='purple', lw=4)],
                ["Cluster {} → Cluster {}".format(start_cluster, end_cluster),
                 "Cluster {} → Cluster {}".format(end_cluster, start_cluster)],
                loc="upper left", bbox_to_anchor=(1, 1)
            )

            # Add title for this subplot
            ax.set_title(f"Passes: Cluster {start_cluster} ↔ Cluster {end_cluster}", fontsize=14)

        # Add text description below all plots
        fig.text(
            0.5, 0.1,
            "Cluster Passes are visualized with distinct colors.\nOrange arrows: passes between clusters; Purple arrows: reverse direction.",
            ha="center", va="center", fontsize=12
        )

        # Adjust layout to fit all plots and text
        plt.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
        # plt.show()
        plt.savefig(output_path)
        plt.close()

    def analyze_cluster_metrics(self, clusters, centrality_metrics):
        """
        Analyze metrics for detected clusters in a football passing network.

        Parameters:
            clusters (dict): Mapping of node to cluster (e.g., {node: cluster_label}).
            centrality_metrics (dict): Dictionary of centrality metrics.

        Returns:
            pd.DataFrame: Cluster statistics summary.
        """
        # Group nodes by cluster
        cluster_nodes = defaultdict(list)
        for node, cluster in clusters.items():
            cluster_nodes[cluster].append(node)

        # cluster_metrics = {}
        cluster_metrics = []

        # Analyze each cluster
        for cluster, nodes in cluster_nodes.items():
            subgraph = self.G.subgraph(nodes)

            # Cluster Size
            cluster_size = len(nodes)

            # Intra-cluster pass density
            intra_cluster_edges = len(subgraph.edges)
            max_possible_edges = cluster_size * (cluster_size - 1) / 2
            pass_density = intra_cluster_edges / max_possible_edges if max_possible_edges > 0 else 0

            # Average centrality values
            avg_centrality = {metric: sum(centrality_metrics[metric][node] for node in nodes) / cluster_size
                              for metric in centrality_metrics}

            # Inter-cluster interactions
            inter_cluster_edges = 0
            inter_cluster_weight = 0
            for node in nodes:
                for neighbor in self.G.neighbors(node):
                    if clusters[neighbor] != cluster:  # Check if the neighbor belongs to a different cluster
                        inter_cluster_edges += 1
                        inter_cluster_weight += self.G[node][neighbor].get("weight", 0)

            # Passing statistics within the cluster
            passes_within_cluster = self.interval_df[
                self.interval_df['lastName'].isin(nodes) & self.interval_df['lastName_next'].isin(nodes)
                ]
            total_passes = passes_within_cluster.shape[0]
            key_passes = passes_within_cluster["keyPass"].sum()
            assists = passes_within_cluster["assist"].sum()

            # Store metrics
            cluster_metrics.append({
                "Cluster": cluster,
                "Size": cluster_size,
                "Pass Density": pass_density,
                "Inter-Cluster Edges": inter_cluster_edges,
                "Inter-Cluster Weight": inter_cluster_weight,
                "Total Passes Within Cluster": total_passes,
                "Key Passes Within Cluster": key_passes,
                "Assists Within Cluster": assists,
                **avg_centrality  # Unpack centrality metrics
            })

        # Convert results to DataFrame for better visualization
        cluster_metrics_df = pd.DataFrame(cluster_metrics)

        return cluster_metrics_df

    def clustering_and_tactical_insights(self, player_positions, centrality_metrics):
        """Perform clustering and provide tactical insights."""
        # self.update_edge_weights(interval_df, player_positions, triangle_edges, square_edges,
        #                          centrality_metrics)

        louvain_partition, louvain_clusters = self.louvain_clustering()
        raw_node2vec_clusters, node2vec_clusters = self.node2vec_clustering(centrality_metrics)

        cluster_metrics_df_nodevec = self.analyze_cluster_metrics(
            clusters=raw_node2vec_clusters,
            centrality_metrics=centrality_metrics,
        )

        cluster_metrics_df_louvain = self.analyze_cluster_metrics(
            clusters=louvain_partition,
            centrality_metrics=centrality_metrics,
        )

        clustering_result = {
            "Louvain": louvain_clusters,
            "Node2Vec": node2vec_clusters
        }

        clustering_metrics = {
            "Louvain": cluster_metrics_df_louvain,
            "Node2Vec": cluster_metrics_df_nodevec
        }

        self.save_clustering_results(clustering_result, clustering_metrics)

        clustering_visualization(
            self.G,
            louvain_partition,
            raw_node2vec_clusters,
            player_positions,
            self.output_path
        )

        louvain_output = f"{self.output_path}/inter-intra-cluster-passes_louvain.png"
        self.analyze_clusters_positional_play(louvain_clusters, louvain_output)

        node2vec_output = f"{self.output_path}/inter-intra-cluster-passes_node2vec.png"
        self.analyze_clusters_positional_play(node2vec_clusters, node2vec_output)

    def save_clustering_results(self, clustering_result, cluster_metrics):
        """
        Save the clustering results to a txt file.

        Parameters:
            clustering_result (dict): Dictionary of clustering results of Louvain and Node2Vec Algorithms.
            cluster_metrics (dict): Dictionary of clustering metrics for each method.

        """

        os.makedirs(self.output_path, exist_ok=True)
        file_path = os.path.join(self.output_path, "clustering_results.txt")

        with open(file_path, "w") as file:
            for method, results in clustering_result.items():
                file.write(f"{method} Clustering Results:\n")
                file.write("=" * 30 + "\n\n")
                for cluster_id, players in results.items():
                    file.write(f"Cluster {cluster_id}:\n")
                    file.write("- " + "\n- ".join(players) + "\n")
                    file.write("\n")
                file.write("\n\n")
                file.write(f"{method} Clustering Metrics:\n\n")
                table_string = tabulate(cluster_metrics[method], headers='keys', tablefmt='grid', showindex=False)
                file.write(table_string)
                file.write("\n\n")
        # print(f"Clustering results saved to: {file_path}")

