# src/utils/visualization.py
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
from mplsoccer import Pitch
import random
from matplotlib.gridspec import GridSpec
import networkx as nx
from src.config import PITCH_WIDTH, PITCH_LENGTH
import matplotlib



def draw_pitch_graph(
        player_positions,
        edges,
        edge_weight_attr="weight", node_cluster=None,
        centrality=None, centrality_metric=None,
        title=None, output_path=None, figsize=(12, 8), edge_color="blue", node_color="skyblue", ax=None
):
    """
    Draw a soccer pitch graph with nodes and edges, with optional centrality visualization.

    Parameters:
        player_positions (dict): Dictionary of player positions {node: (x, y)}.
        edges (list): List of edges with attributes [(node1, node2, attr_dict)].
        edge_weight_attr (str): Attribute to use for edge thickness. Default is "weight".
        node_cluster (dict): Optional cluster assignment for each node {node: cluster_id}.
        centrality (dict): Dictionary of centrality values for each player {node: centrality_value}.
        centrality_metric (str): Name of the centrality metric (for title and visualization).
        title (str): Title for the plot.
        figsize (tuple): Figure size for the plot.
        edge_color (str): Color for the edges.
        node_color (str): Default color for nodes (ignored if node_cluster is provided).
        ax (matplotlib.axes._subplots.AxesSubplot): Axis to plot on (optional).

    Returns:
        matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot: The plot figure and axes.
    """
    # Create the pitch and figure if no axis is provided
    if ax is None:
        pitch = Pitch(pitch_length=PITCH_LENGTH, pitch_width=PITCH_WIDTH, line_color="black")
        fig, ax = plt.subplots(figsize=figsize)
        pitch.draw(ax=ax)
    else:
        pitch = Pitch(pitch_length=PITCH_LENGTH, pitch_width=PITCH_WIDTH, line_color="black")
        pitch.draw(ax=ax)

    # Predefined color palette for clusters
    cluster_colors = plt.cm.tab10.colors if node_cluster else None
    # If centrality is provided, set up the color mapping
    if centrality:
        centrality_values = list(centrality.values())
        vmin = min(centrality_values)
        vmax = max(centrality_values)
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        cmap = cm.plasma  # Use the 'plasma' colormap (you can choose another)
    else:
        norm = None
        cmap = None

    # Draw nodes (players)
    for node, (x, y) in player_positions.items():
        # If centrality is provided, use it for coloring or sizing
        if centrality:
            centrality_value = centrality.get(node, 0)
            color = cmap(norm(centrality_value))  # Color based on centrality value
            size = 500 + centrality_value * 1500  # Size based on centrality
        else:
            color = node_color
            size = 1500  # Default size

        if node_cluster:
            cluster_id = node_cluster[node]
            color = cluster_colors[cluster_id % 10]

        ax.scatter(x, y, s=size, color=color, edgecolors="black", zorder=3)
        ax.text(x, y, node, fontsize=12, ha="center", va="center", weight="bold", zorder=4)

    # Draw edges (passes)
    for player_from, player_to, attributes in edges:
        start_x, start_y = player_positions[player_from]
        end_x, end_y = player_positions[player_to]
        edge_thickness = attributes.get(edge_weight_attr, 1)  # Default weight is 1
        pitch.lines(
            start_x, start_y, end_x, end_y,
            lw=edge_thickness, color=edge_color, alpha=0.7, ax=ax, zorder=2
        )

    # Add colorbar if centrality is visualized
    if centrality:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Empty array for colorbar to work
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label(f'{centrality_metric} Value', fontsize=12)

    # Add cluster legend if clusters are provided
    if node_cluster:
        unique_clusters = sorted(set(node_cluster.values()))  # Get unique cluster IDs
        legend_patches = [
            mpatches.Patch(color=cluster_colors[cluster_id % 10], label=f"Cluster {cluster_id}")
            for cluster_id in unique_clusters
        ]
        ax.legend(
            handles=legend_patches,
            title="Clusters",
            loc="upper right",
            fontsize=10,
            title_fontsize=12
        )

    # Set plot title
    if title:
        ax.set_title(title, fontsize=16)

    plt.tight_layout()

    # Save the plot if an output path is provided
    if output_path:
        plt.savefig(output_path)
    # plt.close()

    return ax



def visualize_temporal_motifs(interval_df, triangles, squares, output_path):

    # Create a figure with two subplots (one for triangles, one for squares)
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))  # 1 row, 2 columns
    pitch_length, pitch_width = 144, 96

    # Function to generate random color
    def random_color():
        return [random.random() for _ in range(3)]  # RGB values between 0 and 1

    # Function to draw shapes
    def plot_shapes_on_pitch(ax, shapes, shape_type):
        pitch = Pitch(pitch_length=pitch_length, pitch_width=pitch_width, line_color="black", line_zorder=2)
        pitch.draw(ax=ax)
        ax.set_title(f"{shape_type.capitalize()} Visualization", fontsize=16)

        # Create a dictionary to map player names and timestamps to their positions
        player_positions = {}
        for _, row in interval_df.iterrows():
            player = row["lastName"]
            timestamp = row["eventSec"]
            if player not in player_positions:
                player_positions[player] = {}
            player_positions[player][timestamp] = (row["x"], row["y"])

        # Plot shapes (triangles or squares)
        for shape in shapes:
            edges = []
            timestamps = []
            for edge in shape:
                player_from, player_to, timestamp = edge
                edges.append((player_from, player_to))
                timestamps.append(timestamp)

            x_coords = []
            y_coords = []
            plotted_players = set()
            for i, (player_from, player_to) in enumerate(edges):
                timestamp = timestamps[i]
                if player_from in player_positions and timestamp in player_positions[player_from]:
                    x, y = player_positions[player_from][timestamp]
                    x_coords.append(x)
                    y_coords.append(y)
                    if player_from not in plotted_players:
                        ax.text(x, y, player_from, fontsize=8, ha='center', color='black')
                        plotted_players.add(player_from)
                if player_to in player_positions and timestamp in player_positions[player_to]:
                    x, y = player_positions[player_to][timestamp]
                    x_coords.append(x)
                    y_coords.append(y)
                    if player_to not in plotted_players:
                        ax.text(x, y, player_to, fontsize=8, ha='center', color='black')
                        plotted_players.add(player_to)

            # Close the shape by connecting the last player back to the first
            if len(x_coords) > 0 and len(y_coords) > 0:
                x_coords.append(x_coords[0])
                y_coords.append(y_coords[0])
                ax.plot(x_coords, y_coords, marker='o', color=random_color(), linewidth=2, zorder=3)

    # Plot triangles on the first subplot
    plot_shapes_on_pitch(axes[0], triangles, "triangles")

    # Plot squares on the second subplot
    plot_shapes_on_pitch(axes[1], squares, "squares")

    # Adjust layout and show the combined image
    plt.tight_layout()
    plt.suptitle("Accurate and Valid Strictly Unique Edge Temporal Shapes", fontsize=20)
    # plt.show()
    plt.savefig(f"{output_path}/temporal-motifs.png")
    plt.close()


def clustering_visualization(G, louvain_partition, clusters_nodevec, player_positions, output_path):
    """
    Visualize the clustering results for the graph and pitch.
    :param G:
    :param louvain_partition:
    :param clusters_nodevec:
    :param player_positions:
    :param output_path:
    :return:
    """

    # Create a figure and a grid layout for subplots
    fig = plt.figure(figsize=(20, 12))
    grid = GridSpec(2, 2, figure=fig)  # Adjust grid dimensions as needed

    # Louvain Clustering
    num_clusters_louvain = len(set(louvain_partition.values()))

    ax1 = fig.add_subplot(grid[0, 0])
    nx.draw(
        G, node_color=[louvain_partition[node] for node in G.nodes()],
        with_labels=True, cmap=plt.cm.rainbow, ax=ax1
    )
    ax1.set_title(f"Graph Clustering with {num_clusters_louvain} Louvain Algorithm")

    # Node2Vec Clustering
    num_clusters_nodevec = len(set(clusters_nodevec.values()))
    ax2 = fig.add_subplot(grid[0, 1])
    nx.draw(
        G, node_color=[clusters_nodevec[node] for node in G.nodes()],
        with_labels=True, cmap=plt.cm.rainbow, ax=ax2
    )
    ax2.set_title(f"Graph Clustering with {num_clusters_nodevec} Clusters Node2Vec Algorithm")

    # Louvain Pitch Graph
    ax3 = fig.add_subplot(grid[1, 0])
    draw_pitch_graph(
        player_positions=player_positions, edges=G.edges(data=True),
        node_cluster=louvain_partition, title="Pitch Graph with Louvain Clustering",
        output_path=None, ax=ax3
    )

    # Node2Vec Pitch Graph
    ax4 = fig.add_subplot(grid[1, 1])
    draw_pitch_graph(
        player_positions=player_positions, edges=G.edges(data=True),
        node_cluster=clusters_nodevec, title="Pitch Graph with Node2Vec Clustering",
        output_path=None, ax=ax4
    )

    # Adjust layout and save/show
    plt.tight_layout()
    plt.savefig(f"{output_path}/clusters_visualization.png")
    plt.close()
