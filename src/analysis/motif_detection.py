# src/analysis/motif_detection.py
from itertools import combinations
import networkx as nx
from networkx.algorithms.isomorphism import DiGraphMatcher
import os
from src.utils.visualization import visualize_temporal_motifs
from src.utils.output_manager import generate_file_path, load_yaml_config

class MotifAnalyzer:
    def __init__(self, G, output_path):
        self.G = G
        self.output_path = output_path + "/motifs"

    def count_3_motifs(self):
        """
        Counts the occurrences of 3-node motifs in a directed graph.

        Parameters:
            graph (nx.DiGraph): A directed graph.

        Returns:
            dict: A dictionary with motif counts for each of the 13 motif types.
        """

        motif_templates = {
            "3-cycle": nx.DiGraph([(0, 1), (1, 2), (2, 0)]),  # Directed 3-cycle (cyclic triangle)
            "Acyclic Triangle": nx.DiGraph([(0, 1), (1, 2), (2, 1)]),  # Acyclic directed triangle
            "Chain": nx.DiGraph([(0, 1), (1, 2)]),  # Chain (2 directed edges)
            "Out-star": nx.DiGraph([(0, 1), (0, 2)]),  # Out-star (1 node points to 2 others)
            "In-star": nx.DiGraph([(1, 0), (2, 0)]),  # In-star (2 nodes point to 1)
            "Fork": nx.DiGraph([(0, 1), (0, 2)]),  # Fork (one central node with 2 out edges)
            "2-cycle": nx.DiGraph([(0, 1), (1, 0)]),  # 2-cycle (bidirectional)
            "Disjoint": nx.DiGraph([(0, 1), (2, 3)]),  # Disjoint edges
            "Star": nx.DiGraph([(0, 1), (0, 2), (0, 3)]),  # Star (one central node with 3 out edges)
            "Transitive Triad": nx.DiGraph([(0, 1), (1, 2), (0, 2)]),  # Transitive triangle (a -> b -> c, a -> c)
            "V-structure": nx.DiGraph([(0, 1), (0, 2), (1, 2)]),  # V-structure (a -> b, a -> c, b -> c)
            "W-structure": nx.DiGraph([(0, 1), (0, 2), (1, 2)]),  # W-structure (a -> b, a -> c, b -> c)
            "Tandem": nx.DiGraph([(0, 1), (1, 2)]),  # Tandem (a -> b, b -> c, forming chain)
        }

        # Initialize motif counts
        motif_counts = {key: 0 for key in motif_templates.keys()}
        motif_player_details = {key: [] for key in motif_templates.keys()}

        # Find all 3-node subgraphs
        for nodes in combinations(self.G.nodes(), 3):
            subgraph = self.G.subgraph(nodes)

            # Check each motif template for isomorphism
            for motif_name, motif_graph in motif_templates.items():
                matcher = DiGraphMatcher(subgraph, motif_graph)
                if matcher.is_isomorphic():
                    motif_counts[motif_name] += 1
                    # Add player names involved in this motif
                    node_mapping = {list(subgraph.nodes())[i]: i for i in range(len(subgraph.nodes()))}
                    motif_player_details[motif_name].append(node_mapping)
                    break  # Each subgraph can only match one motif

        # Save results to a text file
        self.save_3_motif_results(motif_counts, motif_player_details)

    def temporal_motifs(self, interval_df):
        """
        Detect temporal motifs in the passing network (triangles and squares).
        :param interval_df:
        :return: triangles, squares
        """
        triangles = []
        squares = []
        used_edges_timestamps = set()  # Track used (edge, timestamp) pairs
        triangle_edges = set()  # Set to track edges involved in triangles
        square_edges = set()  # Set to track edges involved in squares

        # Detect triangles
        for u, v, data_uv in self.G.edges(data=True):
            for w in self.G.successors(v):
                if w != u and self.G.has_edge(w, u):  # Check if a triangle is formed
                    data_vw = self.G[v][w]
                    data_wu = self.G[w][u]

                    timestamps_uv = [
                        t for t, acc in zip(data_uv['timestamps'], data_uv['accurate'])
                        if acc == 1
                    ]
                    timestamps_vw = [
                        t for t, acc in zip(data_vw['timestamps'], data_vw['accurate'])
                        if acc == 1
                    ]
                    timestamps_wu = [
                        t for t, acc in zip(data_wu['timestamps'], data_wu['accurate'])
                        if acc == 1
                    ]

                    # Check temporal order of edges
                    for t1 in timestamps_uv:
                        for t2 in timestamps_vw:
                            if t1 < t2:  # Ensure temporal order between first two edges
                                for t3 in timestamps_wu:
                                    if t2 < t3:  # Ensure temporal order for the complete triangle
                                        # Check if any edge with timestamp has already been used
                                        if (
                                                ((u, v, t1) in used_edges_timestamps) or
                                                ((v, w, t2) in used_edges_timestamps) or
                                                ((w, u, t3) in used_edges_timestamps)
                                        ):
                                            continue  # Skip if any edge and timestamp are already used

                                        # Check for validity (continuity)
                                        valid = True

                                        for x, y, data_xy in self.G.edges(data=True):
                                            if {x, y}.intersection({u, v, w}) == set():
                                                for timestamp in data_xy['timestamps']:
                                                    if (t1 < timestamp < t2) or (t2 < timestamp < t3):
                                                        valid = False
                                                        break
                                            if not valid:
                                                break

                                        if valid:
                                            triangles.append(((u, v, t1), (v, w, t2), (w, u, t3)))
                                            used_edges_timestamps.update({
                                                (u, v, t1),
                                                (v, w, t2),
                                                (w, u, t3)
                                            })
                                            # Add these edges to the triangle_edges set
                                            triangle_edges.update({
                                                (u, v),
                                                (v, w),
                                                (w, u)
                                            })

                                            break  # Exit inner loop as timestamps are now used

        # Detect squares
        for u, v, data_uv in self.G.edges(data=True):
            for w in self.G.successors(v):
                if w != u and self.G.has_edge(w, u):
                    for x in self.G.successors(w):
                        if x != v and x != u and self.G.has_edge(x, u):  # Check if a square is formed
                            data_wx = self.G[w][x]
                            data_xu = self.G[x][u]

                            timestamps_uv = [
                                t for t, acc in zip(data_uv['timestamps'], data_uv['accurate'])
                                if acc == 1
                            ]
                            timestamps_vw = [
                                t for t, acc in zip(data_vw['timestamps'], data_vw['accurate'])
                                if acc == 1
                            ]
                            timestamps_wx = [
                                t for t, acc in zip(data_wx['timestamps'], data_wx['accurate'])
                                if acc == 1
                            ]
                            timestamps_xu = [
                                t for t, acc in zip(data_xu['timestamps'], data_xu['accurate'])
                                if acc == 1
                            ]

                            for t1 in timestamps_uv:
                                for t2 in timestamps_vw:
                                    if t1 < t2:  # Ensure temporal order between the first two edges
                                        for t3 in timestamps_wx:
                                            if t2 < t3:  # Ensure temporal order for the third edge
                                                for t4 in timestamps_xu:
                                                    if t3 < t4:  # Ensure temporal order for the full square
                                                        # Check if any edge with timestamp has already been used
                                                        if (
                                                                ((u, v, t1) in used_edges_timestamps) or
                                                                ((v, w, t2) in used_edges_timestamps) or
                                                                ((w, x, t3) in used_edges_timestamps) or
                                                                ((x, u, t4) in used_edges_timestamps)
                                                        ):
                                                            continue  # Skip if any edge and timestamp are already used

                                                        # Check for validity (continuity)
                                                        valid = True

                                                        for y, z, data_yz in self.G.edges(data=True):
                                                            if {y, z}.intersection({u, v, w, x}) == set():
                                                                for timestamp in data_yz['timestamps']:
                                                                    if (t1 < timestamp < t2) or (
                                                                            t2 < timestamp < t3) or (
                                                                            t3 < timestamp < t4):
                                                                        valid = False
                                                                        break
                                                            if not valid:
                                                                break

                                                        if valid:
                                                            squares.append(
                                                                ((u, v, t1), (v, w, t2), (w, x, t3), (x, u, t4)))
                                                            used_edges_timestamps.update({
                                                                (u, v, t1),
                                                                (v, w, t2),
                                                                (w, x, t3),
                                                                (x, u, t4)
                                                            })
                                                            # Add these edges to the square_edges set
                                                            square_edges.update({
                                                                (u, v),
                                                                (v, w),
                                                                (w, x),
                                                                (x, u)
                                                            })

                                                            break  # Exit inner loop as timestamps are now used
        #

        self.save_temporal_motifs(triangles, squares)
        visualize_temporal_motifs(interval_df, triangles, squares, self.output_path)

        return triangle_edges, square_edges

    def save_3_motif_results(self, motif_counts, motif_player_details):
        """
        Save the 3-motif results to a txt file.

        Parameters:
            motif_counts (dict): Dictionary of motif counts {motif: count}.
            motif_player_details (dict): Dictionary of player details for each motif {motif: details}.
        """

        os.makedirs(self.output_path, exist_ok=True)
        file_path = os.path.join(self.output_path, "3-nodes_motifs.txt")

        with open(file_path, "w") as file:
            # Write motif counts
            file.write("3-node motif counts:\n")
            for motif, count in motif_counts.items():
                file.write(f"{motif}: {count}\n")
            file.write("\n")

            # Write player details
            file.write("Player names involved in each motif:\n\n")
            for motif, details in motif_player_details.items():
                file.write(f"{motif}:\n")
                if details:
                    for mapping in details:
                        file.write(f"{mapping}\n")
                else:
                    file.write("(None)\n")

                file.write("\n")

    def save_temporal_motifs(self, triangles, squares):
        """
        Save the temporal motifs to a CSV file.

        Parameters:
            triangles (dict): Dictionary of temporal triangles {triangle: count}.
            squares (dict): Dictionary of temporal squares {square: count}.
        """

        os.makedirs(self.output_path, exist_ok=True)
        file_path = os.path.join(self.output_path, "temporal_motifs.txt")

        def write_motif(file, motifs, motif_name):
            file.write(f"Temporal {motif_name}: {len(motifs)}\n\n")
            for i, motif in enumerate(motifs):
                file.write(f"{motif_name} {i + 1}:\n")
                for edge in motif:
                    file.write(f"{edge[0]} -> {edge[1]}, Timestamp: {edge[2]}\n")
                file.write("-" * 20 + "\n")
            file.write("\n\n")

        with open(file_path, "w") as file:

            write_motif(file, triangles, "Triangles")
            write_motif(file, squares, "Squares")

        # print(f"Temporal motifs saved to: {file_path}")

