import os
import sys
import pandas as pd
from src.utils.data_loader import load_match_data
from src.utils.graph_utils import create_network
from src.utils.players_position import get_players_position
from src.utils.visualization import draw_pitch_graph
from src.analysis.adjacency_matrix import compute_adjacency_matrix
from src.analysis.centrality_analysis import compute_centrality_matrix
from src.analysis.motif_detection import MotifAnalyzer
from src.analysis.clustering import GraphClustering


class FootballDataAnalysis:
    def __init__(self, match_name, data_path):
        self.match_name = match_name
        self.data_path = data_path
        self.data = None
        self.halves = None
        self.interval_folder = None
        self.teams = None

    def load_data(self):
        """
        Load and preprocess match data.
        """
        # Load the dataset
        # all_data = pd.read_csv(self.data_path)
        all_data = load_match_data(self.data_path)

        # Convert match names to lowercase for consistency
        self.match_name = self.match_name.lower()
        all_data["matchName"] = all_data["matchName"].str.lower()

        # Filter data for the specified match
        self.data = all_data[all_data["matchName"] == self.match_name]
        if self.data.empty:
            raise ValueError(f"No data found for match: {self.match_name}")

        # Load each team's data
        home_team, away_team = self.match_name.split("-")
        # print(self.data[(self.data["teamName"] == home_team) & (self.data["matchPeriod"] == "1H")])
        self.halves = {
            home_team: {
                "1H": self.data[(self.data["matchPeriod"] == "1H") & (self.data["teamName"] == home_team)],
                "2H": self.data[(self.data["matchPeriod"] == "2H") & (self.data["teamName"] == home_team)]
            },
            away_team: {
                "1H": self.data[(self.data["matchPeriod"] == "1H") & (self.data["teamName"] == away_team)],
                "2H": self.data[(self.data["matchPeriod"] == "2H") & (self.data["teamName"] == away_team)]
            }
            }

        print(f"Data loaded for match: {self.match_name}")

    def process_interval(self, interval_df, half, start, end, output_path):
        """
        Process a specific interval of the match.
        """
        print(f"Processing interval {start}-{end}...")
        subfolders = ["network", "clustering", "motifs", "metrics"]
        # Ensure subfolders exist
        for sub in subfolders:
            path = os.path.join(output_path, sub)
            os.makedirs(path, exist_ok=True)

        # Get player positions
        player_positions = get_players_position(interval_df)

        # Create network
        G = create_network(interval_df)
        print("Network created.")

        # # Visualize Network
        draw_pitch_graph(
            player_positions=player_positions,
            edges=G.edges(data=True),
            title="Pass Network",
            output_path=output_path + f"/network/pass_network_{start}-{end}.png"
        )

        # Compute adjacency matrix
        adj_title = f"{self.match_name} - {half} - {start}-{end}"
        compute_adjacency_matrix(G, output_path, adj_title)
        print("Adjacency matrix computed.")

        # Centrality analysis
        centrality_metrics = compute_centrality_matrix(G, player_positions, output_path)
        print("Centrality metrics computed.")


        # Motif detection
        # motif_analyzer = MotifAnalyzer(self.G, output_path)
        motif_analyzer = MotifAnalyzer(G, output_path)
        motif_analyzer.count_3_motifs()
        print("3-node motifs detected.")
        triangle_edges, square_edges = motif_analyzer.temporal_motifs(interval_df)
        print("Temporal motifs detected.")

        # Clustering and Tactical insights
        clustering = GraphClustering(G, interval_df, output_path)
        clustering.update_edge_weights(triangle_edges, square_edges)
        clustering.clustering_and_tactical_insights(player_positions, centrality_metrics)
        print("Clustering and Tactical insights generated.")

        print(f"Interval {start}-{end} processed.")
        print("=" * 50)

    def handler(self):
        """
        Main handler for processing all intervals of the match.
        """
        output_folder = f"outputs/{self.match_name}"
        os.makedirs(output_folder, exist_ok=True)

        for team, df_team in self.halves.items():
            print(f"Processing {team}...")
            output_folder_team = os.path.join(output_folder, team)
            for half, df_half in df_team.items():
                print(f"Processing {half}...")
                max_time = int(df_half["eventSec"].max())

                intervals = [(0, 900), (900, 1800), (1800, max_time)]
                half_folder = os.path.join(output_folder_team, half)
                os.makedirs(half_folder, exist_ok=True)

                for start, end in intervals:
                    interval_df = df_half[(df_half["eventSec"] >= start) & (df_half["eventSec"] < end)]
                    if interval_df.empty:
                        print(f"No data for interval {start}-{end} in {half}.")
                        continue

                    # print(interval_df)

                    interval_output_path = os.path.join(half_folder, f"interval_{start}_{end}")
                    self.process_interval(interval_df, half, start, end, interval_output_path)

        print("All intervals processed.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m src.main <match_name>")
        sys.exit(1)

    match_name = sys.argv[1]  # Match name as passed in the terminal
    data_file = "data/matches_data.csv"  # Path to your dataset file

    try:
        analysis = FootballDataAnalysis(match_name, data_file)
        analysis.load_data()
        analysis.handler()
    except Exception as e:
        print(f"Error: {e}")
