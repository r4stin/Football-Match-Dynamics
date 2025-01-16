# src/utils/graph_utils.py
import networkx as nx


def create_network(interval_df):
    G = nx.DiGraph()

    for _, row in interval_df.iterrows():
        player_from = row["lastName"]
        player_to = row["lastName_next"]
        timestamp = row["eventSec"]
        accurate = row["accurate"]

        if G.has_edge(player_from, player_to):
            G[player_from][player_to]["timestamps"].append(timestamp)
            G[player_from][player_to]["accurate"].append(accurate)
            G[player_from][player_to]["weight"] += 1
        else:
            G.add_edge(player_from, player_to, weight=1, timestamps=[timestamp], accurate=[accurate])
    return G
