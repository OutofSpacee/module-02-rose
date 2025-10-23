"""
Cluster and visualize topics within the top N Twitch streamers.

Approach
- Load nodes and edges from twitch_game_nodes.csv / twitch_game_edges.csv
- Filter to top N streamers by viewer_count
- Build induced subgraph among those streamers
- Detect communities (clusters) using modularity-based community detection
- Label clusters by their most common game (as a proxy for "topic")
- Compute importance: node importance = viewer_count / max_viewer_count_in_cluster
  (A node is more important if it has more views than other streamers in the same topic)
- Visualize: nodes sized by viewers, colored by cluster; save PNG and CSV outputs
- Print analysis showing most important streamers per topic cluster

Usage
  python cluster_topics.py --top 1000 --layout spring

Outputs
- twitch_topics_top<top>_clusters.png: static plot
- twitch_topics_top<top>_clusters.csv: streamer-to-cluster mapping with importance scores
- Console: detailed analysis of top clusters and most important streamers per topic

Dependencies
- pandas, networkx, matplotlib

Notes
- If you don't have edges collected yet, run build_game_network.py first.
- Importance score ranges from 0 to 1, where 1 = most important (most views) in that topic cluster.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


def load_data(nodes_path: str, edges_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    nodes_df = pd.read_csv(nodes_path)
    # edges file may be large; read minimally needed columns
    edges_df = pd.read_csv(edges_path, usecols=["streamer1", "streamer2", "game"]) 
    return nodes_df, edges_df


def filter_top_streamers(nodes_df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    # Ensure numeric
    nodes_df = nodes_df.copy()
    nodes_df["viewer_count"] = pd.to_numeric(nodes_df["viewer_count"], errors="coerce").fillna(0)
    filtered = nodes_df.nlargest(top_n, "viewer_count").reset_index(drop=True)
    return filtered


def build_graph(filtered_nodes: pd.DataFrame, edges_df: pd.DataFrame) -> nx.Graph:
    keep = set(filtered_nodes["streamer_name"].astype(str))
    # Filter edges to only within the top set
    fe = edges_df[(edges_df["streamer1"].isin(keep)) & (edges_df["streamer2"].isin(keep))]

    # Build graph
    g = nx.Graph()
    for _, row in filtered_nodes.iterrows():
        g.add_node(
            str(row["streamer_name"]),
            viewer_count=float(row.get("viewer_count", 0) or 0.0),
            game=str(row.get("game", "Unknown")),
        )

    # Add edges (undirected)
    for _, row in fe.iterrows():
        g.add_edge(str(row["streamer1"]), str(row["streamer2"]), game=str(row.get("game", "")))

    return g


def detect_communities(g: nx.Graph) -> Dict[str, int]:
    """Return mapping node -> community_id using greedy modularity communities.
    Isolated nodes each become their own community.
    """
    if g.number_of_nodes() == 0:
        return {}

    # If no edges, assign each node its own cluster
    if g.number_of_edges() == 0:
        return {n: i for i, n in enumerate(g.nodes())}

    # Work on largest connected component to avoid exploding number of tiny clusters
    components = list(nx.connected_components(g))
    # Run community detection per connected component and merge labels
    from networkx.algorithms.community import greedy_modularity_communities

    node_to_comm: Dict[str, int] = {}
    next_comm_id = 0
    for comp_nodes in components:
        sub = g.subgraph(comp_nodes)
        comms = list(greedy_modularity_communities(sub))
        for ci, comm_nodes in enumerate(comms):
            for n in comm_nodes:
                node_to_comm[n] = next_comm_id + ci
        next_comm_id += len(comms)

    return node_to_comm


def label_communities_by_game(g: nx.Graph, node_to_comm: Dict[str, int]) -> Dict[int, str]:
    """For each community, label by the most common game among its nodes."""
    comm_games: Dict[int, List[str]] = {}
    for n, cid in node_to_comm.items():
        comm_games.setdefault(cid, []).append(g.nodes[n].get("game", "Unknown"))
    comm_label: Dict[int, str] = {}
    for cid, games in comm_games.items():
        most_common_game, cnt = Counter(games).most_common(1)[0]
        comm_label[cid] = f"{most_common_game}"
    return comm_label


def compute_importance_scores(g: nx.Graph, node_to_comm: Dict[str, int]) -> Dict[str, float]:
    """
    Compute importance score for each node within its topic cluster.
    Importance = viewer_count / max_viewer_count_in_cluster
    Range: [0, 1], where 1 = most important streamer in that topic.
    """
    # Group nodes by cluster
    cluster_members: Dict[int, List[str]] = {}
    for n, cid in node_to_comm.items():
        cluster_members.setdefault(cid, []).append(n)
    
    # Compute max viewers per cluster
    cluster_max_viewers: Dict[int, float] = {}
    for cid, members in cluster_members.items():
        max_v = max((g.nodes[n].get("viewer_count", 0.0) for n in members), default=1.0)
        cluster_max_viewers[cid] = max_v if max_v > 0 else 1.0
    
    # Compute normalized importance
    importance: Dict[str, float] = {}
    for n, cid in node_to_comm.items():
        vc = g.nodes[n].get("viewer_count", 0.0)
        importance[n] = vc / cluster_max_viewers[cid]
    
    return importance


def print_cluster_analysis(
    g: nx.Graph,
    node_to_comm: Dict[str, int],
    comm_label: Dict[int, str],
    importance: Dict[str, float],
    top_k: int = 5,
):
    """Print analysis of clusters with most important nodes per cluster."""
    from collections import defaultdict
    
    cluster_members: Dict[int, List[str]] = defaultdict(list)
    for n, cid in node_to_comm.items():
        cluster_members[cid].append(n)
    
    # Sort clusters by size
    sorted_clusters = sorted(cluster_members.items(), key=lambda x: len(x[1]), reverse=True)
    
    print("\n" + "="*80)
    print("CLUSTER ANALYSIS (Importance = Viewer Count within Topic)")
    print("="*80)
    
    for rank, (cid, members) in enumerate(sorted_clusters[:15], 1):  # Top 15 clusters
        label = comm_label.get(cid, f"Cluster {cid}")
        print(f"\n#{rank} Topic: {label} ({len(members)} streamers)")
        
        # Get top K most important streamers in this cluster
        ranked = sorted(
            members,
            key=lambda n: importance.get(n, 0.0),
            reverse=True
        )[:top_k]
        
        print(f"  Most Important Streamers:")
        for i, n in enumerate(ranked, 1):
            vc = g.nodes[n].get("viewer_count", 0.0)
            imp = importance.get(n, 0.0)
            game = g.nodes[n].get("game", "Unknown")
            degree = g.degree[n]
            print(f"    {i}. {n:25s} | Views: {int(vc):>8,} | Importance: {imp:.3f} | Connections: {degree:>4} | {game}")
    
    print("\n" + "="*80)


def compute_layout(g: nx.Graph, layout: str) -> Dict[str, Tuple[float, float]]:
    if layout == "spring":
        return nx.spring_layout(g, k=0.6, iterations=200, seed=42)
    if layout == "kamada_kawai":
        return nx.kamada_kawai_layout(g)
    if layout == "fr":
        return nx.fruchterman_reingold_layout(g, seed=42)
    # Default
    return nx.spring_layout(g, k=0.6, iterations=200, seed=42)


def visualize(
    g: nx.Graph,
    node_to_comm: Dict[str, int],
    comm_label: Dict[int, str],
    top_n: int,
    layout: str = "spring",
    out_dir: str = ".",
):
    pos = compute_layout(g, layout)

    # Map communities to consecutive color indices
    unique_comms = sorted(set(node_to_comm.values()))
    cid_to_color_idx = {cid: i for i, cid in enumerate(unique_comms)}

    # Node attributes
    viewer_counts = [g.nodes[n].get("viewer_count", 0.0) for n in g.nodes()]
    max_v = max(viewer_counts) if viewer_counts else 1.0
    min_size, max_size = 20, 800
    sizes = [min_size + (vc / max_v) * (max_size - min_size) for vc in viewer_counts]
    colors = [cid_to_color_idx[node_to_comm[n]] for n in g.nodes()]

    plt.figure(figsize=(22, 18))
    nx.draw_networkx_edges(g, pos, alpha=0.08, width=0.3, edge_color="#888888")
    nodes = nx.draw_networkx_nodes(
        g,
        pos,
        node_size=sizes,
        node_color=colors,
        cmap=plt.cm.tab20,
        alpha=0.9,
    )

    # Add labels for the biggest streamers to avoid clutter
    top_labeled = sorted(g.nodes(), key=lambda n: g.nodes[n].get("viewer_count", 0.0), reverse=True)[:25]
    nx.draw_networkx_labels(g, pos, {n: n for n in top_labeled}, font_size=8, font_weight="bold")

    # Add simple cluster annotations near cluster centroids (top 10 clusters by size)
    comm_sizes = Counter(node_to_comm.values())
    largest_cids = [cid for cid, _ in comm_sizes.most_common(10)]
    for cid in largest_cids:
        members = [n for n, c in node_to_comm.items() if c == cid]
        if not members:
            continue
        # centroid
        xs, ys = zip(*[pos[n] for n in members])
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        label = comm_label.get(cid, f"Cluster {cid}")
        plt.text(cx, cy, label, fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.6))

    plt.title(
        f"Twitch Topic Clusters — Top {top_n} Streamers\nColors = Communities, Size = Viewer Count",
        fontsize=18,
        fontweight="bold",
    )
    sm = plt.cm.ScalarMappable(cmap=plt.cm.tab20)
    sm.set_array([])
    plt.axis("off")
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"twitch_topics_top{top_n}_clusters.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved visualization to: {out_path}")
    plt.close()


def export_cluster_csv(
    g: nx.Graph,
    node_to_comm: Dict[str, int],
    comm_label: Dict[int, str],
    importance: Dict[str, float],
    top_n: int,
    out_dir: str = ".",
):
    rows = []
    for n in g.nodes():
        rows.append(
            {
                "streamer_name": n,
                "viewer_count": g.nodes[n].get("viewer_count", 0.0),
                "game": g.nodes[n].get("game", "Unknown"),
                "cluster_id": node_to_comm.get(n, -1),
                "cluster_label": comm_label.get(node_to_comm.get(n, -1), ""),
                "importance": importance.get(n, 0.0),
                "degree": g.degree[n],
            }
        )
    df = pd.DataFrame(rows).sort_values(["cluster_id", "importance"], ascending=[True, False])
    out_csv = os.path.join(out_dir, f"twitch_topics_top{top_n}_clusters.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved cluster assignments to: {out_csv}")


def main():
    parser = argparse.ArgumentParser(description="Cluster and visualize topics for top Twitch streamers")
    parser.add_argument("--nodes", default="twitch_game_nodes.csv", help="Path to nodes CSV")
    parser.add_argument("--edges", default="twitch_game_edges.csv", help="Path to edges CSV")
    parser.add_argument("--top", type=int, default=1000, help="Top N streamers by viewer_count")
    parser.add_argument(
        "--layout",
        choices=["spring", "kamada_kawai", "fr"],
        default="spring",
        help="Layout algorithm for visualization",
    )
    args = parser.parse_args()

    print("Loading data…")
    nodes_df, edges_df = load_data(args.nodes, args.edges)
    print(f"Nodes: {len(nodes_df):,} | Edges: {len(edges_df):,}")

    print(f"Filtering to top {args.top} streamers by viewer_count…")
    top_nodes = filter_top_streamers(nodes_df, args.top)
    print(f"Kept {len(top_nodes):,} streamers")

    print("Building induced subgraph…")
    g = build_graph(top_nodes, edges_df)
    print(f"Graph nodes: {g.number_of_nodes():,} | edges: {g.number_of_edges():,} | density: {nx.density(g):.6f}")

    print("Detecting communities…")
    node_to_comm = detect_communities(g)
    num_clusters = len(set(node_to_comm.values())) if node_to_comm else 0
    print(f"Found {num_clusters} clusters")

    print("Labeling clusters by most common game…")
    comm_label = label_communities_by_game(g, node_to_comm)

    print("Computing importance scores (viewer count within topic)…")
    importance = compute_importance_scores(g, node_to_comm)

    print("Exporting results…")
    export_cluster_csv(g, node_to_comm, comm_label, importance, args.top)

    print("Rendering visualization… (this may take a moment)")
    visualize(g, node_to_comm, comm_label, args.top, layout=args.layout)

    # Print detailed cluster analysis
    print_cluster_analysis(g, node_to_comm, comm_label, importance, top_k=5)

    # Quick stats
    if g.number_of_nodes() > 0:
        avg_deg = sum(dict(g.degree()).values()) / g.number_of_nodes()
        print(f"\nOverall Network Stats:")
        print(f"  Average degree: {avg_deg:.2f}")
        print(f"  Connected components: {nx.number_connected_components(g)}")


if __name__ == "__main__":
    main()
