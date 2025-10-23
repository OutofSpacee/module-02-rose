# imports

from __future__ import annotations

import argparse
import os
from collections import Counter
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


def load_data(nodes_path: str, edges_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load node and edge data from CSV files."""
    nodes_df = pd.read_csv(nodes_path)
    # Edges file may be large; read only essential columns to save memory
    edges_df = pd.read_csv(edges_path, usecols=["streamer1", "streamer2", "game"]) 
    return nodes_df, edges_df


def filter_top_streamers(nodes_df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Filter to top N streamers by viewer count."""
    # Ensure viewer_count is numeric and handle any missing/invalid values
    nodes_df = nodes_df.copy()
    nodes_df["viewer_count"] = pd.to_numeric(nodes_df["viewer_count"], errors="coerce").fillna(0)
    # Select top N streamers by viewer count
    filtered = nodes_df.nlargest(top_n, "viewer_count").reset_index(drop=True)
    return filtered


def build_graph(filtered_nodes: pd.DataFrame, edges_df: pd.DataFrame) -> nx.Graph:
    """Build induced subgraph from filtered nodes and edges."""
    # Get set of streamer names to keep
    keep = set(filtered_nodes["streamer_name"].astype(str))
    # Filter edges to only include connections between top streamers
    fe = edges_df[(edges_df["streamer1"].isin(keep)) & (edges_df["streamer2"].isin(keep))]

    # Build undirected graph
    g = nx.Graph()
    # Add nodes with viewer count and game as attributes
    for _, row in filtered_nodes.iterrows():
        g.add_node(
            str(row["streamer_name"]),
            viewer_count=float(row.get("viewer_count", 0) or 0.0),
            game=str(row.get("game", "Unknown")),
        )

    # Add edges between streamers (connected via shared games)
    for _, row in fe.iterrows():
        g.add_edge(str(row["streamer1"]), str(row["streamer2"]), game=str(row.get("game", "")))

    return g


def detect_communities(g: nx.Graph) -> Dict[str, int]:
    """Return mapping node -> community_id using greedy modularity communities.
    Isolated nodes each become their own community.
    """
    if g.number_of_nodes() == 0:
        return {}

    # If no edges exist, assign each node its own cluster
    if g.number_of_edges() == 0:
        return {n: i for i, n in enumerate(g.nodes())}

    # Get all connected components in the graph
    components = list(nx.connected_components(g))
    # Run community detection on each component separately and merge results
    from networkx.algorithms.community import greedy_modularity_communities

    node_to_comm: Dict[str, int] = {}
    next_comm_id = 0
    for comp_nodes in components:
        # Create subgraph for this connected component
        sub = g.subgraph(comp_nodes)
        # Detect communities using greedy modularity optimization
        comms = list(greedy_modularity_communities(sub))
        # Assign community IDs to nodes
        for ci, comm_nodes in enumerate(comms):
            for n in comm_nodes:
                node_to_comm[n] = next_comm_id + ci
        next_comm_id += len(comms)

    return node_to_comm


def label_communities_by_game(g: nx.Graph, node_to_comm: Dict[str, int]) -> Dict[int, str]:
    """For each community, label by the most common game among its nodes."""
    # Collect all games for each cluster
    comm_games: Dict[int, List[str]] = {}
    for n, cid in node_to_comm.items():
        comm_games.setdefault(cid, []).append(g.nodes[n].get("game", "Unknown"))
    
    # Find most common game per cluster to use as label
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
    # Group nodes by their cluster assignment
    cluster_members: Dict[int, List[str]] = {}
    for n, cid in node_to_comm.items():
        cluster_members.setdefault(cid, []).append(n)
    
    # Find the max viewer count within each cluster
    cluster_max_viewers: Dict[int, float] = {}
    for cid, members in cluster_members.items():
        max_v = max((g.nodes[n].get("viewer_count", 0.0) for n in members), default=1.0)
        cluster_max_viewers[cid] = max_v if max_v > 0 else 1.0
    
    # Compute normalized importance score (relative to cluster max)
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
    
    # Group nodes by cluster
    cluster_members: Dict[int, List[str]] = defaultdict(list)
    for n, cid in node_to_comm.items():
        cluster_members[cid].append(n)
    
    # Sort clusters by size (number of members) descending
    sorted_clusters = sorted(cluster_members.items(), key=lambda x: len(x[1]), reverse=True)
    
    print("\n" + "="*80)
    print("CLUSTER ANALYSIS (Importance = Viewer Count within Topic)")
    print("="*80)
    
    # Print top 15 largest clusters
    for rank, (cid, members) in enumerate(sorted_clusters[:15], 1):
        label = comm_label.get(cid, f"Cluster {cid}")
        print(f"\n#{rank} Topic: {label} ({len(members)} streamers)")
        
        # Rank streamers by importance within this cluster
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
            degree = g.degree[n]  # Number of connections
            print(f"    {i}. {n:25s} | Views: {int(vc):>8,} | Importance: {imp:.3f} | Connections: {degree:>4} | {game}")
    
    print("\n" + "="*80)


def compute_layout(g: nx.Graph, layout: str) -> Dict[str, Tuple[float, float]]:
    """Compute node positions using specified layout algorithm."""
    if layout == "spring":
        # Spring layout: force-directed algorithm
        return nx.spring_layout(g, k=0.6, iterations=200, seed=42)
    if layout == "kamada_kawai":
        # Kamada-Kawai: minimize energy of spring system
        return nx.kamada_kawai_layout(g)
    if layout == "fr":
        # Fruchterman-Reingold: another force-directed layout
        return nx.fruchterman_reingold_layout(g, seed=42)
    # Default to spring layout
    return nx.spring_layout(g, k=0.6, iterations=200, seed=42)


def visualize(
    g: nx.Graph,
    node_to_comm: Dict[str, int],
    comm_label: Dict[int, str],
    top_n: int,
    layout: str = "spring",
    out_dir: str = ".",
):
    """Create and save network visualization with clusters colored."""
    # Compute node positions using layout algorithm
    pos = compute_layout(g, layout)

    # Map cluster IDs to consecutive color indices for color mapping
    unique_comms = sorted(set(node_to_comm.values()))
    cid_to_color_idx = {cid: i for i, cid in enumerate(unique_comms)}

    # Prepare node visual attributes
    viewer_counts = [g.nodes[n].get("viewer_count", 0.0) for n in g.nodes()]
    max_v = max(viewer_counts) if viewer_counts else 1.0
    # Normalize viewer counts to node size range
    min_size, max_size = 20, 800
    sizes = [min_size + (vc / max_v) * (max_size - min_size) for vc in viewer_counts]
    # Map each node to its cluster's color index
    colors = [cid_to_color_idx[node_to_comm[n]] for n in g.nodes()]

    # Create figure
    plt.figure(figsize=(22, 18))
    
    # Draw edges (connections between streamers)
    nx.draw_networkx_edges(g, pos, alpha=0.08, width=0.3, edge_color="#888888")
    
    # Draw nodes with size based on viewers and color based on cluster
    nodes = nx.draw_networkx_nodes(
        g,
        pos,
        node_size=sizes,
        node_color=colors,
        cmap=plt.cm.tab20,  # 20-color discrete colormap
        alpha=0.9,
    )

    # Add labels for top 25 streamers only (to avoid clutter)
    top_labeled = sorted(g.nodes(), key=lambda n: g.nodes[n].get("viewer_count", 0.0), reverse=True)[:25]
    nx.draw_networkx_labels(g, pos, {n: n for n in top_labeled}, font_size=8, font_weight="bold")

    # Add cluster labels near cluster centroids (top 10 largest clusters)
    comm_sizes = Counter(node_to_comm.values())
    largest_cids = [cid for cid, _ in comm_sizes.most_common(10)]
    for cid in largest_cids:
        members = [n for n, c in node_to_comm.items() if c == cid]
        if not members:
            continue
        # Calculate centroid (average position) of cluster
        xs, ys = zip(*[pos[n] for n in members])
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        label = comm_label.get(cid, f"Cluster {cid}")
        # Add text annotation at centroid
        plt.text(cx, cy, label, fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.6))

    # Add title and formatting
    plt.title(
        f"Twitch Topic Clusters — Top {top_n} Streamers\nColors = Communities, Size = Viewer Count",
        fontsize=18,
        fontweight="bold",
    )
    sm = plt.cm.ScalarMappable(cmap=plt.cm.tab20)
    sm.set_array([])
    plt.axis('off')
    plt.tight_layout()

    # Save to file
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
    """Export cluster assignments and node metadata to CSV file."""
    rows = []
    # Collect data for each node
    for n in g.nodes():
        rows.append(
            {
                "streamer_name": n,
                "viewer_count": g.nodes[n].get("viewer_count", 0.0),
                "game": g.nodes[n].get("game", "Unknown"),
                "cluster_id": node_to_comm.get(n, -1),
                "cluster_label": comm_label.get(node_to_comm.get(n, -1), ""),
                "importance": importance.get(n, 0.0),  # Importance within cluster
                "degree": g.degree[n],  # Number of connections
            }
        )
    # Sort by cluster ID, then by importance (descending)
    df = pd.DataFrame(rows).sort_values(["cluster_id", "importance"], ascending=[True, False])
    
    # Save to CSV
    out_csv = os.path.join(out_dir, f"twitch_topics_top{top_n}_clusters.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved cluster assignments to: {out_csv}")


def main():
    """Main execution function."""
    # Parse command-line arguments
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

    # Load data from CSV files
    print("Loading data…")
    nodes_df, edges_df = load_data(args.nodes, args.edges)
    print(f"Nodes: {len(nodes_df):,} | Edges: {len(edges_df):,}")

    # Filter to top N streamers
    print(f"Filtering to top {args.top} streamers by viewer_count…")
    top_nodes = filter_top_streamers(nodes_df, args.top)
    print(f"Kept {len(top_nodes):,} streamers")

    # Build network graph with filtered nodes and edges
    print("Building induced subgraph…")
    g = build_graph(top_nodes, edges_df)
    print(f"Graph nodes: {g.number_of_nodes():,} | edges: {g.number_of_edges():,} | density: {nx.density(g):.6f}")

    # Detect topic clusters using community detection
    print("Detecting communities…")
    node_to_comm = detect_communities(g)
    num_clusters = len(set(node_to_comm.values())) if node_to_comm else 0
    print(f"Found {num_clusters} clusters")

    # Label each cluster by its most common game
    print("Labeling clusters by most common game…")
    comm_label = label_communities_by_game(g, node_to_comm)

    # Compute importance scores (relative viewer count within each cluster)
    print("Computing importance scores (viewer count within topic)…")
    importance = compute_importance_scores(g, node_to_comm)

    # Export results to CSV
    print("Exporting results…")
    export_cluster_csv(g, node_to_comm, comm_label, importance, args.top)

    # Create and save visualization
    print("Rendering visualization… (this may take a moment)")
    visualize(g, node_to_comm, comm_label, args.top, layout=args.layout)

    # Print detailed cluster analysis to console
    print_cluster_analysis(g, node_to_comm, comm_label, importance, top_k=5)

    # Print overall network statistics
    if g.number_of_nodes() > 0:
        avg_deg = sum(dict(g.degree()).values()) / g.number_of_nodes()
        print(f"\nOverall Network Stats:")
        print(f"  Average degree: {avg_deg:.2f}")
        print(f"  Connected components: {nx.number_connected_components(g)}")


if __name__ == "__main__":
    main()
