import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

# Load the data from CSV files
print("Loading data...")
nodes_df = pd.read_csv('twitch_game_nodes.csv')
edges_df = pd.read_csv('twitch_game_edges.csv')

# Filter to top 1000 streamers by viewer count
print(f"Total streamers: {len(nodes_df)}")
TOP_N = 1000
filtered_nodes = nodes_df.nlargest(TOP_N, 'viewer_count')
print(f"Filtered to top {TOP_N} streamers")

# Get the set of streamers we're keeping for edge filtering
streamer_names = set(filtered_nodes['streamer_name'])

# Filter edges to only include connections between top streamers
filtered_edges = edges_df[
    (edges_df['streamer1'].isin(streamer_names)) & 
    (edges_df['streamer2'].isin(streamer_names))
]
print(f"Edges between top streamers: {len(filtered_edges)}")

# Build the NetworkX graph
print("Building graph...")
g = nx.Graph()  # Undirected graph

# Add nodes with viewer_count and game as attributes
for _, row in filtered_nodes.iterrows():
    g.add_node(row['streamer_name'], 
               viewer_count=row['viewer_count'],
               game=row['game'])

# Add edges between streamers (connected by shared games)
for _, row in filtered_edges.iterrows():
    g.add_edge(row['streamer1'], row['streamer2'], game=row['game'])

# Print basic graph statistics
print(f"Graph nodes: {g.number_of_nodes()}")
print(f"Graph edges: {g.number_of_edges()}")
print(f"Network density: {nx.density(g):.6f}")

# Extract viewer counts for node sizing
viewer_counts = [g.nodes[node]['viewer_count'] for node in g.nodes()]

# Normalize viewer counts to node size range for better visualization
min_size = 20
max_size = 800
max_viewers = max(viewer_counts)
norm_sizes = [(v / max_viewers) * (max_size - min_size) + min_size 
              for v in viewer_counts]

# Create visualization
print("Creating visualization...")
plt.figure(figsize=(20, 16))

# Use spring layout for positioning nodes (force-directed algorithm)
print("Computing layout (this may take a moment)...")
pos = nx.spring_layout(g, k=0.5, iterations=50, seed=42)

# Draw network edges (connections between streamers)
nx.draw_networkx_edges(g, pos, alpha=0.15, width=0.3, edge_color='gray')

# Draw network nodes with size based on viewer count and color gradient
nodes = nx.draw_networkx_nodes(g, pos, 
                               node_size=norm_sizes,
                               node_color=viewer_counts,
                               cmap=plt.cm.YlOrRd,  # Yellow to Red color map
                               alpha=0.8,
                               vmin=min(viewer_counts),
                               vmax=max(viewer_counts))

# Add labels for top 30 streamers only (to avoid clutter)
top_streamers = sorted(g.nodes(), 
                       key=lambda x: g.nodes[x]['viewer_count'], 
                       reverse=True)[:30]
labels = {node: node for node in top_streamers}
nx.draw_networkx_labels(g, pos, labels, font_size=7, font_weight='bold')

# Add title and color bar
plt.title(f'Twitch Streamer Network - Top {TOP_N} by Viewer Count\n(Node Size = Viewer Count)', 
          fontsize=18, fontweight='bold')
plt.colorbar(nodes, label='Viewer Count', shrink=0.8)
plt.axis('off')
plt.tight_layout()

# Save the figure to file
output_file = 'twitch_network_top1000_weighted.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved visualization to: {output_file}")
plt.show()

# Print network statistics
print("\n" + "="*60)
print("Network Statistics:")
print("="*60)
print(f"Top 10 streamers by viewers:")
for i, streamer in enumerate(top_streamers[:10], 1):
    viewer_count = g.nodes[streamer]['viewer_count']
    game = g.nodes[streamer]['game']
    degree = g.degree[streamer]  # Number of connections
    print(f"  {i:2d}. {streamer:20s} | {viewer_count:6,} viewers | {degree:4d} connections | {game}")

# Calculate and display additional network metrics
if g.number_of_nodes() > 0:
    print(f"\nNetwork Metrics:")
    # Average number of connections per node
    print(f"  Average degree: {sum(dict(g.degree()).values()) / g.number_of_nodes():.2f}")
    # Number of separate subgraphs (disconnected groups)
    print(f"  Number of connected components: {nx.number_connected_components(g)}")
    
    # Calculate path metrics if network is fully connected
    if nx.is_connected(g):
        print(f"  Average shortest path length: {nx.average_shortest_path_length(g):.2f}")
        print(f"  Diameter: {nx.diameter(g)}")
    else:
        print(f"  Network is not fully connected")
        largest_cc = max(nx.connected_components(g), key=len)
        print(f"  Largest connected component: {len(largest_cc)} nodes")

    # --- Calculate and display top 3 most important nodes by centrality measures ---
    print("\nTop 3 Most Important Nodes:")
    
    # Degree centrality: measures connections relative to total possible connections
    degree_centrality = nx.degree_centrality(g)
    top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:3]
    print("  By Degree Centrality:")
    for i, (node, score) in enumerate(top_degree, 1):
        print(f"    {i}. {node:20s} | Degree Centrality: {score:.4f} | Viewers: {g.nodes[node]['viewer_count']:,}")

    # Betweenness centrality: measures how often node appears on shortest paths
    betweenness_centrality = nx.betweenness_centrality(g)
    top_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:3]
    print("  By Betweenness Centrality:")
    for i, (node, score) in enumerate(top_betweenness, 1):
        print(f"    {i}. {node:20s} | Betweenness: {score:.4f} | Viewers: {g.nodes[node]['viewer_count']:,}")

    # Eigenvector centrality: measures influence based on connections to important nodes
    try:
        eigenvector_centrality = nx.eigenvector_centrality(g, max_iter=1000)
        top_eigenvector = sorted(eigenvector_centrality.items(), key=lambda x: x[1], reverse=True)[:3]
        print("  By Eigenvector Centrality:")
        for i, (node, score) in enumerate(top_eigenvector, 1):
            print(f"    {i}. {node:20s} | Eigenvector: {score:.4f} | Viewers: {g.nodes[node]['viewer_count']:,}")
    except Exception as e:
        print(f"  Eigenvector centrality could not be computed: {e}")