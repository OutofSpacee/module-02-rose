# Build Twitch network using shared games/categories (no special auth needed)
# Creates connections between streamers who play the same games
# Runs for 5 minutes to collect as much data as possible

from twitchAPI.twitch import Twitch
import asyncio
import networkx as nx
import time
import csv

app_id = '____' # removed for privacy
app_secret = '_____' #  removed for privacy

# Run for 5 minutes
RUN_DURATION = 5 * 60  # 5 minutes in seconds
# Logging controls
VERBOSE = False               # Set True to see detailed logs
PROGRESS_INTERVAL_SEC = 30    # Print a brief progress update every N seconds

async def build_game_network():
    # Initialize with just app authentication
    twitch = await Twitch(app_id, app_secret)
    
    g = nx.Graph()  # Undirected graph
    
    print("Building Twitch network from shared games...")
    print(f"Running for {RUN_DURATION // 60} minutes to collect as much data as possible...")
    print("=" * 60)
    
    start_time = time.time()
    
    # Get currently live streams (continuously for 5 minutes)
    if VERBOSE:
        print("\n1. Collecting currently live streamers...")
    
    streamers = []
    games_dict = {}  # game_name -> list of streamer names
    seen_streamers = set()  # Track unique streamers
    
    stream_count = 0
    cycle_count = 0
    last_progress_ts = start_time
    
    while time.time() - start_time < RUN_DURATION:
        cycle_count += 1
        cycle_start = time.time()
        if VERBOSE:
            print(f"\n--- Collection Cycle {cycle_count} (Time elapsed: {int(time.time() - start_time)}s) ---")
        
        try:
            batch_count = 0
            async for stream in twitch.get_streams(first=100):  # Max is 100
                streamer_name = stream.user_name
                streamer_id = stream.user_id
                game_name = stream.game_name
                viewer_count = stream.viewer_count
                
                # Only add if we haven't seen this streamer yet
                if streamer_name not in seen_streamers:
                    seen_streamers.add(streamer_name)
                    
                    g.add_node(streamer_name, 
                               user_id=streamer_id, 
                               game=game_name,
                               viewer_count=viewer_count)
                    
                    streamers.append({'name': streamer_name, 'id': streamer_id, 'game': game_name})
                    
                    # Track which streamers play which games
                    if game_name not in games_dict:
                        games_dict[game_name] = []
                    games_dict[game_name].append(streamer_name)
                    
                    stream_count += 1
                    batch_count += 1
                
                # Check time limit during iteration
                if time.time() - start_time >= RUN_DURATION:
                    if VERBOSE:
                        print(f"\n  Time limit reached!")
                    break
            # Throttled progress update
            now = time.time()
            if now - last_progress_ts >= PROGRESS_INTERVAL_SEC:
                print(f"Progress: {stream_count} unique streamers across {len(games_dict)} games (t={int(now - start_time)}s)")
                last_progress_ts = now
            
            # Small delay between cycles to avoid rate limiting
            await asyncio.sleep(2)
            
        except Exception as e:
            print(f"  Error during collection: {e}")
            await asyncio.sleep(5)  # Wait a bit on error
    
    elapsed_time = time.time() - start_time
    print(f"\nCollection complete! Ran for {int(elapsed_time)}s ({elapsed_time/60:.1f} minutes)")
    print(f"Found {len(streamers)} unique live streamers across {len(games_dict)} games")
    
    # Create edges between streamers who play the same game
    if VERBOSE:
        print("\n2. Creating edges between streamers playing the same games...")
    edge_count = 0
    
    for game_name, streamers_in_game in games_dict.items():
        if len(streamers_in_game) > 1:  # Only if multiple streamers play this game
            # Connect all streamers playing the same game
            for i, streamer1 in enumerate(streamers_in_game):
                for streamer2 in streamers_in_game[i+1:]:
                    g.add_edge(streamer1, streamer2, game=game_name)
                    edge_count += 1
            # Intentionally avoid per-connection or per-game connection printing
    
    print("\n" + "=" * 60)
    print(f"Network Statistics:")
    print(f"  Total nodes (streamers): {g.number_of_nodes()}")
    print(f"  Total edges (connections): {g.number_of_edges()} (edges created this run: {edge_count})")
    print(f"  Network density: {nx.density(g):.4f}")

    # Save nodes as CSV
    with open("twitch_game_nodes.csv", "w", newline='', encoding='utf-8') as nodefile:
        writer = csv.writer(nodefile)
        writer.writerow(["streamer_name", "user_id", "game", "viewer_count"])
        for n, attr in g.nodes(data=True):
            writer.writerow([n, attr.get("user_id", ""), attr.get("game", ""), attr.get("viewer_count", "")])

    print("Node list saved to: twitch_game_nodes.csv")

    # Save edges as CSV
    with open("twitch_game_edges.csv", "w", newline='', encoding='utf-8') as edgefile:
        writer = csv.writer(edgefile)
        writer.writerow(["streamer1", "streamer2", "game"])
        for u, v, attr in g.edges(data=True):
            writer.writerow([u, v, attr.get("game", "")])

    print("Edge list saved to: twitch_game_edges.csv")

    # Analysis intentionally omitted; perform in a separate script if needed

    await twitch.close()
    return g

if __name__ == "__main__":
    network = asyncio.run(build_game_network())
