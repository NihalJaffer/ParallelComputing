"""
Script to compare different maze exploration algorithms.
"""

import time
import argparse
import json
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

from src.maze import create_maze
from src.explorer import Explorer
from src.astar_explorer import AStarExplorer
from src.bidirectional_explorer import BidirectionalExplorer

def run_comparison(maze_type="static", width=30, height=30, visualize=False):
    """
    Run a comparison of different explorer algorithms.
    
    Args:
        maze_type: Type of maze ("random" or "static")
        width: Width of maze for random mazes
        height: Height of maze for random mazes
        visualize: Whether to visualize the exploration
        
    Returns:
        dict: Results of the comparison
    """
    # Create maze
    maze = create_maze(width, height, maze_type)
    
    # Create explorers
    explorers = {
        "Right-Hand Rule": Explorer(maze, visualize=visualize),
        "A* Search": AStarExplorer(maze, visualize=visualize),
        "Bidirectional": BidirectionalExplorer(maze, visualize=visualize)
    }
    
    # Run each explorer and collect results
    results = {}
    
    for name, explorer in explorers.items():
        print(f"\nRunning {name} explorer...")
        elapsed_time, moves = explorer.solve()
        
        results[name] = {
            'algorithm': name,
            'moves': len(moves),
            'backtracks': explorer.backtrack_count,
            'elapsed_time': elapsed_time,
            'moves_per_second': len(moves) / elapsed_time if elapsed_time > 0 else 0
        }
        
        print(f"{name} completed in {elapsed_time:.4f} seconds with {len(moves)} moves")
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"explorer_comparison_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump({
            'metadata': {
                'timestamp': timestamp,
                'maze_type': maze_type,
                'width': width,
                'height': height
            },
            'results': results
        }, f, indent=2)
    
    print(f"\nResults saved to {filename}")
    
    # Print comparison summary
    print("\n===== ALGORITHM COMPARISON =====")
    print(f"{'Algorithm':<20} {'Moves':<10} {'Time (s)':<12} {'Backtracks':<12} {'Moves/s':<12}")
    print("-" * 65)
    
    for name, result in results.items():
        print(f"{name:<20} {result['moves']:<10} {result['elapsed_time']:<12.6f} "
              f"{result['backtracks']:<12} {result['moves_per_second']:<12.2f}")
    
    # Identify the best algorithm by moves and time
    best_by_moves = min(results.items(), key=lambda x: x[1]['moves'])
    best_by_time = min(results.items(), key=lambda x: x[1]['elapsed_time'])
    
    print("\n===== BEST ALGORITHMS =====")
    print(f"Best by moves: {best_by_moves[0]} with {best_by_moves[1]['moves']} moves")
    print(f"Best by time: {best_by_time[0]} with {best_by_time[1]['elapsed_time']:.6f} seconds")
    
    # Calculate improvement percentages
    baseline_moves = results["Right-Hand Rule"]["moves"]
    baseline_time = results["Right-Hand Rule"]["elapsed_time"]
    
    for name, result in results.items():
        if name != "Right-Hand Rule":
            move_improvement = (baseline_moves - result["moves"]) / baseline_moves * 100
            time_diff = (result["elapsed_time"] - baseline_time) / baseline_time * 100
            
            print(f"\n{name} vs Right-Hand Rule:")
            print(f"  Move reduction: {move_improvement:.2f}%")
            print(f"  Time difference: {time_diff:.2f}%")
    
    # Generate and save visualizations
    create_visualizations(results, timestamp)
    
    return results

def create_visualizations(results, timestamp):
    """
    Create visualizations of the comparison results.
    
    Args:
        results: Dictionary of algorithm results
        timestamp: Timestamp for file naming
    """
    # Set up the figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Extract data for plotting
    algorithms = list(results.keys())
    moves = [results[algo]['moves'] for algo in algorithms]
    times = [results[algo]['elapsed_time'] for algo in algorithms]
    
    # Color mapping
    colors = ['blue', 'green', 'red']
    
    # Create bar chart for number of moves
    ax1.bar(algorithms, moves, color=colors)
    ax1.set_title('Number of Moves by Algorithm')
    ax1.set_ylabel('Number of Moves')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of each bar
    for i, v in enumerate(moves):
        ax1.text(i, v + 5, str(v), ha='center')
    
    # Create bar chart for execution time
    ax2.bar(algorithms, times, color=colors)
    ax2.set_title('Execution Time by Algorithm')
    ax2.set_ylabel('Time (seconds)')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of each bar
    for i, v in enumerate(times):
        ax2.text(i, v + 0.0001, f"{v:.6f}", ha='center')
    
    # Adjust layout and save figure
    plt.tight_layout()
    plt.savefig(f"explorer_comparison_{timestamp}.png")
    print(f"Visualization saved to explorer_comparison_{timestamp}.png")

def main():
    parser = argparse.ArgumentParser(description="Compare Maze Explorer Algorithms")
    parser.add_argument("--type", choices=["random", "static"], default="static",
                        help="Type of maze to generate (random or static)")
    parser.add_argument("--width", type=int, default=30,
                        help="Width of the maze (default: 30, ignored for static mazes)")
    parser.add_argument("--height", type=int, default=30,
                        help="Height of the maze (default: 30, ignored for static mazes)")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize the exploration in real-time")
    
    args = parser.parse_args()
    
    print(f"Comparing maze explorers on {args.type} maze...")
    if args.type == "random":
        print(f"Maze dimensions: {args.width}x{args.height}")
    
    run_comparison(
        maze_type=args.type,
        width=args.width,
        height=args.height,
        visualize=args.visualize
    )

if __name__ == "__main__":
    main()