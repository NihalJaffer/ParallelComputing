"""
Main entry point for the maze runner game.
"""

import argparse
import time
import random
import json
import multiprocessing
from multiprocessing import Pool
from datetime import datetime

from src.game import run_game
from src.explorer import Explorer
from src.maze import create_maze


def run_explorer(seed=None, maze_type="random", width=30, height=30):
    """
    Run a single maze explorer with the given parameters and return its statistics.
    
    Args:
        seed: Random seed for maze generation
        maze_type: Type of maze to generate ("random" or "static")
        width: Width of the maze (for random mazes)
        height: Height of the maze (for random mazes)
        
    Returns:
        dict: Statistics about the explorer's performance
    """
    # If seed is provided, set the random seed
    if seed is not None:
        random.seed(seed)
    
    # Create a maze
    maze = create_maze(width, height, maze_type)
    
    # Create and run an explorer (without visualization)
    explorer = Explorer(maze, visualize=False)
    
    # Solve the maze and get statistics
    elapsed_time, moves = explorer.solve()
    
    # Return statistics as a dictionary
    return {
        'seed': seed,
        'maze_type': maze_type,
        'moves': len(moves),
        'backtracks': explorer.backtrack_count,
        'elapsed_time': elapsed_time,
        'moves_per_second': len(moves) / elapsed_time if elapsed_time > 0 else 0
    }


def run_multiple_explorers(num_explorers, maze_type="random", width=30, height=30, seed=None):
    """
    Run multiple maze explorers in parallel using multiprocessing.
    
    Args:
        num_explorers: Number of explorers to run
        maze_type: Type of maze to generate ("random" or "static")
        width: Width of the maze (for random mazes)
        height: Height of the maze (for random mazes)
        seed: Starting seed (will be incremented for each explorer)
        
    Returns:
        list: List of statistics for each explorer
    """
    # Create a list of seeds for each explorer
    seeds = None
    if seed is not None:
        seeds = [seed + i for i in range(num_explorers)]
    else:
        seeds = [random.randint(1, 10000) for _ in range(num_explorers)]
    
    # Prepare arguments for each explorer
    args = [(seed, maze_type, width, height) for seed in seeds]
    
    # Run explorers in parallel using a process pool
    with Pool(processes=min(num_explorers, multiprocessing.cpu_count())) as pool:
        results = pool.starmap(run_explorer, args)
    
    return results


def display_results(results):
    """
    Display a summary of the results and identify the best explorer.
    
    Args:
        results: List of explorer statistics
    """
    # Sort results by number of moves (ascending)
    sorted_results = sorted(results, key=lambda x: x['moves'])
    
    print("\n===== EXPLORER RESULTS =====")
    print(f"Total explorers: {len(results)}")
    print("\nTOP 5 EXPLORERS:")
    
    for i, result in enumerate(sorted_results[:5]):
        print(f"\nExplorer #{i+1}:")
        print(f"  Seed: {result['seed']}")
        print(f"  Moves: {result['moves']}")
        print(f"  Backtracks: {result['backtracks']}")
        print(f"  Time: {result['elapsed_time']:.4f} seconds")
        print(f"  Moves/second: {result['moves_per_second']:.2f}")
    
    # Identify the best explorer
    best = sorted_results[0]
    print("\n===== BEST EXPLORER =====")
    print(f"Seed: {best['seed']}")
    print(f"Moves: {best['moves']}")
    print(f"Backtracks: {best['backtracks']}")
    print(f"Time: {best['elapsed_time']:.4f} seconds")
    print(f"Moves/second: {best['moves_per_second']:.2f}")
    
    # Calculate aggregate statistics
    total_moves = sum(r['moves'] for r in results)
    avg_moves = total_moves / len(results)
    min_moves = min(r['moves'] for r in results)
    max_moves = max(r['moves'] for r in results)
    
    print("\n===== AGGREGATE STATISTICS =====")
    print(f"Average moves: {avg_moves:.2f}")
    print(f"Minimum moves: {min_moves}")
    print(f"Maximum moves: {max_moves}")
    print(f"Improvement over average: {(avg_moves - min_moves) / avg_moves * 100:.2f}%")
    
    # Also save the results to a file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"explorer_results_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump({
            'metadata': {
                'timestamp': timestamp,
                'num_explorers': len(results),
                'maze_type': results[0]['maze_type']
            },
            'results': results,
            'best_explorer': best,
            'aggregate_stats': {
                'avg_moves': avg_moves,
                'min_moves': min_moves,
                'max_moves': max_moves
            }
        }, f, indent=2)
    
    print(f"\nResults saved to {filename}")
    
    return best


def main():
    parser = argparse.ArgumentParser(description="Maze Runner Game")
    parser.add_argument("--type", choices=["random", "static"], default="random",
                        help="Type of maze to generate (random or static)")
    parser.add_argument("--width", type=int, default=30,
                        help="Width of the maze (default: 30, ignored for static mazes)")
    parser.add_argument("--height", type=int, default=30,
                        help="Height of the maze (default: 30, ignored for static mazes)")
    parser.add_argument("--auto", action="store_true",
                        help="Run automated maze exploration")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize the automated exploration in real-time")
    
    # Add new arguments for parallel explorers
    parser.add_argument("--parallel", action="store_true",
                        help="Run multiple explorers in parallel")
    parser.add_argument("--num", type=int, default=10,
                        help="Number of parallel explorers to run (default: 10)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for maze generation (default: random)")
    
    args = parser.parse_args()
    
    # Check for incompatible options
    if args.parallel and args.visualize:
        print("Error: Cannot use --parallel with --visualize")
        return
    
    # Handle parallel mode
    if args.parallel:
        print(f"Running {args.num} maze explorers in parallel...")
        print(f"Maze type: {args.type}")
        if args.type == "random":
            print(f"Maze dimensions: {args.width}x{args.height}")
        elif args.type == "static":
            print("Note: Width and height arguments are ignored for the static maze")
        
        start_time = time.time()
        
        # Run explorers in parallel
        results = run_multiple_explorers(
            args.num, 
            maze_type=args.type, 
            width=args.width, 
            height=args.height,
            seed=args.seed
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Display results
        best_explorer = display_results(results)
        
        print(f"\nTotal processing time: {total_time:.4f} seconds")
        print(f"Average processing time per explorer: {total_time/args.num:.4f} seconds")
        total_sequential_time = sum(r['elapsed_time'] for r in results)
        print(f"Speedup factor: {total_sequential_time / total_time:.2f}x")
        
        # Return to avoid running the regular game
        return
    
    if args.auto:
        # Create maze and run automated exploration
        maze = create_maze(args.width, args.height, args.type)
        explorer = Explorer(maze, visualize=args.visualize)
        time_taken, moves = explorer.solve()
        print(f"Maze solved in {time_taken:.2f} seconds")
        print(f"Number of moves: {len(moves)}")
        if args.type == "static":
            print("Note: Width and height arguments were ignored for the static maze")
    else:
        # Run the interactive game
        run_game(maze_type=args.type, width=args.width, height=args.height)


if __name__ == "__main__":
    main()