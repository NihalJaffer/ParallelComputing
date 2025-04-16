"""
Main entry point for running multiple maze explorers using Celery.
"""

import argparse
import time
import random
import json
from datetime import datetime
from celery import group
from celery.result import allow_join_result

from src.celery_tasks import explore_maze

def display_results(results):
    """
    Display a summary of the results and identify the best explorer.
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
    
    # Save results to a file
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
    parser = argparse.ArgumentParser(description="Maze Explorer with Celery")
    parser.add_argument("--type", choices=["random", "static"], default="static",
                        help="Type of maze to generate (random or static)")
    parser.add_argument("--width", type=int, default=30,
                        help="Width of the maze (default: 30, ignored for static mazes)")
    parser.add_argument("--height", type=int, default=30,
                        help="Height of the maze (default: 30, ignored for static mazes)")
    parser.add_argument("--num", type=int, default=50,
                        help="Number of explorers to run (default: 50)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Starting random seed (default: random)")
    
    args = parser.parse_args()
    
    print(f"Running {args.num} maze explorers using Celery...")
    print(f"Maze type: {args.type}")
    if args.type == "random":
        print(f"Maze dimensions: {args.width}x{args.height}")
    
    start_time = time.time()
    
    # Generate seeds for all explorers
    if args.seed is not None:
        seeds = [args.seed + i for i in range(args.num)]
    else:
        seeds = [random.randint(1, 10000) for _ in range(args.num)]
    
    # Create a group of tasks
    tasks = group(explore_maze.s(
        seed=s,
        maze_type=args.type,
        width=args.width,
        height=args.height
    ) for s in seeds)
    
    print(f"Submitting {args.num} tasks to Celery workers...")
    
    # Execute tasks
    results = tasks.apply_async()
    
    # Monitor progress
    completed = 0
    total = len(results)
    
    while not results.ready():
        new_completed = sum(1 for r in results if r.ready())
        if new_completed > completed:
            completed = new_completed
            print(f"Progress: {completed}/{total} tasks completed ({completed/total*100:.1f}%)")
        time.sleep(1)
    
    print(f"All {total} tasks completed!")
    
    # Get results
    with allow_join_result():
        explorer_results = results.get()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Display results
    best_explorer = display_results(explorer_results)
    
    print(f"\nTotal processing time: {total_time:.4f} seconds")
    print(f"Average processing time per explorer: {total_time/args.num:.4f} seconds")
    total_sequential_time = sum(r['elapsed_time'] for r in explorer_results)
    print(f"Speedup factor: {total_sequential_time / total_time:.2f}x")

if __name__ == "__main__":
    main()