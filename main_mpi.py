"""
MPI-based Maze Explorer

This script extends the main program to support running explorers
across multiple machines using MPI (Message Passing Interface).
"""

import argparse
import time
import random
import json
import sys
from datetime import datetime
from mpi4py import MPI

# Import local modules
from src.maze import create_maze
from src.explorer import Explorer


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
        'moves_per_second': len(moves) / elapsed_time if elapsed_time > 0 else 0,
        'hostname': MPI.Get_processor_name()
    }


def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Parse arguments only on the root process
    if rank == 0:
        parser = argparse.ArgumentParser(description="MPI Maze Explorer")
        
        parser.add_argument("--num", type=int, default=10,
                          help="Number of explorers to run (default: 10)")
        parser.add_argument("--type", type=str, default="random", choices=["random", "static"],
                          help="Type of maze to generate (default: random)")
        parser.add_argument("--width", type=int, default=30,
                          help="Width of the maze (default: 30, ignored for static mazes)")
        parser.add_argument("--height", type=int, default=30,
                          help="Height of the maze (default: 30, ignored for static mazes)")
        parser.add_argument("--seed", type=int, default=None,
                          help="Random seed for maze generation (default: random)")
        
        args = parser.parse_args()
        
        # Calculate seeds for each task
        base_seed = args.seed if args.seed is not None else random.randint(1, 10000)
        seeds = [base_seed + i for i in range(args.num)]
        
        print(f"Running {args.num} maze explorers in parallel using MPI with {size} processes...")
        print(f"Maze type: {args.type}")
        if args.type == "random":
            print(f"Maze dimensions: {args.width}x{args.height}")
        elif args.type == "static":
            print("Note: Width and height arguments are ignored for the static maze")
        
        # Create task parameters
        tasks = []
        for i in range(args.num):
            tasks.append({
                'seed': seeds[i],
                'maze_type': args.type,
                'width': args.width,
                'height': args.height
            })
    else:
        tasks = None
    
    # Start timing
    start_time = time.time()
    
    # Broadcast the total number of tasks from root to all processes
    if rank == 0:
        num_tasks = len(tasks)
    else:
        num_tasks = None
    num_tasks = comm.bcast(num_tasks, root=0)
    
    # Distribute tasks among processes
    local_tasks = []
    local_results = []
    
    # Scatter tasks to processes (manual distribution since we might have uneven distribution)
    if rank == 0:
        # Determine how many tasks each process should handle
        tasks_per_process = [num_tasks // size + (1 if i < num_tasks % size else 0) for i in range(size)]
        start_idx = 0
        
        # Send tasks to each process
        for i in range(1, size):
            end_idx = start_idx + tasks_per_process[i]
            if start_idx < end_idx:
                comm.send(tasks[start_idx:end_idx], dest=i, tag=i)
            start_idx = end_idx
        
        # Keep root's tasks
        local_tasks = tasks[0:tasks_per_process[0]]
    else:
        # Receive tasks from root
        local_tasks = comm.recv(source=0, tag=rank)
    
    # Process local tasks
    for task in local_tasks:
        result = run_explorer(
            seed=task['seed'], 
            maze_type=task['maze_type'],
            width=task['width'],
            height=task['height']
        )
        local_results.append(result)
    
    # Gather all results to root
    all_results = comm.gather(local_results, root=0)
    
    # Process results on root
    if rank == 0:
        # Flatten results list
        results = [result for sublist in all_results for result in sublist]
        
        # Sort results by number of moves (ascending)
        sorted_results = sorted(results, key=lambda x: x['moves'])
        
        end_time = time.time()
        total_time = end_time - start_time
        
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
            print(f"  Hostname: {result['hostname']}")
        
        # Identify the best explorer
        best = sorted_results[0]
        print("\n===== BEST EXPLORER =====")
        print(f"Seed: {best['seed']}")
        print(f"Moves: {best['moves']}")
        print(f"Backtracks: {best['backtracks']}")
        print(f"Time: {best['elapsed_time']:.4f} seconds")
        print(f"Moves/second: {best['moves_per_second']:.2f}")
        print(f"Hostname: {best['hostname']}")
        
        # Group results by hostname to see distribution
        hostnames = {}
        for result in results:
            hostname = result['hostname']
            if hostname not in hostnames:
                hostnames[hostname] = 0
            hostnames[hostname] += 1
        
        print("\n===== TASK DISTRIBUTION =====")
        for hostname, count in hostnames.items():
            print(f"{hostname}: {count} tasks")
        
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
        filename = f"mpi_explorer_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump({
                'metadata': {
                    'timestamp': timestamp,
                    'num_explorers': len(results),
                    'mpi_processes': size,
                    'maze_type': results[0]['maze_type']
                },
                'results': results,
                'best_explorer': best,
                'hostname_distribution': {host: count for host, count in hostnames.items()},
                'aggregate_stats': {
                    'avg_moves': avg_moves,
                    'min_moves': min_moves,
                    'max_moves': max_moves
                }
            }, f, indent=2)
        
        print(f"\nResults saved to {filename}")
        print(f"Total processing time: {total_time:.4f} seconds")
        
        # Report the speedup achieved by parallelization
        sequential_time = sum(r['elapsed_time'] for r in results)
        print(f"Speedup factor: {sequential_time / total_time:.2f}x")
        print(f"Efficiency: {(sequential_time / total_time) / size:.2f}")


if __name__ == "__main__":
    main()