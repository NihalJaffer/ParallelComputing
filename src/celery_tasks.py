"""
Celery tasks for running maze explorers.
"""

import random
from celery import Celery
from src.explorer import Explorer
from src.maze import create_maze

# Set up Celery app
app = Celery('maze_explorer',
             broker='amqp://guest:guest@localhost:5672//',
             backend='rpc://')

# Configure Celery
app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)

@app.task
def explore_maze(seed=None, maze_type="random", width=30, height=30):
    """
    Celery task to run a single maze explorer.
    """
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
    
    # Create maze
    maze = create_maze(width, height, maze_type)
    
    # Create and run explorer (without visualization)
    explorer = Explorer(maze, visualize=False)
    
    # Solve maze and get statistics
    elapsed_time, moves = explorer.solve()
    
    # Return statistics
    return {
        'seed': seed,
        'maze_type': maze_type,
        'moves': len(moves),
        'backtracks': explorer.backtrack_count,
        'elapsed_time': elapsed_time,
        'moves_per_second': len(moves) / elapsed_time if elapsed_time > 0 else 0
    }