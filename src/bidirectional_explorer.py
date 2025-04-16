"""
Bidirectional Search implementation for maze exploration.
"""

import time
from collections import deque
from typing import Tuple, List, Dict, Set, Deque
import pygame
from .explorer import Explorer

class BidirectionalExplorer(Explorer):
    """
    Enhanced explorer using bidirectional search for faster path finding.
    This algorithm searches simultaneously from the start and goal positions,
    potentially finding the path much faster than unidirectional search.
    """
    
    def __init__(self, maze, visualize: bool = False):
        super().__init__(maze, visualize)
        # Additional data structures for bidirectional search
        self.forward_visited = set()  # Nodes visited from start
        self.backward_visited = set()  # Nodes visited from goal
        self.forward_queue = deque()   # BFS queue from start
        self.backward_queue = deque()  # BFS queue from goal
        self.forward_parent = {}       # Parent pointers for forward search
        self.backward_parent = {}      # Parent pointers for backward search
    
    def reconstruct_path(self, meeting_point: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Reconstruct the path from both directions meeting at the specified point.
        """
        # Reconstruct path from start to meeting point
        forward_path = []
        current = meeting_point
        while current in self.forward_parent:
            forward_path.append(current)
            current = self.forward_parent[current]
        forward_path.append(self.maze.start_pos)  # Add start position
        forward_path.reverse()  # Reverse to get path from start to meeting point
        
        # Reconstruct path from meeting point to goal
        backward_path = []
        current = meeting_point
        while current in self.backward_parent:
            current = self.backward_parent[current]
            backward_path.append(current)
        
        # Combine paths (exclude meeting point from backward path to avoid duplication)
        full_path = forward_path + backward_path
        
        # Return path excluding the start position (for consistency with base Explorer)
        return full_path[1:]
    
    def solve(self) -> Tuple[float, List[Tuple[int, int]]]:
        """
        Solve the maze using bidirectional search.
        Returns the time taken and the list of moves made.
        """
        self.start_time = time.time()
        
        # Initialize search from start
        start = self.maze.start_pos
        self.forward_queue.append(start)
        self.forward_visited.add(start)
        
        # Initialize search from goal
        goal = self.maze.end_pos
        self.backward_queue.append(goal)
        self.backward_visited.add(goal)
        
        # Track number of nodes explored for statistics
        nodes_explored = 0
        
        # Run bidirectional BFS until a meeting point is found
        meeting_point = None
        
        while self.forward_queue and self.backward_queue:
            # Alternate between forward and backward search to ensure balanced exploration
            
            # Process a node from forward direction
            if self.forward_queue:
                current = self.forward_queue.popleft()
                nodes_explored += 1
                
                # Visualize if enabled
                if self.visualize:
                    self.x, self.y = current
                    self.draw_state()
                
                # Check if forward search meets backward search
                if current in self.backward_visited:
                    meeting_point = current
                    break
                
                # Explore neighbors in forward direction
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    neighbor = (current[0] + dx, current[1] + dy)
                    
                    if (0 <= neighbor[0] < self.maze.width and 
                        0 <= neighbor[1] < self.maze.height and 
                        self.maze.grid[neighbor[1]][neighbor[0]] == 0 and
                        neighbor not in self.forward_visited):
                        
                        self.forward_queue.append(neighbor)
                        self.forward_visited.add(neighbor)
                        self.forward_parent[neighbor] = current
            
            # Process a node from backward direction
            if self.backward_queue:
                current = self.backward_queue.popleft()
                nodes_explored += 1
                
                # Visualize if enabled (show backward search in a different color if desired)
                if self.visualize:
                    # Optional: temporarily change explorer color for backward search
                    temp_x, temp_y = self.x, self.y
                    self.x, self.y = current
                    self.draw_state()
                    self.x, self.y = temp_x, temp_y
                
                # Check if backward search meets forward search
                if current in self.forward_visited:
                    meeting_point = current
                    break
                
                # Explore neighbors in backward direction
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    neighbor = (current[0] + dx, current[1] + dy)
                    
                    if (0 <= neighbor[0] < self.maze.width and 
                        0 <= neighbor[1] < self.maze.height and 
                        self.maze.grid[neighbor[1]][neighbor[0]] == 0 and
                        neighbor not in self.backward_visited):
                        
                        self.backward_queue.append(neighbor)
                        self.backward_visited.add(neighbor)
                        self.backward_parent[neighbor] = current
        
        # If a meeting point was found, reconstruct the path
        if meeting_point:
            self.moves = self.reconstruct_path(meeting_point)
            self.end_time = time.time()
            time_taken = self.end_time - self.start_time
            
            # Update explorer position for final visualization
            self.x, self.y = self.maze.end_pos
            
            # Print statistics
            self.print_statistics(time_taken)
            print(f"Nodes explored: {nodes_explored}")
            print(f"Forward search visited {len(self.forward_visited)} nodes")
            print(f"Backward search visited {len(self.backward_visited)} nodes")
            
            return time_taken, self.moves
        
        # If no path was found (should not happen in a valid maze)
        self.end_time = time.time()
        time_taken = self.end_time - self.start_time
        self.print_statistics(time_taken)
        print("No path found to goal!")
        
        return time_taken, self.moves