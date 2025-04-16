"""
A* Search implementation for maze exploration.
"""

import time
import heapq
from typing import Tuple, List, Dict, Set
import pygame
from .explorer import Explorer

class AStarExplorer(Explorer):
    """
    Enhanced explorer using A* search algorithm for finding the shortest path.
    A* uses a best-first search approach with a heuristic to efficiently
    find the shortest path to the goal.
    """
    
    def __init__(self, maze, visualize: bool = False):
        super().__init__(maze, visualize)
        # Additional A* specific data structures
        self.open_set = []  # Priority queue for nodes to explore
        self.closed_set = set()  # Set of already explored nodes
        self.came_from = {}  # Dictionary to reconstruct path
        self.g_score = {}  # Cost from start to current node
        self.f_score = {}  # Estimated total cost (g_score + heuristic)
    
    def heuristic(self, pos: Tuple[int, int]) -> int:
        """
        Manhattan distance heuristic to estimate distance to goal.
        This is admissible (never overestimates) for grid-based movement.
        """
        x1, y1 = pos
        x2, y2 = self.maze.end_pos
        return abs(x1 - x2) + abs(y1 - y2)
    
    def reconstruct_path(self, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Reconstruct the path from start to goal using the came_from dictionary.
        """
        total_path = [current]
        while current in self.came_from:
            current = self.came_from[current]
            total_path.append(current)
        
        # The path is constructed from goal to start, so reverse it
        return total_path[::-1]
    
    def solve(self) -> Tuple[float, List[Tuple[int, int]]]:
        """
        Solve the maze using A* search algorithm.
        Returns the time taken and the list of moves made.
        """
        self.start_time = time.time()
        
        # Initialize A* data structures
        start = (self.x, self.y)
        goal = self.maze.end_pos
        
        # Add start node to open set with priority 0
        heapq.heappush(self.open_set, (0, start))
        
        # Initialize costs
        self.g_score[start] = 0
        self.f_score[start] = self.heuristic(start)
        
        # Track number of nodes explored for statistics
        nodes_explored = 0
        
        while self.open_set:
            # Get node with lowest f_score from priority queue
            _, current = heapq.heappop(self.open_set)
            nodes_explored += 1
            
            # If we've reached the goal, reconstruct and return the path
            if current == goal:
                self.moves = self.reconstruct_path(current)[1:]  # Exclude starting position
                
                self.end_time = time.time()
                time_taken = self.end_time - self.start_time
                
                # Update explorer position for visualization and statistics
                self.x, self.y = goal
                
                # Print statistics
                self.print_statistics(time_taken)
                print(f"Nodes explored: {nodes_explored}")
                
                return time_taken, self.moves
            
            # Add current to closed set
            self.closed_set.add(current)
            
            # Visualize if enabled (show exploration process)
            if self.visualize:
                self.x, self.y = current
                self.draw_state()
            
            # Check all four adjacent neighbors
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # Skip if neighbor is a wall or already explored
                if (not (0 <= neighbor[0] < self.maze.width and 
                       0 <= neighbor[1] < self.maze.height) or
                    self.maze.grid[neighbor[1]][neighbor[0]] == 1 or
                    neighbor in self.closed_set):
                    continue
                
                # Calculate tentative g_score (cost from start to neighbor through current)
                tentative_g_score = self.g_score[current] + 1  # Cost of 1 for each step
                
                # If this path is better than previous ones or neighbor is new
                if neighbor not in self.g_score or tentative_g_score < self.g_score[neighbor]:
                    # Update path and scores
                    self.came_from[neighbor] = current
                    self.g_score[neighbor] = tentative_g_score
                    self.f_score[neighbor] = tentative_g_score + self.heuristic(neighbor)
                    
                    # Add to open set if not already there
                    if not any(n == neighbor for _, n in self.open_set):
                        heapq.heappush(self.open_set, (self.f_score[neighbor], neighbor))
        
        # If we get here, no path was found (should not happen in a valid maze)
        self.end_time = time.time()
        time_taken = self.end_time - self.start_time
        self.print_statistics(time_taken)
        print("No path found to goal!")
        
        return time_taken, self.moves