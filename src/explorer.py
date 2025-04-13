"""
Maze Explorer module that implements automated maze solving.
"""

import time
import pygame
from typing import Tuple, List, Optional, Deque
from collections import deque
from .constants import BLUE, WHITE, CELL_SIZE, WINDOW_SIZE

class Explorer:
    def __init__(self, maze, visualize: bool = False):
        self.maze = maze
        self.x, self.y = maze.start_pos
        self.direction = (1, 0)  # Start facing right
        self.moves = []
        self.start_time = None
        self.end_time = None
        self.visualize = visualize
        self.move_history = deque(maxlen=3)  # Keep track of last 3 moves
        self.backtracking = False
        self.backtrack_path = []
        self.backtrack_count = 0  # Count number of backtrack operations
        if visualize:
            pygame.init()
            self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
            pygame.display.set_caption("Maze Explorer - Automated Solving")
            self.clock = pygame.time.Clock()

    def turn_right(self):
        """Turn 90 degrees to the right."""
        x, y = self.direction
        self.direction = (-y, x)

    def turn_left(self):
        """Turn 90 degrees to the left."""
        x, y = self.direction
        self.direction = (y, -x)

    def can_move_forward(self) -> bool:
        """Check if we can move forward in the current direction."""
        dx, dy = self.direction
        new_x, new_y = self.x + dx, self.y + dy
        return (0 <= new_x < self.maze.width and 
                0 <= new_y < self.maze.height and 
                self.maze.grid[new_y][new_x] == 0)

    def move_forward(self):
        """Move forward in the current direction."""
        dx, dy = self.direction
        self.x += dx
        self.y += dy
        current_move = (self.x, self.y)
        self.moves.append(current_move)
        self.move_history.append(current_move)
        if self.visualize:
            self.draw_state()

    def is_stuck(self) -> bool:
        """Check if the explorer is stuck in a loop."""
        if len(self.move_history) < 3:
            return False
        # Check if the last 3 moves are the same
        return (self.move_history[0] == self.move_history[1] == self.move_history[2])

    def backtrack(self) -> bool:
        """Backtrack to the last position where we had multiple choices."""
        if not self.backtrack_path:
            # If we don't have a backtrack path, find one
            self.backtrack_path = self.find_backtrack_path()
        
        if self.backtrack_path:
            # Move to the next position in the backtrack path
            next_pos = self.backtrack_path.pop()
            self.x, self.y = next_pos
            self.backtrack_count += 1
            if self.visualize:
                self.draw_state()
            return True
        return False

    def find_backtrack_path(self) -> List[Tuple[int, int]]:
        """Find a path back to a position with multiple choices."""
        # Start from current position and go backwards through moves
        path = []
        current_pos = (self.x, self.y)
        visited = set()
        
        # Look for a position where we had multiple choices
        for i in range(len(self.moves) - 1, -1, -1):
            pos = self.moves[i]
            if pos in visited:
                continue
            visited.add(pos)
            path.append(pos)
            
            # Check if this position had multiple choices
            choices = self.count_available_choices(pos)
            if choices > 1:
                return path[::-1]  # Return reversed path
        
        return path[::-1]  # Return reversed path if no better position found

    def count_available_choices(self, pos: Tuple[int, int]) -> int:
        """Count the number of available moves from a position."""
        x, y = pos
        choices = 0
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            new_x, new_y = x + dx, y + dy
            if (0 <= new_x < self.maze.width and 
                0 <= new_y < self.maze.height and 
                self.maze.grid[new_y][new_x] == 0):
                choices += 1
        return choices

    def draw_state(self):
        """Draw the current state of the maze and explorer."""
        self.screen.fill(WHITE)
        
        # Draw maze
        for y in range(self.maze.height):
            for x in range(self.maze.width):
                if self.maze.grid[y][x] == 1:
                    pygame.draw.rect(self.screen, (0, 0, 0),
                                   (x * CELL_SIZE, y * CELL_SIZE,
                                    CELL_SIZE, CELL_SIZE))
        
        # Draw start and end points
        pygame.draw.rect(self.screen, (0, 255, 0),
                        (self.maze.start_pos[0] * CELL_SIZE,
                         self.maze.start_pos[1] * CELL_SIZE,
                         CELL_SIZE, CELL_SIZE))
        pygame.draw.rect(self.screen, (255, 0, 0),
                        (self.maze.end_pos[0] * CELL_SIZE,
                         self.maze.end_pos[1] * CELL_SIZE,
                         CELL_SIZE, CELL_SIZE))
        
        # Draw explorer
        pygame.draw.rect(self.screen, BLUE,
                        (self.x * CELL_SIZE, self.y * CELL_SIZE,
                         CELL_SIZE, CELL_SIZE))
        
        pygame.display.flip()
        self.clock.tick(30)  # Control visualization speed

    def print_statistics(self, time_taken: float):
        """Print detailed statistics about the exploration."""
        print("\n=== Maze Exploration Statistics ===")
        print(f"Total time taken: {time_taken:.2f} seconds")
        print(f"Total moves made: {len(self.moves)}")
        print(f"Number of backtrack operations: {self.backtrack_count}")
        print(f"Average moves per second: {len(self.moves)/time_taken:.2f}")
        print("==================================\n")

    def solve(self) -> Tuple[float, List[Tuple[int, int]]]:
        """
        Solve the maze using the right-hand rule algorithm with backtracking.
        Returns the time taken and the list of moves made.
        """
        self.start_time = time.time()
        
        # Keep track of visited positions to detect loops
        visited = set()
        visited.add((self.x, self.y))
        
        if self.visualize:
            self.draw_state()
        
        while (self.x, self.y) != self.maze.end_pos:
            if self.is_stuck():
                # If stuck, try backtracking
                if not self.backtrack():
                    # If backtracking fails, try a different direction
                    self.turn_left()
                    self.turn_left()  # Turn around
                    self.move_forward()
                self.backtracking = True
            else:
                self.backtracking = False
                # Try to turn right first
                self.turn_right()
                if self.can_move_forward():
                    self.move_forward()
                    visited.add((self.x, self.y))
                else:
                    # If we can't move right, try forward
                    self.turn_left()
                    if self.can_move_forward():
                        self.move_forward()
                        visited.add((self.x, self.y))
                    else:
                        # If we can't move forward, try left
                        self.turn_left()
                        if self.can_move_forward():
                            self.move_forward()
                            visited.add((self.x, self.y))
                        else:
                            # If we can't move left, turn around
                            self.turn_left()
                            self.move_forward()
                            visited.add((self.x, self.y))

        self.end_time = time.time()
        time_taken = self.end_time - self.start_time
        
        if self.visualize:
            # Show final state for a few seconds
            pygame.time.wait(2000)
            pygame.quit()
        
        # Print detailed statistics
        self.print_statistics(time_taken)
            
        return time_taken, self.moves 