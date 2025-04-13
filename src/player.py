"""
Player module containing the Player class for managing player movement and rendering.
"""

import pygame


class Player:
    def __init__(self, maze, cell_size):
        self.maze = maze
        self.x, self.y = maze.start_pos
        self.cell_size = cell_size
        self.radius = cell_size // 3

    def move(self, dx, dy):
        """Move the player if the new position is valid."""
        new_x = self.x + dx
        new_y = self.y + dy
        
        # Check if the new position is within bounds and not a wall
        if (0 <= new_x < self.maze.width and 0 <= new_y < self.maze.height and 
            self.maze.grid[new_y][new_x] == 0):
            self.x = new_x
            self.y = new_y

    def draw(self, surface, color):
        """Draw the player on the given surface."""
        pygame.draw.circle(surface, color, 
                         (self.x * self.cell_size + self.cell_size // 2,
                          self.y * self.cell_size + self.cell_size // 2),
                         self.radius) 