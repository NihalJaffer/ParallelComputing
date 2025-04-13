"""
Visualization utilities for the maze game.
"""

import pygame
import time
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from .constants import WINDOW_SIZE, CELL_SIZE, WHITE, BLACK, RED, GREEN, BLUE
from .explorer import Explorer

def visualize_maze(maze, screen):
    """
    Visualize a maze on the given screen.
    
    Args:
        maze: The maze to visualize
        screen: The Pygame screen to draw on
    """
    # Clear the screen
    screen.fill(WHITE)
    
    # Draw maze
    for y in range(maze.height):
        for x in range(maze.width):
            if maze.grid[y][x] == 1:
                pygame.draw.rect(screen, BLACK,
                               (x * CELL_SIZE, y * CELL_SIZE,
                                CELL_SIZE, CELL_SIZE))
    
    # Draw start and end points
    pygame.draw.rect(screen, GREEN,
                    (maze.start_pos[0] * CELL_SIZE,
                     maze.start_pos[1] * CELL_SIZE,
                     CELL_SIZE, CELL_SIZE))
    pygame.draw.rect(screen, RED,
                    (maze.end_pos[0] * CELL_SIZE,
                     maze.end_pos[1] * CELL_SIZE,
                     CELL_SIZE, CELL_SIZE))
    
    # Update the display
    pygame.display.flip()
    
    # Convert Pygame surface to numpy array for display
    data = pygame.surfarray.array3d(screen)
    data = np.transpose(data, (1, 0, 2))
    
    # Display the maze
    plt.figure(figsize=(10, 10))
    plt.imshow(data)
    plt.axis('off')
    plt.show()

class JupyterExplorer(Explorer):
    """
    Explorer class adapted for Jupyter notebook visualization.
    """
    def __init__(self, maze, screen):
        super().__init__(maze, visualize=True)
        self.screen = screen
        
    def draw_state(self):
        """Override draw_state to work with Jupyter"""
        # Clear the screen
        self.screen.fill(WHITE)
        
        # Draw maze
        for y in range(self.maze.height):
            for x in range(self.maze.width):
                if self.maze.grid[y][x] == 1:
                    pygame.draw.rect(self.screen, BLACK,
                                   (x * CELL_SIZE, y * CELL_SIZE,
                                    CELL_SIZE, CELL_SIZE))
        
        # Draw start and end points
        pygame.draw.rect(self.screen, GREEN,
                        (self.maze.start_pos[0] * CELL_SIZE,
                         self.maze.start_pos[1] * CELL_SIZE,
                         CELL_SIZE, CELL_SIZE))
        pygame.draw.rect(self.screen, RED,
                        (self.maze.end_pos[0] * CELL_SIZE,
                         self.maze.end_pos[1] * CELL_SIZE,
                         CELL_SIZE, CELL_SIZE))
        
        # Draw explorer
        pygame.draw.rect(self.screen, BLUE,
                        (self.x * CELL_SIZE, self.y * CELL_SIZE,
                         CELL_SIZE, CELL_SIZE))
        
        # Update the display
        pygame.display.flip()
        
        # Convert Pygame surface to numpy array for display
        data = pygame.surfarray.array3d(self.screen)
        data = np.transpose(data, (1, 0, 2))
        
        # Display the current state
        plt.figure(figsize=(10, 10))
        plt.imshow(data)
        plt.axis('off')
        plt.show()
        
        # Small delay to see the movement
        time.sleep(0.1)
        
        # Clear the output for the next frame
        clear_output(wait=True) 