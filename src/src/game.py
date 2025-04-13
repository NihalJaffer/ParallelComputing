"""
Game module containing the main game implementation.
"""

import sys
import pygame
import time

from .constants import (
    WINDOW_SIZE, CELL_SIZE, FPS,
    WHITE, BLACK, RED, GREEN, BLUE
)
from .maze import create_maze
from .player import Player


def run_game(maze_type="random", width=20, height=20):
    """Initialize and run the game."""
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Maze Explorer")
    clock = pygame.time.Clock()

    # Initialize game objects
    maze = create_maze(width, height, maze_type)
    player = Player(maze, CELL_SIZE)
    running = True
    
    # Initialize game stats
    move_count = 0
    start_time = time.time()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    player.move(0, -1)
                    move_count += 1
                elif event.key == pygame.K_DOWN:
                    player.move(0, 1)
                    move_count += 1
                elif event.key == pygame.K_LEFT:
                    player.move(-1, 0)
                    move_count += 1
                elif event.key == pygame.K_RIGHT:
                    player.move(1, 0)
                    move_count += 1

        # Draw everything
        screen.fill(WHITE)
        
        # Draw maze using the maze's actual dimensions
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
        
        # Draw player
        player.draw(screen, BLUE)

        # Check if player reached the end
        if (player.x, player.y) == maze.end_pos:
            end_time = time.time()
            elapsed_time = int(end_time - start_time)
            
            # Display congratulation message
            screen.fill(WHITE)
            font = pygame.font.Font(None, 74)
            text = font.render('Congratulations!', True, BLACK)
            screen.blit(text, (WINDOW_SIZE//2 - 150, WINDOW_SIZE//2 - 100))
            
            font = pygame.font.Font(None, 48)
            moves_text = font.render(f'Moves: {move_count}', True, BLACK)
            screen.blit(moves_text, (WINDOW_SIZE//2 - 100, WINDOW_SIZE//2))
            
            time_text = font.render(f'Time: {elapsed_time} seconds', True, BLACK)
            screen.blit(time_text, (WINDOW_SIZE//2 - 100, WINDOW_SIZE//2 + 50))
            
            pygame.display.flip()
            pygame.time.wait(5000)  # Show message for 5 seconds
            running = False

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit() 