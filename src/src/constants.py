"""
Constants module containing all game-related constants.
"""

# Window settings
WINDOW_SIZE = 768  # Adjusted to fit the 64x64 static maze (64 * 12 = 768)
CELL_SIZE = 12  # Reduced from 20 to 12 to make cells smaller
GRID_SIZE = WINDOW_SIZE // CELL_SIZE
FPS = 60

# Maze dimensions
MAZE_WIDTH = 30
MAZE_HEIGHT = 30

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255) 