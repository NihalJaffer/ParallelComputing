"""
Maze module containing the Maze class for generating and managing the game maze.
"""

import random


class Maze:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = [[1 for _ in range(width)] for _ in range(height)]
        # Don't seed the random number generator here to ensure different mazes each time
        self.generate_maze()
        self.set_start_end_positions()

    def generate_maze(self):
        """Generate a random maze using depth-first search algorithm."""
        # Initialize all cells as walls
        self.grid = [[1 for _ in range(self.width)] for _ in range(self.height)]
        
        # Start from a random cell
        start_x = random.randint(1, self.width-2)
        start_y = random.randint(1, self.height-2)
        stack = [(start_x, start_y)]
        self.grid[start_y][start_x] = 0

        while stack:
            current = stack[-1]
            x, y = current

            # Find unvisited neighbors that are two cells away
            neighbors = []
            for dx, dy in [(0, 2), (2, 0), (0, -2), (-2, 0)]:
                nx, ny = x + dx, y + dy
                if 0 < nx < self.width-1 and 0 < ny < self.height-1 and self.grid[ny][nx] == 1:
                    neighbors.append((nx, ny))

            if neighbors:
                # Choose a random neighbor
                next_cell = random.choice(neighbors)
                nx, ny = next_cell

                # Remove wall between current and next cell
                wx, wy = (x + nx) // 2, (y + ny) // 2
                self.grid[wy][wx] = 0
                self.grid[ny][nx] = 0

                stack.append(next_cell)
            else:
                stack.pop()

    def set_start_end_positions(self):
        """Set start and end positions ensuring they are on white squares.
        Start position will be on the topmost line with white cells,
        and end position will be on the bottommost line with white cells."""
        # Find all white squares (0s) in the maze
        white_squares = []
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x] == 0:
                    white_squares.append((x, y))
        
        if not white_squares:
            # If no white squares found, regenerate the maze
            self.generate_maze()
            self.set_start_end_positions()
            return
        
        # Find topmost and bottommost lines with white cells
        topmost_line = min(y for x, y in white_squares)
        bottommost_line = max(y for x, y in white_squares)
        
        # Get white cells on topmost and bottommost lines
        top_white_cells = [(x, y) for x, y in white_squares if y == topmost_line]
        bottom_white_cells = [(x, y) for x, y in white_squares if y == bottommost_line]
        
        if not top_white_cells or not bottom_white_cells:
            # If no white cells on top or bottom, regenerate the maze
            self.generate_maze()
            self.set_start_end_positions()
            return
        
        # Choose random positions from top and bottom lines
        self.start_pos = random.choice(top_white_cells)
        self.end_pos = random.choice(bottom_white_cells)
        
        # Ensure start and end positions are not walls
        self.grid[self.start_pos[1]][self.start_pos[0]] = 0
        self.grid[self.end_pos[1]][self.end_pos[0]] = 0

    def is_wall(self, pos):
        """Check if a position is a wall."""
        x, y = pos
        return self.grid[y][x] == 1


class StaticMaze(Maze):
    # The predefined maze pattern
    STATIC_PATTERN = [
        "11111111111011111111111111111111111111111111111111111111111111",
        "10000000010000000000000001011001000000000001001000000000000101",
        "10010011111111110011111001011011111111001001001001001111111101",
        "10010000000000000010001000000001011001001000000001000000000001",
        "11110010011111111110011110001011001011111111001001001111111111",
        "10000010000010000000000001011000000000000001001001000000000101",
        "10111111110010010010011111111011111111001001001111001111100101",
        "10010000010010010010000000001000001000001001001000001000000001",
        "10010010011110011110011111111011111011111111001111111111111101",
        "10010010010010010000001000001011001011000001000001001000000001",
        "11110011110010010011111111011111001011001111001001001111111101",
        "10000000010000010010000001011001000000001000001000000000100001",
        "11110011110010011110011111001011111011111111111001111111101111",
        "10000000000010000000000000000000000001000001000000000000100001",
        "10111111111111110011111011111111111011111001111001111111111101",
        "10010000000010000000001011000000001000000001000000001000000001",
        "10010011111110011111111001011011111011111001001111111111111101",
        "10000000000010000000001011001000001011001001000000000000000101",
        "11110011111111110010011111111001011111001001111001111111111111",
        "10000010000000000010000001011001011000000000000001000000000101",
        "10111110011111110010001011001011011111001001111001111000101101",
        "10010010000000010010001000001011000001001001001000000000101101",
        "11110011110011111111111111111001011111001111001111001111111101",
        "10000010010000010000001011000001000000000000000001000000000101",
        "11110010011111111110001011111111111111001001111111001111100101",
        "10000000000000000000000000000001000000001001001000000000100001",
        "10111111111110011111111011111111001011111001001001111111101111",
        "10011111111110011111110011111111001011111001001001111111001111",
        "10000010000010010000000000001000001011000000001001001000000001",
        "11110011110010010011111011111111111011111001111001001001111101",
        "10010000010010010010000000000001000000001001001000001000000101",
        "10111110010011111110001011111001011111111001001001111111111101",
        "10000010000000010000001000001000001011000001000001000000101101",
        "10111110011111111111111111111111001011111111111001001111100101",
        "10010010000000010000000000000001000000001000000000001000000001",
        "10010010011110011111111011111001011011111111111001111111100101",
        "10000000000010010000001000001000001000001000001000001000101101",
        "11111110011111110011111111011111111001001111001111111000101111",
        "10000010000010010000000001000000001011001000000000000000101101",
        "10010010010010010011111111111001011111111111111001111001111101",
        "10010000010000000010000000000001000000001001001000001000000001",
        "11111111111111110011111011111011111111111001001001001111111101",
        "10010000000000010010000001000001000000001000000001000000101101",
        "10010011111111110010001011111111001011001001001111111111100101",
        "10010010010000010010001011000001011001001001000001000000101101",
        "10010010010011111111111001011011111011111001001001001000101101",
        "10000000000000000000000000001000000000000001001001001000000001",
        "11111111111110111111111111111111111111111111111111111111111111"
    ]

    def __init__(self, width, height):
        self.width = len(self.STATIC_PATTERN[0])  # Width is fixed by the pattern
        self.height = len(self.STATIC_PATTERN)    # Height is fixed by the pattern
        self.grid = [[1 for _ in range(self.width)] for _ in range(self.height)]
        self.start_pos = (11, 0)  # Position where the pattern has the entrance
        self.end_pos = (13, self.height-1)  # Position where the pattern has the exit (14th cell)
        self.generate_static_maze()
        self.grid[self.start_pos[1]][self.start_pos[0]] = 0
        self.grid[self.end_pos[1]][self.end_pos[0]] = 0

    def generate_static_maze(self):
        """Generate the predefined static maze pattern."""
        for y, row in enumerate(self.STATIC_PATTERN):
            for x, cell in enumerate(row):
                self.grid[y][x] = int(cell)


def create_maze(width, height, maze_type="random"):
    """Create a maze of the specified type."""
    if maze_type == "static":
        return StaticMaze(width, height)  # width and height are ignored for static maze
    return Maze(width, height) 