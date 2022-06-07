import copy
import time
import heapq
import random

class Puzzle:
    """A sliding-block puzzle."""

    possible_directions = ['N', 'S', 'E', 'W']
    dirs_to_moves = {'N': (-1, 0), 'S': (1, 0), 'E': (0, 1), 'W': (0, -1)}
    moves_to_dirs = {value: key for key, value in dirs_to_moves.items()}
    opposite_dirs = {'N':'S', 'S':'N', 'E':'W', 'W':'E'}
  
    def __init__(self, grid):
        """Instances differ by their number configurations."""
        self.grid = copy.deepcopy(grid) # No aliasing!
        self.grid_size = len(self.grid)
    
    def display(self):
        """Print the puzzle."""
        for row in self.grid:
            for number in row:
                print(number, end="")
            print()
        print()

    def find_coord_of_num(self, grid, num):
        """Return the (x,y) coordinates of the number/blank tile as a tuple."""
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == num:
                    return (i, j)

        return (-1, -1)

    def moves(self):
        """Return a list of possible moves given the current configuration."""
        # YOU FILL THIS IN
        move_list = []
        blank_tile_loc = self.find_coord_of_num(self.grid, ' ')
        for direction in Puzzle.possible_directions:
            move = Puzzle.dirs_to_moves[direction]
            # Make sure the move is within the bounds of the grid.
            if 0 <= blank_tile_loc[0]+move[0] < self.grid_size:
                if 0 <= blank_tile_loc[1]+move[1] < self.grid_size:
                    move_list.append(direction)

        return move_list
    
    def neighbor(self, move):
        """Return a Puzzle instance like this one but with one move made."""
        # YOU FILL THIS IN
        new_grid = copy.deepcopy(self.grid)
        blank_tile_loc = self.find_coord_of_num(self.grid, ' ')
        move_coord = Puzzle.dirs_to_moves[move]
        new_coord = (blank_tile_loc[0]+move_coord[0], blank_tile_loc[1]+move_coord[1])
        # Switch the blank tile with the other number tile.
        temp = new_grid[new_coord[0]][new_coord[1]]
        new_grid[new_coord[0]][new_coord[1]] = ' '
        new_grid[blank_tile_loc[0]][blank_tile_loc[1]] = temp

        return Puzzle(new_grid)

    def get_manhattan_dist(self, first_coord, second_coord):
        """Return the Manhattan distance between two coordinates."""
        return abs(first_coord[0]-second_coord[0]) + abs(first_coord[1]-second_coord[1])

    def h(self, goal):
        """Compute the Manhattan distance heuristic from this instance to the goal."""
        # YOU FILL THIS IN
        h_value = 0

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                num = self.grid[i][j]
                if num == ' ':  # ignore the blank tile.
                    continue
                goal_coord = self.find_coord_of_num(goal.grid, num)
                man_dist = self.get_manhattan_dist((i,j), goal_coord)
                h_value += man_dist

        return h_value

    # def h(self, goal):
    #     """Compute the misplaced tiles distance heuristic from this instance to the goal."""
    #     h_value = 0
    #
    #     for i in range(self.grid_size):
    #         for j in range(self.grid_size):
    #             if self.grid[i][j] == ' ':  # ignore the blank tile.
    #                 continue
    #             if self.grid[i][j] != goal.grid[i][j]:
    #                 h_value += 1
    #
    #     return h_value

    def get_min_neighbor(self, goal):
        """Return the neighbor puzzle with the minimum heuristic value."""
        possible_moves = self.moves()
        min_neighbor = self
        min_move = None
        for move in possible_moves:
            puzzle_neighbor = self.neighbor(move)
            if puzzle_neighbor.h(goal) < self.h(goal):
                min_neighbor = puzzle_neighbor
                min_move = move

        return (min_neighbor, min_move)

    def __hash__(self):
        return hash(str(self.grid))

    def __eq__(self, other):
        return (
            self.__class__ == other.__class__ and
            self.grid == other.grid
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        return self.__hash__() < other.__hash__()


class Agent:
    """Knows how to solve a sliding-block puzzle with A* search."""
    num_nodes_explored = 0

    class Node():
        """The Node class is used to create a tree to make path tracing efficient."""

        def __init__(self, data, parent=None):
            self.data = data
            self.children = []
            self.parent = parent

        def get_depth(self):
            depth = 0
            curr_node = self
            while curr_node:
                curr_node = curr_node.parent
                depth += 1

            return depth

        def __hash__(self):
            return hash(self.data) + hash(str(self.children))

        def __eq__(self, other):
            return (
                self.__class__ == other.__class__ and
                self.data == other.data and
                self.children == other.children
            )

        def __ne__(self, other):
            return not self.__eq__(other)

        def __lt__(self, other):
            return self.get_depth() < other.get_depth()
    
    def astar(self, puzzle, goal):
        """Return a list of moves to get the puzzle to match the goal."""
        # YOU FILL THIS IN
        # Use heapq to implement a priority queue.
        root = Agent.Node(puzzle.find_coord_of_num(puzzle.grid, ' '))
        node = None
        finished = set()  # to keep track of puzzle arrangements already visited.
        finished.add(puzzle)
        frontier = []  # priority queue.
        # ^^ Puzzles arrangements with lower cost go to the front of the queue.
        puzzle_f = puzzle.h(goal)  # total cost of the puzzle.
        heapq.heappush(frontier, (puzzle_f, root, puzzle))
        Agent.num_nodes_explored = 0  # reset value when starting new search.

        # Keep popping nodes from the frontier until the node popped is the
        # goal node or we run out of nodes to pop.
        while frontier:
            Agent.num_nodes_explored += 1
            curr_puzzle_f, node, curr_puzzle = heapq.heappop(frontier)
            if curr_puzzle == goal:  # found solution!
                break
            curr_puzzle_h = curr_puzzle.h(goal)  # heuristic value of current puzzle.
            g_cost = curr_puzzle_f - curr_puzzle_h  # cost so far to reach the goal.

            for direction in curr_puzzle.moves():
                move_coord = Puzzle.dirs_to_moves[direction]
                new_coord = (node.data[0]+move_coord[0], node.data[1]+move_coord[1])
                child_node = Agent.Node(new_coord, node)
                node.children.append(child_node)
                puzzle_neighbor = curr_puzzle.neighbor(direction)
                if puzzle_neighbor in finished: # Avoid previously visited states.
                    continue
                puzzle_neighbor_h = puzzle_neighbor.h(goal) # heuristic value of neighbor.
                # The cost to go from one node to the next is 1.
                # f(n) = g(n) + h(n)
                puzzle_neighbor_f = 1 + g_cost + puzzle_neighbor_h
                heapq.heappush(frontier, (puzzle_neighbor_f, child_node, puzzle_neighbor))
                finished.add(puzzle_neighbor)

        # Get the path from the start to the goal node by traversing up the tree, all the way to root.
        path_moves = []
        while node:
            path_moves.append(node.data)
            node = node.parent
        path_moves.reverse()

        # Convert the path moves to path directions.
        path_directions = []
        for i in range(len(path_moves) - 1):
            path_directions.append(Puzzle.moves_to_dirs[(path_moves[i+1][0] - path_moves[i][0],
                                                       path_moves[i+1][1] - path_moves[i][1])])

        return path_directions

    def random_walk(self, puzzle, goal, move_limit):
        """
        Use random walk search to get from the puzzle to the goal.
        Return a list of moves to get the puzzle to match the goal.
        """
        path_directions = []
        curr_puzzle = puzzle
        prev_direction = ''
        move_count = 0
        Agent.num_nodes_explored = 0  # reset value when starting new search.

        while move_count < move_limit:
            # Take random walks around the puzzle but avoid going back to the previous state.
            possible_moves = curr_puzzle.moves()
            if prev_direction != '':
                possible_moves.remove(curr_puzzle.opposite_dirs[prev_direction])
            rand_direction = random.choice(possible_moves)
            curr_puzzle = curr_puzzle.neighbor(rand_direction)
            prev_direction = rand_direction
            path_directions.append(rand_direction)
            move_count += 1
            Agent.num_nodes_explored += 1
            if curr_puzzle == goal:
                return path_directions

        return path_directions

    def hill_climbing(self, puzzle, goal):
        """
        Use hill climbing search (gradient descent) to get from the puzzle to the goal.
        Return a list of moves to get the puzzle to match the goal.
        """
        path_directions = []
        curr_puzzle = puzzle
        Agent.num_nodes_explored = 0  # reset value when starting new search.
        min_neighbor, min_move = curr_puzzle.get_min_neighbor(goal)

        # Climb down the valley.
        while curr_puzzle != min_neighbor:
            path_directions.append(min_move)
            curr_puzzle = min_neighbor
            min_neighbor, min_move = curr_puzzle.get_min_neighbor(goal)

        return path_directions


def main():
    """Create a puzzle, solve it with A*, and console-animate."""
    
    puzzle = Puzzle([[1, 2, 5], [4, 8, 7], [3, 6, ' ']])
    #puzzle = Puzzle([[1, 3, 2], [4, 6, 5], [' ', 7, 8]]) # for timing puzzle 3.
    puzzle.display()
    
    agent = Agent()
    goal = Puzzle([[' ', 1, 2], [3, 4, 5], [6, 7, 8]])
    path = agent.astar(puzzle, goal)
    
    while path:
        move = path.pop(0)
        puzzle = puzzle.neighbor(move)
        time.sleep(1)
        puzzle.display()

    # Measuring runtime performance of algorithms:
    # num_runs = 10000

    # Astar search:
    # astar_total_time = 0
    # for i in range(num_runs):
    #     start_time = time.time()
    #     path = agent.astar(puzzle, goal)
    #     end_time = time.time()
    #     time_elapsed = end_time - start_time
    #     astar_total_time += time_elapsed
    # print("A* search:")
    # print("Time taken to find the solution using the Manhattan distance heuristic: %.3f ms." % (astar_total_time/num_runs * 10**3))
    # print(f"Number of nodes explored: {Agent.num_nodes_explored}")

    # Random walk:
    # rwalk_solve_time = 0
    # num_solutions = 0
    # for i in range(num_runs):
    #     start_time = time.time()
    #     path = agent.random_walk(puzzle, goal, 20)
    #     end_time = time.time()
    #     time_elapsed = end_time - start_time
    #     if path == ['N', 'E', 'N', 'E', 'S', 'S', 'W', 'W', 'N', 'E', 'S', 'E', 'N', 'N', 'W', 'W']:
    #         rwalk_solve_time += time_elapsed
    #         num_solutions += 1
    # print("Random walk:")
    # print(f"Number of times the agent finds a solution: {num_solutions}")
    # print("Percent of times the random agent finds a solution: %.2f%%." % (num_solutions/num_runs))
    # if num_solutions > 0:
    #     print("Average time taken to find a solution: %.3f ms." % (rwalk_solve_time/num_solutions * 10**3))
    # else:
    #     print("No solution found!")

    # Hill climbing:
    # print("Hill climbing: (gradient descent)")
    # start_time = time.time()
    # path = agent.hill_climbing(puzzle, goal)
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # if path == ['N', 'E', 'N', 'E', 'S', 'S', 'W', 'W', 'N', 'E', 'S', 'E', 'N', 'N', 'W', 'W']:
    #     print("Solution found!")
    #     print("Time taken to find the solution using the Manhattan distance heuristic: %.3f ms." % (elapsed_time * 10**3))
    # else:
    #     print("No solution found!")
    # print("Solution path: " + str(path))

if __name__ == '__main__':
    main()