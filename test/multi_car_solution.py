import numpy as np
import pandas as pd
import time
import sys
from mpi4py import MPI


def calculate_multi_car_fitness(routes, distance_matrix, return_to_depot=True):
    """
    Calculate fitness for multiple car routes.
    
    Parameters:
        routes: List of routes, where each route is a list of nodes for one car
        distance_matrix: The distance matrix between nodes
        return_to_depot: Whether cars need to return to the depot (node 0)
        
    Returns:
        Negative total distance (fitness)
    """
    total_distance = 0
    infeasible = False
    
    for route in routes:
        # Check if route starts at depot
        if route[0] != 0:
            infeasible = True
            break
            
        # Calculate route distance
        route_distance = 0
        for i in range(len(route) - 1):
            node1, node2 = route[i], route[i + 1]
            distance = distance_matrix[node1, node2]
            
            if distance == 100000:  # Infeasible connection
                infeasible = True
                break
                
            route_distance += distance
        
        # Add return to depot if required
        if return_to_depot and len(route) > 1:
            last_node = route[-1]
            return_distance = distance_matrix[last_node, 0]
            
            if return_distance == 100000:
                infeasible = True
                break
                
            route_distance += return_distance
            
        total_distance += route_distance
        
        if infeasible:
            break
    
    if infeasible:
        return -1000000  # Large penalty
    else:
        return -total_distance  # Negative because we aim to minimize


def generate_multi_car_population(population_size, num_nodes, num_cars, min_nodes_per_car=3):
    """
    Generate a population of solutions with multiple cars.
    
    Parameters:
        population_size: Number of individuals in the population
        num_nodes: Total number of nodes (including depot)
        num_cars: Number of cars to use
        min_nodes_per_car: Minimum number of nodes each car must visit
        
    Returns:
        List of individuals, where each individual is a list of routes
    """
    population = []
    
    for _ in range(population_size):
        # Create a random permutation of all non-depot nodes
        all_nodes = list(np.random.permutation(np.arange(1, num_nodes)))
        
        # Calculate how many nodes per car (roughly equal distribution)
        nodes_per_car = len(all_nodes) // num_cars
        remainder = len(all_nodes) % num_cars
        
        # Ensure minimum nodes per car
        if nodes_per_car < min_nodes_per_car:
            # If we can't give each car the minimum, we'll use fewer cars
            usable_cars = len(all_nodes) // min_nodes_per_car
            if usable_cars < 1:
                usable_cars = 1  # Always use at least one car
            
            nodes_per_car = len(all_nodes) // usable_cars
            remainder = len(all_nodes) % usable_cars
            actual_cars = usable_cars
        else:
            actual_cars = num_cars
        
        # Create routes for each car
        routes = []
        start_idx = 0
        
        for car in range(actual_cars):
            # Determine how many nodes this car gets
            if car < remainder:
                car_nodes = nodes_per_car + 1
            else:
                car_nodes = nodes_per_car
            
            # Create the route, starting at depot
            if start_idx + car_nodes <= len(all_nodes):
                route = [0] + all_nodes[start_idx:start_idx + car_nodes]
                start_idx += car_nodes
                routes.append(route)
            
        # If we have unassigned nodes, add them to the last car
        if start_idx < len(all_nodes):
            routes[-1].extend(all_nodes[start_idx:])
        
        population.append(routes)
    
    return population


def fix_duplicates(solution):
    """
    Fix a multi-car solution by removing duplicate nodes.
    
    Parameters:
        solution: List of routes
        
    Returns:
        Fixed solution with no duplicates
    """
    # Get all nodes in the solution
    all_nodes = []
    for route in solution:
        all_nodes.extend(route[1:])  # Skip depot
    
    # Find duplicates
    seen = set()
    duplicates = []
    unique_nodes = []
    
    for node in all_nodes:
        if node in seen:
            duplicates.append(node)
        else:
            seen.add(node)
            unique_nodes.append(node)
    
    # If no duplicates, return original
    if not duplicates:
        return solution
    
    # Find missing nodes
    all_possible = set(range(1, max(all_nodes) + 1))
    missing = list(all_possible - seen)
    
    # Replace duplicates with missing nodes if available
    dup_to_missing = {}
    for i in range(min(len(duplicates), len(missing))):
        dup_to_missing[duplicates[i]] = missing[i]
    
    # Apply fixes
    fixed_solution = []
    for route in solution:
        fixed_route = [0]  # Start with depot
        for node in route[1:]:
            if node in dup_to_missing:
                fixed_route.append(dup_to_missing[node])
                del dup_to_missing[node]
            elif node not in seen or node in unique_nodes:
                fixed_route.append(node)
                if node in unique_nodes:
                    unique_nodes.remove(node)
        
        # Only add non-empty routes
        if len(fixed_route) > 1:
            fixed_solution.append(fixed_route)
    
    return fixed_solution


def order_crossover(parent1, parent2):
    """
    Order crossover (OX) for permutations.
    
    Parameters:
        - parent1 (list): The first parent route.
        - parent2 (list): The second parent route.

    Returns:
        - list: The offspring route generated by the crossover.
    """
    size = len(parent1)
    
    # Handle edge cases
    if size <= 1:
        return parent1.copy()
    
    if size != len(parent2):
        # If parents have different lengths, just return the first parent
        return parent1.copy()
    
    # Select two random crossover points
    if size <= 2:
        # For very short routes, just use one crossover point
        start, end = 0, 0
    else:
        # Choose random points, ensuring start <= end
        idx1 = np.random.randint(0, size)
        idx2 = np.random.randint(0, size)
        start, end = min(idx1, idx2), max(idx1, idx2)
    
    # Create offspring
    offspring = [None] * size
    
    # Copy segment from first parent
    offspring[start:end+1] = parent1[start:end+1]
    
    # Get values from second parent that aren't in the copied segment
    segment_values = set(offspring[start:end+1])
    fill_values = [x for x in parent2 if x not in segment_values and x is not None]
    
    # In case we don't have enough fill values (due to duplicates, etc.)
    if len(fill_values) < size - (end - start + 1):
        # Add missing values from parent1
        remaining = [x for x in parent1 if x not in segment_values and x not in fill_values and x is not None]
        fill_values.extend(remaining)
    
    # Fill the rest of the offspring
    idx = 0
    for i in range(size):
        if offspring[i] is None:
            if idx < len(fill_values):
                offspring[i] = fill_values[idx]
                idx += 1
            else:
                # If we run out of fill values, use a value from parent1
                offspring[i] = parent1[i]
    
    return offspring


def crossover_multi_car(parent1, parent2, crossover_type="route"):
    """
    Crossover operator for multi-car solutions.
    
    Parameters:
        parent1: First parent (list of routes)
        parent2: Second parent (list of routes)
        crossover_type: 'route' (swap entire routes) or 'node' (crossover within routes)
        
    Returns:
        Two offspring (list of routes)
    """
    if crossover_type == "route" and len(parent1) > 1 and len(parent2) > 1:
        # Route-level crossover: swap entire routes between parents
        crossover_point = np.random.randint(1, min(len(parent1), len(parent2)))
        
        offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
        offspring2 = parent2[:crossover_point] + parent1[crossover_point:]
        
        # Check for duplicate nodes and fix
        offspring1 = fix_duplicates(offspring1)
        offspring2 = fix_duplicates(offspring2)
        
        return offspring1, offspring2
    else:
        # Node-level crossover: perform order crossover on each route pair
        offspring1 = []
        offspring2 = []
        
        # Pair routes (may be uneven)
        max_routes = max(len(parent1), len(parent2))
        
        for i in range(max_routes):
            if i < len(parent1) and i < len(parent2):
                # Both parents have this route
                route1 = parent1[i]
                route2 = parent2[i]
                
                # Skip depot (index 0) for crossover
                child1 = [0] + order_crossover(route1[1:], route2[1:])
                child2 = [0] + order_crossover(route2[1:], route1[1:])
                
                offspring1.append(child1)
                offspring2.append(child2)
            elif i < len(parent1):
                # Only parent1 has this route
                offspring1.append(parent1[i].copy())
                offspring2.append(parent1[i].copy())
            else:
                # Only parent2 has this route
                offspring1.append(parent2[i].copy())
                offspring2.append(parent2[i].copy())
        
        # Check for duplicate nodes and fix
        offspring1 = fix_duplicates(offspring1)
        offspring2 = fix_duplicates(offspring2)
        
        return offspring1, offspring2


def mutate_multi_car(solution, mutation_rate=0.1, car_rebalance_rate=0.2):
    """
    Mutation operator for multi-car solutions.
    
    Parameters:
        solution: List of routes
        mutation_rate: Probability of mutating a route
        car_rebalance_rate: Probability of rebalancing nodes between cars
        
    Returns:
        Mutated solution
    """
    mutated = [route.copy() for route in solution]
    
    # 1. Route-level mutation (swap nodes within routes)
    for i, route in enumerate(mutated):
        if np.random.rand() < mutation_rate and len(route) > 3:
            # Choose two positions to swap (excluding depot)
            pos1, pos2 = np.random.choice(range(1, len(route)), 2, replace=False)
            route[pos1], route[pos2] = route[pos2], route[pos1]
    
    # 2. Car rebalancing (move nodes between cars)
    if len(mutated) > 1 and np.random.rand() < car_rebalance_rate:
        # Choose source and destination cars
        source_idx, dest_idx = np.random.choice(range(len(mutated)), 2, replace=False)
        source_route = mutated[source_idx]
        dest_route = mutated[dest_idx]
        
        # Only move if source has enough nodes
        if len(source_route) > 2:  # Need at least one node besides depot
            # Choose a node to move
            node_idx = np.random.randint(1, len(source_route))
            node = source_route.pop(node_idx)
            
            # Insert at random position in destination
            insert_pos = np.random.randint(1, len(dest_route) + 1)
            dest_route.insert(insert_pos, node)
    
    return mutated


def tournament_selection(fitness_values, tournament_size):
    """
    Tournament selection for the genetic algorithm.
    
    Parameters:
        fitness_values: List of fitness values
        tournament_size: Number of individuals in each tournament
        
    Returns:
        Index of the selected individual
    """
    # Randomly select tournament_size individuals
    tournament_indices = np.random.choice(len(fitness_values), tournament_size, replace=False)
    tournament_fitness = [fitness_values[i] for i in tournament_indices]
    
    # Find the best individual in the tournament
    winner_idx = np.argmax(tournament_fitness)
    return tournament_indices[winner_idx]


def run_multi_car_ga(distance_matrix_path, num_cars=3, population_size=200, generations=100, min_nodes_per_car=2):
    """
    Run the genetic algorithm for multiple cars.
    
    Parameters:
        distance_matrix_path: Path to the distance matrix CSV
        num_cars: Number of cars to use
        population_size: Size of the population
        generations: Number of generations to run
        min_nodes_per_car: Minimum nodes each car must visit
        
    Returns:
        Best solution found and its fitness
    """
    # Load distance matrix
    try:
        distance_matrix = pd.read_csv(distance_matrix_path).to_numpy()
        print(f"Loaded distance matrix with {distance_matrix.shape[0]} nodes")
    except Exception as e:
        print(f"Error loading distance matrix: {e}")
        return None, None
    
    num_nodes = distance_matrix.shape[0]
    
    # Initialize population
    print(f"Generating initial population for {num_cars} cars...")
    population = generate_multi_car_population(population_size, num_nodes, num_cars, min_nodes_per_car)
    
    # GA parameters
    mutation_rate = 0.2
    crossover_rate = 0.8
    tournament_size = 5
    
    # Track best solution
    best_solution = None
    best_fitness = -float('inf')
    
    # Main GA loop
    start_time = time.time()
    
    for generation in range(generations):
        gen_start = time.time()
        
        # Evaluate fitness
        fitness_values = [calculate_multi_car_fitness(solution, distance_matrix) 
                          for solution in population]
        
        # Find best solution in this generation
        current_best_idx = np.argmax(fitness_values)
        current_best_fitness = fitness_values[current_best_idx]
        
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_solution = [route.copy() for route in population[current_best_idx]]
            print(f"Generation {generation}: New best solution with fitness {best_fitness}")
            
            # Print route details
            print(f"  Car routes:")
            for i, route in enumerate(best_solution):
                distance = 0
                for j in range(len(route) - 1):
                    distance += distance_matrix[route[j], route[j+1]]
                print(f"    Car {i+1}: {route} (distance: {distance})")
        
        if generation % 10 == 0:
            print(f"Generation {generation}: Best fitness = {current_best_fitness}")
        
        # Create next generation
        next_population = []
        
        # Elitism - keep best solution
        next_population.append(population[current_best_idx])
        
        # Tournament selection, crossover, and mutation
        while len(next_population) < population_size:
            # Select parents
            parent1_idx = tournament_selection(fitness_values, tournament_size)
            parent2_idx = tournament_selection(fitness_values, tournament_size)
            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]
            
            # Crossover
            if np.random.rand() < crossover_rate:
                # Alternate between route-level and node-level crossover
                crossover_type = "route" if np.random.rand() < 0.5 else "node"
                offspring1, offspring2 = crossover_multi_car(parent1, parent2, crossover_type)
            else:
                offspring1, offspring2 = [route.copy() for route in parent1], [route.copy() for route in parent2]
            
            # Mutation
            offspring1 = mutate_multi_car(offspring1, mutation_rate)
            offspring2 = mutate_multi_car(offspring2, mutation_rate)
            
            # Add to next generation
            next_population.append(offspring1)
            if len(next_population) < population_size:
                next_population.append(offspring2)
        
        # Replace population
        population = next_population
        
        gen_time = time.time() - gen_start
        if generation % 10 == 0:
            print(f"  Generation time: {gen_time:.2f} seconds")
    
    total_time = time.time() - start_time
    
    print(f"\nGA completed in {total_time:.2f} seconds")
    
    # Final evaluation of best solution
    if best_solution:
        print("\n===== BEST MULTI-CAR SOLUTION =====")
        print(f"Number of cars: {num_cars}")
        total_distance = -best_fitness
        print(f"Total distance: {total_distance}")
        
        print("\nRoutes:")
        for i, route in enumerate(best_solution):
            distance = 0
            for j in range(len(route) - 1):
                node1, node2 = route[j], route[j+1]
                distance += distance_matrix[node1, node2]
            # Add return to depot
            if len(route) > 1:
                distance += distance_matrix[route[-1], 0]
            print(f"Car {i+1}: {route} → Distance: {distance}")
            
        # Calculate load balancing statistics
        route_lengths = [len(route) - 1 for route in best_solution]  # Exclude depot
        min_nodes = min(route_lengths)
        max_nodes = max(route_lengths)
        avg_nodes = sum(route_lengths) / len(route_lengths)
        
        print(f"\nLoad balancing:")
        print(f"  Minimum nodes per car: {min_nodes}")
        print(f"  Maximum nodes per car: {max_nodes}")
        print(f"  Average nodes per car: {avg_nodes:.2f}")
        print(f"  Imbalance ratio: {max_nodes/min_nodes if min_nodes > 0 else 'N/A'}")
    else:
        print("\nNo feasible solution found.")
    
    return best_solution, best_fitness


def main():
    """
    Main function to run the multi-car genetic algorithm.
    """
    # Get command line arguments
    if len(sys.argv) > 1:
        distance_matrix_path = sys.argv[1]
    else:
        distance_matrix_path = 'city_distances_extended.csv'
        if not pd.io.common.file_exists(distance_matrix_path):
            distance_matrix_path = '../data/city_distances_extended.csv'
            if not pd.io.common.file_exists(distance_matrix_path):
                distance_matrix_path = 'city_distances.csv'
                if not pd.io.common.file_exists(distance_matrix_path):
                    distance_matrix_path = '../data/city_distances.csv'
    
    # Get number of cars from command line if provided
    num_cars = 3
    if len(sys.argv) > 2:
        try:
            num_cars = int(sys.argv[2])
        except ValueError:
            print(f"Invalid number of cars: {sys.argv[2]}. Using default: {num_cars}")
    
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print(f"Starting multi-car GA with {size} processes")
        print(f"Using distance matrix: {distance_matrix_path}")
        print(f"Number of cars: {num_cars}")
    
    # Different parameters for each process to explore solution space better
    population_size = 200 + rank * 50
    generations = 100 - rank * 10
    min_nodes_per_car = 2 + rank % 3
    
    # Run GA on each process
    solution, fitness = run_multi_car_ga(
        distance_matrix_path,
        num_cars=num_cars,
        population_size=population_size,
        generations=generations,
        min_nodes_per_car=min_nodes_per_car
    )
    
    # Gather results
    results = comm.gather((solution, fitness, rank), root=0)
    
    # Process 0 finds the best overall solution
    if rank == 0:
        best_fitness = -float('inf')
        best_solution = None
        best_rank = -1
        
        for sol, fit, r in results:
            if fit > best_fitness:
                best_fitness = fit
                best_solution = sol
                best_rank = r
        
        print("\n===== OVERALL BEST SOLUTION =====")
        print(f"Found by process {best_rank}")
        print(f"With fitness: {best_fitness}")
        
        # Load distance matrix for final evaluation
        try:
            distance_matrix = pd.read_csv(distance_matrix_path).to_numpy()
            
            total_distance = -best_fitness
            print(f"Total distance: {total_distance}")
            
            print("\nRoutes:")
            for i, route in enumerate(best_solution):
                distance = 0
                for j in range(len(route) - 1):
                    node1, node2 = route[j], route[j+1]
                    distance += distance_matrix[node1, node2]
                # Add return to depot
                if len(route) > 1:
                    distance += distance_matrix[route[-1], 0]
                print(f"Car {i+1}: {route} → Distance: {distance}")
        except Exception as e:
            print(f"Error in final evaluation: {e}")


if __name__ == "__main__":
    main()