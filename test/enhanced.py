from mpi4py import MPI
import numpy as np
import pandas as pd
import time
import sys

# Import the genetic algorithms functions
from genetic_algorithms_functions import calculate_fitness, select_in_tournament, order_crossover, mutate


def generate_unique_population(population_size, num_nodes):
    """
    Generate a unique population of individuals for a genetic algorithm.
    This version correctly handles any size of distance matrix.
    """
    population = set()
    while len(population) < population_size:
        individual = [0] + list(np.random.permutation(np.arange(1, num_nodes)))
        population.add(tuple(individual))
    return [list(ind) for ind in population]


def calculate_diversity(population):
    """Calculate population diversity based on route differences."""
    if len(population) <= 1:
        return 0.0
    
    # Sample at most 100 individuals for efficiency
    if len(population) > 100:
        sample_indices = np.random.choice(len(population), 100, replace=False)
        sample = [population[i] for i in sample_indices]
    else:
        sample = population
    
    # Count unique edges
    edge_count = {}
    for route in sample:
        for i in range(len(route) - 1):
            edge = (min(route[i], route[i+1]), max(route[i], route[i+1]))
            edge_count[edge] = edge_count.get(edge, 0) + 1
    
    # Calculate diversity as the average number of unique edges per route
    avg_unique_edges = len(edge_count) / len(sample)
    return avg_unique_edges / (len(sample[0]) - 1)  # Normalize by route length


def local_search(route, distance_matrix, max_iterations=100):
    """Apply a local search (2-opt) to improve a route."""
    best_route = route.copy()
    best_distance = -calculate_fitness(route, distance_matrix)
    improved = True
    iterations = 0
    
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        
        # Try to swap pairs of edges (2-opt)
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route) - 1):
                # Skip if the nodes aren't connected
                if distance_matrix[route[i-1], route[j]] >= 100000 or distance_matrix[route[i], route[j+1]] >= 100000:
                    continue
                
                # Create new route with 2-opt swap
                new_route = best_route.copy()
                new_route[i:j+1] = reversed(best_route[i:j+1])
                
                # Check if the new route is better
                new_distance = -calculate_fitness(new_route, distance_matrix)
                if new_distance < best_distance and new_distance < 999999:  # Ensure it's feasible
                    best_route = new_route
                    best_distance = new_distance
                    improved = True
                    break
            
            if improved:
                break
    
    return best_route


def run_island_ga(comm_rank, comm_size, comm, distance_matrix_path):
    """
    Run an enhanced GA instance with island model migration and adaptive parameters.
    
    Parameters:
        comm_rank: The rank of this process
        comm_size: Total number of processes
        comm: MPI communicator
        distance_matrix_path: Path to the distance matrix file
    """
    # Parameters - could be loaded from a config file
    population_size = 5000
    num_generations = 150
    initial_mutation_rate = 0.1
    tournament_size = 5
    migration_interval = 20  # Migrate every 20 generations
    migration_count = 5      # Number of individuals to migrate
    local_search_frequency = 10  # Apply local search every 10 generations
    local_search_top_n = 5   # Apply local search to top 5 individuals
    
    # Load the distance matrix
    try:
        distance_matrix = pd.read_csv(distance_matrix_path).to_numpy()
        if comm_rank == 0:
            print(f"Process {comm_rank}: Loaded distance matrix with shape {distance_matrix.shape}")
    except Exception as e:
        print(f"Process {comm_rank}: Failed to load distance matrix: {e}")
        return None
    
    num_nodes = distance_matrix.shape[0]
    
    # Set different random seed for each island
    np.random.seed(42 + comm_rank * 100)
    
    # Generate initial population for this island
    population = generate_unique_population(population_size, num_nodes)
    
    # Initialize tracking variables
    best_fitness = -float('inf')
    best_route = None
    stagnation_counter = 0
    generation_times = []
    improvement_generations = []
    diversity_history = []
    
    # Adaptive parameters
    current_mutation_rate = initial_mutation_rate
    stagnation_limit = 20  # Start with a higher stagnation limit
    
    # Main GA loop
    start_time = time.time()
    for generation in range(num_generations):
        generation_start = time.time()
        
        # Evaluate fitness
        fitness_values = np.array([calculate_fitness(route, distance_matrix) for route in population])
        
        # Calculate population diversity for adaptive parameters
        diversity = calculate_diversity(population)
        diversity_history.append(diversity)
        
        # Adapt mutation rate based on diversity
        if diversity < 0.3:  # Low diversity
            current_mutation_rate = min(0.3, current_mutation_rate * 1.2)  # Increase mutation rate
        elif diversity > 0.7:  # High diversity
            current_mutation_rate = max(0.05, current_mutation_rate * 0.9)  # Decrease mutation rate
        
        # Check for improvement
        current_best_idx = np.argmax(fitness_values)
        current_best_fitness = fitness_values[current_best_idx]
        
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_route = population[current_best_idx].copy()
            stagnation_counter = 0
            improvement_generations.append(generation)
            
            if comm_rank == 0 and generation % 10 == 0:
                print(f"Gen {generation}: Island {comm_rank} found new best (fitness: {best_fitness})")
        else:
            stagnation_counter += 1
        
        # Perform local search periodically on top individuals
        if generation % local_search_frequency == 0:
            top_indices = np.argsort(fitness_values)[-local_search_top_n:]
            for idx in top_indices:
                improved_route = local_search(population[idx], distance_matrix)
                improved_fitness = calculate_fitness(improved_route, distance_matrix)
                
                if improved_fitness > fitness_values[idx]:
                    population[idx] = improved_route
                    fitness_values[idx] = improved_fitness
                    
                    if improved_fitness > best_fitness:
                        best_fitness = improved_fitness
                        best_route = improved_route.copy()
                        stagnation_counter = 0
        
        # Migrate individuals between islands
        if generation > 0 and generation % migration_interval == 0:
            # Sort population by fitness
            sorted_indices = np.argsort(fitness_values)
            
            # Select best individuals to send
            migrants_to_send = [population[idx].copy() for idx in sorted_indices[-migration_count:]]
            
            # Determine destination rank (ring topology)
            dest_rank = (comm_rank + 1) % comm_size
            source_rank = (comm_rank - 1) % comm_size
            
            # Send and receive migrants
            send_req = comm.isend(migrants_to_send, dest=dest_rank)
            migrants_received = comm.recv(source=source_rank)
            send_req.wait()
            
            # Replace worst individuals with migrants
            for i, migrant in enumerate(migrants_received):
                if i < len(sorted_indices):
                    population[sorted_indices[i]] = migrant
            
            print(f"Island {comm_rank}: Migration at generation {generation}")
        
        # Regenerate part of the population if stagnation limit is reached
        if stagnation_counter >= stagnation_limit:
            # Keep the best 25% individuals
            keep_count = population_size // 4
            sorted_indices = np.argsort(fitness_values)
            elite = [population[idx].copy() for idx in sorted_indices[-keep_count:]]
            
            # Regenerate the rest
            new_individuals = generate_unique_population(population_size - keep_count, num_nodes)
            
            # Combine
            population = new_individuals + elite
            stagnation_counter = 0
            stagnation_limit = max(5, stagnation_limit - 2)  # Decrease stagnation limit
            
            print(f"Island {comm_rank}: Partial regeneration at generation {generation}")
            continue
        
        # Selection, crossover, and mutation
        selected = select_in_tournament(population, fitness_values, 
                                        number_tournaments=population_size // 4, 
                                        tournament_size=tournament_size)
        
        # Make sure we have an even number of selected individuals
        if len(selected) % 2 == 1:
            selected.append(selected[0])
        
        # Crossover
        offspring = []
        for i in range(0, len(selected), 2):
            if i + 1 < len(selected):
                parent1, parent2 = selected[i], selected[i + 1]
                child_route = order_crossover(parent1[1:], parent2[1:])
                offspring.append([0] + child_route)
        
        # Mutation with adaptive rate
        mutated_offspring = [mutate(route.copy(), current_mutation_rate) for route in offspring]
        
        # Replacement: Replace worst individuals with offspring
        replacement_indices = np.argsort(fitness_values)[:len(mutated_offspring)]
        for idx, route in zip(replacement_indices, mutated_offspring):
            population[idx] = route
        
        # Record generation time
        generation_time = time.time() - generation_start
        generation_times.append(generation_time)
    
    # Final timing
    total_time = time.time() - start_time
    
    # Return results
    return {
        'best_route': best_route,
        'best_fitness': best_fitness,
        'island_id': comm_rank,
        'runtime': total_time,
        'avg_generation_time': np.mean(generation_times),
        'improvement_generations': improvement_generations,
        'diversity_history': diversity_history,
        'final_mutation_rate': current_mutation_rate
    }


def main():
    # Get the distance matrix file from command line if provided
    if len(sys.argv) > 1:
        distance_matrix_path = sys.argv[1]
    else:
        # Default paths to try
        distance_matrix_path = 'city_distances_extended.csv'
        if not pd.io.common.file_exists(distance_matrix_path):
            distance_matrix_path = '../data/city_distances_extended.csv'
            if not pd.io.common.file_exists(distance_matrix_path):
                distance_matrix_path = 'city_distances.csv'
                if not pd.io.common.file_exists(distance_matrix_path):
                    distance_matrix_path = '../data/city_distances.csv'
    
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print(f"Starting enhanced parallel GA with {size} islands")
        print(f"Using distance matrix: {distance_matrix_path}")
        print(f"Process IDs: {list(range(size))}")
    
    # Synchronize all processes before starting
    comm.Barrier()
    
    # Run the enhanced GA on each process (island)
    result = run_island_ga(rank, size, comm, distance_matrix_path)
    
    # Gather results from all processes
    all_results = comm.gather(result, root=0)
    
    # Process 0 compiles and reports the results
    if rank == 0:
        print(f"\nCollected results from {len(all_results)} islands")
        
        # Find the best overall solution
        best_fitness = -float('inf')
        best_island = None
        
        for res in all_results:
            if res['best_fitness'] > best_fitness:
                best_fitness = res['best_fitness']
                best_island = res
        
        # Apply a final local search to the best route
        try:
            distance_matrix = pd.read_csv(distance_matrix_path).to_numpy()
            
            final_route = local_search(best_island['best_route'], distance_matrix, max_iterations=500)
            final_fitness = calculate_fitness(final_route, distance_matrix)
            
            if final_fitness > best_island['best_fitness']:
                print("Final local search improved the solution!")
                best_route = final_route
                best_fitness = final_fitness
            else:
                best_route = best_island['best_route']
                
            print("\n===== BEST OVERALL SOLUTION =====")
            print(f"From island: {best_island['island_id']}")
            print(f"Best route: {best_route}")
            
            if best_fitness > -1000000:
                print(f"Total distance: {-best_fitness}")
                print("Route is FEASIBLE")
            else:
                print("WARNING: No feasible route found!")
            
            # Calculate performance metrics
            total_runtime = sum(res['runtime'] for res in all_results)
            avg_runtime = total_runtime / len(all_results)
            max_runtime = max(res['runtime'] for res in all_results)
            
            print("\n===== PERFORMANCE METRICS =====")
            print(f"Number of islands: {size}")
            print(f"Average runtime per island: {avg_runtime:.2f} seconds")
            print(f"Maximum runtime: {max_runtime:.2f} seconds")
            print(f"Average generation time: {np.mean([np.mean(res['avg_generation_time']) for res in all_results]):.4f} seconds")
            
            # Print additional statistics about the islands
            print("\n===== ISLAND STATISTICS =====")
            for i, res in enumerate(all_results):
                print(f"Island {i}:")
                print(f"  Best fitness: {res['best_fitness']}")
                print(f"  Runtime: {res['runtime']:.2f} seconds")
                print(f"  Final mutation rate: {res['final_mutation_rate']:.4f}")
                print(f"  Number of improvements: {len(res['improvement_generations'])}")
                print(f"  Final diversity: {res['diversity_history'][-1]:.4f}")
        except Exception as e:
            print(f"Error in final processing: {e}")


if __name__ == "__main__":
    main()