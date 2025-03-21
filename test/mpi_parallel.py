from mpi4py import MPI
import numpy as np
import pandas as pd
import time
from genetic_algorithms_functions import calculate_fitness, select_in_tournament, order_crossover, mutate, generate_unique_population

def run_ga_instance(seed, comm_rank):
    """Run a single instance of the genetic algorithm with a specific random seed."""
    # Parameters
    population_size = 5000
    num_generations = 100
    mutation_rate = 0.2
    num_tournaments = 4
    stagnation_limit = 5
    
    # Load the distance matrix
    distance_matrix = pd.read_csv('../data/city_distances.csv').to_numpy()
    num_nodes = distance_matrix.shape[0]
    
    # Set random seed based on instance ID
    np.random.seed(seed + comm_rank)
    
    # Generate initial population
    population = generate_unique_population(population_size, num_nodes)
    
    # Initialize tracking variables
    best_fitness = -float('inf')
    best_route = None
    stagnation_counter = 0
    
    # Main GA loop
    start_time = time.time()
    for generation in range(num_generations):
        # Evaluate fitness
        fitness_values = np.array([calculate_fitness(route, distance_matrix) for route in population])
        
        # Check for stagnation
        current_best_fitness = np.max(fitness_values)
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_idx = np.argmax(fitness_values)
            best_route = population[best_idx].copy()
            stagnation_counter = 0
        else:
            stagnation_counter += 1
        
        # Regenerate population if stagnation limit is reached
        if stagnation_counter >= stagnation_limit:
            if best_route is not None:
                population = generate_unique_population(population_size - 1, num_nodes)
                population.append(best_route)
            else:
                population = generate_unique_population(population_size, num_nodes)
            stagnation_counter = 0
            continue
        
        # Selection, crossover, and mutation
        selected = select_in_tournament(population, fitness_values, number_tournaments=num_tournaments)
        
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
        
        # Mutation
        mutated_offspring = [mutate(route.copy(), mutation_rate) for route in offspring]
        
        # Replacement
        replacement_indices = np.argsort(fitness_values)[:len(mutated_offspring)]
        for idx, route in zip(replacement_indices, mutated_offspring):
            population[idx] = route
    
    runtime = time.time() - start_time
    
    return {
        'best_route': best_route,
        'best_fitness': best_fitness,
        'runtime': runtime,
        'instance_id': comm_rank
    }

def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Run a GA instance on each process
    base_seed = 42
    result = run_ga_instance(base_seed, rank)
    
    # Gather results from all processes
    all_results = comm.gather(result, root=0)
    
    # Process 0 compiles and reports the results
    if rank == 0:
        print(f"Collected results from {len(all_results)} GA instances")
        
        # Find the best overall solution
        best_fitness = -float('inf')
        best_instance = None
        
        for res in all_results:
            if res['best_fitness'] > best_fitness:
                best_fitness = res['best_fitness']
                best_instance = res
        
        print("\n===== BEST OVERALL SOLUTION =====")
        print(f"From instance: {best_instance['instance_id']}")
        print(f"Best route: {best_instance['best_route']}")
        
        if best_fitness > -1000000:
            print(f"Total distance: {-best_fitness}")
            print("Route is FEASIBLE")
        else:
            print("WARNING: No feasible route found!")
        
        # Calculate performance metrics
        total_runtime = sum(res['runtime'] for res in all_results)
        avg_runtime = total_runtime / len(all_results)
        speedup = max(res['runtime'] for res in all_results) / avg_runtime * size
        
        print("\n===== PERFORMANCE METRICS =====")
        print(f"Number of machines: {size}")
        print(f"Average runtime per instance: {avg_runtime:.2f} seconds")
        print(f"Theoretical speedup: {size:.2f}x")
        print(f"Actual speedup: {speedup:.2f}x")
        print(f"Efficiency: {speedup/size:.2f}")

if __name__ == "__main__":
    main()
