from mpi4py import MPI
import numpy as np
from genetic_algorithms_functions import calculate_fitness, select_in_tournament, crossover, mutate

def parallel_fitness(population, city_distances, rank, size):
    """
    Parallelize fitness evaluation using MPI.
    Parameters:
        - population (list): The population of routes.
        - city_distances (numpy array): The distance matrix between cities.
        - rank (int): The rank of the current MPI process.
        - size (int): The total number of processes.
    Returns:
        - list: Fitness values of the population.
    """
    chunk_size = len(population) // size
    start = rank * chunk_size
    end = (rank + 1) * chunk_size if rank != size - 1 else len(population)
    
    # Calculate fitness for this chunk
    local_fitness = [calculate_fitness(route, city_distances) for route in population[start:end]]
    
    # Gather the results from all processes
    all_fitness = None
    if rank == 0:
        all_fitness = np.zeros(len(population))
    MPI.COMM_WORLD.Gather(local_fitness, all_fitness, root=0)
    
    return all_fitness


def parallel_genetic_algorithm(city_distances, pop_size=100, generations=1000, mutation_rate=0.01):
    """
    Parallelize the genetic algorithm using MPI.
    This function distributes the work of fitness evaluation, selection, and mutation across multiple processes.
    """
    num_nodes = len(city_distances)
    
    # Initialize population
    population = generate_unique_population(pop_size, num_nodes)
    
    # Set up MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    for generation in range(generations):
        # Parallel fitness evaluation
        fitness_values = parallel_fitness(population, city_distances, rank, size)
        
        # If rank 0, perform tournament selection
        if rank == 0:
            selected_parents = select_in_tournament(population, fitness_values)
        
        # Broadcast selected parents to all processes
        selected_parents = comm.bcast(selected_parents, root=0)

        # Parallel crossover and mutation
        next_generation = []
        for i in range(0, len(selected_parents), 2):
            parent1 = selected_parents[i]
            parent2 = selected_parents[i + 1]
            
            # Perform crossover
            child1, child2 = crossover(parent1, parent2)
            
            # Mutate the children
            next_generation.append(mutate(child1, mutation_rate))
            next_generation.append(mutate(child2, mutation_rate))
        
        # Update the population with the new generation
        population = next_generation
        
        # If rank 0, print the best solution in the current generation
        if rank == 0:
            best_fitness_idx = np.argmax(fitness_values)
            best_route = population[best_fitness_idx]
            print(f"Generation {generation+1}: Best distance = {1 / fitness_values[best_fitness_idx]:.2f}")
    
    # Return the best solution found (if rank 0)
    if rank == 0:
        final_fitness_values = [calculate_fitness(route, city_distances) for route in population]
        best_solution_idx = np.argmax(final_fitness_values)
        best_route = population[best_solution_idx]
        return best_route


# Example to run the parallel algorithm:
if __name__ == "__main__":
    city_distances = np.loadtxt('data/city_distances.csv', delimiter=',')
    best_route = parallel_genetic_algorithm(city_distances, pop_size=100, generations=1000, mutation_rate=0.05)
    print(f"Best route found: {best_route}")
