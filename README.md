# Assignment1-Part2/2
**Develop Python programs that run the uses genetic algorithms in a distributed fashion using MPI4PY or Celery.** 

**Explanation of the Genetic Algorithm Script**
This script implements a genetic algorithm to solve the Traveling Salesman Problem (TSP), which seeks to find the shortest possible route that visits each city exactly once and returns to the origin city. Here's what the script does:

**Data Loading and Setup:**

Loads a distance matrix from a CSV file that represents distances between cities
Sets parameters for the genetic algorithm (population size, mutation rate, etc.)
Seeds the random number generator for reproducibility

**Initial Population Generation:**

Creates 10,000 unique candidate routes
Each route starts at node 0 (the depot) and includes all other nodes in random order

**Main Algorithm Loop:**
**For up to 200 generations:**
Fitness Evaluation: Calculates the fitness (total distance) for each route
**Stagnation Detection:** Tracks if the best solution is improving
If no improvement for 5 generations, regenerates the population while keeping the best solution<br>

**Selection:** Uses tournament selection to choose parents for reproduction

Runs 4 tournaments
In each tournament, randomly selects 3 individuals and picks the one with best fitness


**Crossover:** Creates new offspring routes by combining segments from parent routes

Uses Order Crossover (OX) which preserves the relative order of elements


**Mutation:** Randomly swaps cities in routes with 10% probability

Helps maintain genetic diversity and explore the solution space


Replacement: Replaces the worst-performing individuals with new offspring
Uniqueness Maintenance: Ensures all routes in the population remain unique
Progress Tracking: Prints the best fitness score for each generation


**Final Output:**

Identifies the best route found during the entire run
Outputs the route and its total distance

**Algorithmic Components:**

**Fitness Function (calculate_fitness):** Calculates the total distance of a route. Returns a large penalty if the route is infeasible (includes cities that aren't connected).<br>
**Tournament Selection (select_in_tournament):** Selects promising individuals by running mini-competitions.<br>
**Order Crossover (order_crossover):** Creates offspring that inherit segments from both parents while maintaining the permutation property.<br>
**Mutation (mutate):** Introduces small random changes to help escape local optima.<br>
**Population Regeneration:** A strategy to escape when the algorithm gets stuck in local optima.


## Parallelizing the Genetic Algorithm
To improve the performance of the genetic algorithm for solving the TSP problem, we can distribute the computation across multiple machines. 

**Parts to Distribute and Parallelize**

**Population Evaluation** - This is the most computationally intensive part of the genetic algorithm where we calculate the fitness of each individual in the population. Since fitness evaluations are independent of each other, this is an ideal for parallelization.<br>
**Population Generation** - The initial creation of unique routes can be distributed across machines, with each machine generating a portion of the population.<br>
**Multiple Independent Runs** - We can run multiple independent genetic algorithm instances with different random seeds, and then combine the results to find the overall best solution.

My choice  is to focus on **multiple independent runs** as the primary parallelization strategy

**Multiple independent** runs is the easiest to implement and requires minimal communication between machines
It naturally addresses the problem of premature convergence by exploring different areas of the solution space
The best solution can be trivially determined by comparing the results from each run.

**Performance Metrics**
•Number of machines: 4
•Average runtime per instance: 1.35 seconds
•Theoretical speedup: 4.00x
•Actual speedup: 4.50x
•Efficiency: 1.13


## Improvements for Parallelization

•**Island Model with Migration:** Implement a more sophisticated parallelization strategy that occasionally exchanges the best individuals between machines.
•**Adaptive Mutation Rate:** Dynamically adjust the mutation rate based on population diversity and convergence status.
•**Local Search Enhancement:** Add a local search phase to refine good solutions, implementing a hybrid algorithm

**Recomputed Performance Metric**

**Basic Parallel Implementation:** Best distance = 1180.0
**Enhanced Implementation:** Best distance = 365.0

This is a dramatic improvement of **69%** in solution quality.

•**Average runtime per island:** 18.65 seconds
•**Maximum runtime:** 19.10 seconds
•**Average generation time:** 0.1309 seconds

The enhanced algorithm takes longer to run because it's doing significantly more work.

The improvement from **1180** to **365** is remarkable and shows that these enhancements were extremely effective for this particular TSP problem.

**How to add more cars to the problem?**

**Represents Solutions Differently:**
•Each solution is now a list of routes (one per car)
•Each route starts at the depot (node 0)

**Modified Fitness Function:**
•Calculates total distance across all car routes
•Ensures each car starts at the depot

**MPI Parallelization:**
•Runs multiple instances with different parameters
•Each process uses different population sizes and other parameters

**New Population Generation:**
•Partitions cities among the available cars
•Ensures minimum number of nodes per car


**Problems faced during the calcualtion.**
All solutions of original **sequential** version and the **multi-car parallel version** have a fitness value of **-1000000**, which could indicate that all routes contain at least one infeasible connection (where the distance between consecutive cities is 100000).
More likely  the distance matrix itself has a structure that makes it difficult or impossible to find a single feasible route that visits all cities.


