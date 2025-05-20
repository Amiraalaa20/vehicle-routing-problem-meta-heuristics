import random

import numpy as np


class HybridGA_PSOG:
    def __init__(self, ga_params, pso_params, generations=100):
        self.ga_params = ga_params
        self.pso_params = pso_params
        self.generations = generations
        self.problem = None  # You should set this as the problem you're solving (VRP in your case)
    
    def initialize_population(self, population_size, n_customers):
        population = []
        for _ in range(population_size):
            route = random.sample(range(1, n_customers), n_customers - 1)  # Randomly order customers
            route.insert(0, 0)  # Start at the depot (index 0)
            population.append(route)
        return population
    
    def evaluate(self, route):
        # You should adapt this evaluation function to your specific problem
        # For example, if you're working with a VRP, calculate the total distance and penalty
        total_distance = 0
        for i in range(1, len(route)):
            total_distance += np.linalg.norm(self.problem[route[i-1], 1:3] - self.problem[route[i], 1:3])
        return total_distance
    
    def select_parents(self, population, fitness_scores):
        # Tournament selection or other methods
        selected_parents = random.choices(population, weights=fitness_scores, k=2)
        return selected_parents

    def crossover(self, parent1, parent2):
        # Perform crossover operation (e.g., order crossover for VRP)
        crossover_point = random.randint(1, len(parent1) - 1)
        child = parent1[:crossover_point] + [x for x in parent2 if x not in parent1[:crossover_point]]
        return child
    
    def mutate(self, route, mutation_rate):
        if random.random() < mutation_rate:
            swap_idx1 = random.randint(1, len(route) - 1)
            swap_idx2 = random.randint(1, len(route) - 1)
            route[swap_idx1], route[swap_idx2] = route[swap_idx2], route[swap_idx1]
        return route
    
    def run(self):
        ga_population = self.initialize_population(self.ga_params['population_size'], len(self.problem))
        pso_population = self.initialize_population(self.pso_params['population_size'], len(self.problem))

        ga_costs = []
        pso_costs = []

        for generation in range(self.generations):
            # Evaluate GA population
            ga_fitness_scores = [self.evaluate(route) for route in ga_population]
            ga_cost = min(ga_fitness_scores)
            ga_costs.append(ga_cost)

            # Evaluate PSO population
            pso_fitness_scores = [self.evaluate(route) for route in pso_population]
            pso_cost = min(pso_fitness_scores)
            pso_costs.append(pso_cost)

            # Select parents and create offspring for GA
            ga_parents = self.select_parents(ga_population, ga_fitness_scores)
            ga_child = self.crossover(ga_parents[0], ga_parents[1])
            ga_child = self.mutate(ga_child, self.ga_params['mutation_rate'])
            ga_population.append(ga_child)

            # Select parents and create offspring for PSO
            pso_parents = self.select_parents(pso_population, pso_fitness_scores)
            pso_child = self.crossover(pso_parents[0], pso_parents[1])
            pso_child = self.mutate(pso_child, self.pso_params['mutation_rate'])
            pso_population.append(pso_child)

            # Truncate populations to maintain size
            ga_population = sorted(ga_population, key=self.evaluate)[:self.ga_params['population_size']]
            pso_population = sorted(pso_population, key=self.evaluate)[:self.pso_params['population_size']]

        # Return the best solution and costs
        best_solution = ga_population[0] if ga_cost < pso_cost else pso_population[0]
        best_cost = min(ga_cost, pso_cost)

        return best_solution, best_cost, ga_costs, pso_costs

def load_problem(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        customers = []
        for line in lines[9:]:
            parts = line.split()
            if len(parts) < 8:
                continue
            customer = [int(parts[0]), float(parts[1]), float(parts[2]), int(parts[3]), int(parts[4]), int(parts[5]), int(parts[6])]
            customers.append(customer)
        return np.array(customers)

# Load problem
file_path = input("Enter the path to your problem file: ")
problem = load_problem(file_path)

# Set GA and PSO parameters
ga_params = {
    'population_size': 50,
    'mutation_rate': 0.1,
}

pso_params = {
    'population_size': 50,
    'mutation_rate': 0.1,
}

# Run the hybrid GA-PSO
hybrid_optimizer = HybridGA_PSOG(ga_params, pso_params, generations=100)
best_solution, best_cost, ga_costs, pso_costs = hybrid_optimizer.run()

# Output the results
print("Best Solution:", best_solution)
print("Best Cost:", best_cost)

# Plot the evolution of the costs (you can use matplotlib for this part)
import matplotlib.pyplot as plt

plt.plot(ga_costs, label="GA Costs")
plt.plot(pso_costs, label="PSO Costs")
plt.xlabel('Generation')
plt.ylabel('Cost')
plt.legend()
plt.show()
