import os
import random
import copy
import math
import numpy as np
from parser import SolomonFormatParser
from structure import Problem, Customer
from sklearn.cluster import DBSCAN

# Helper functions for VRPTW
def compute_distance_matrix(problem: Problem):
    """Precompute distances between all customers (including depot)."""
    n = len(problem.customers)
    dist_matrix = [[0.0] * n for _ in range(n)]
    for i, c1 in enumerate(problem.customers):
        for j, c2 in enumerate(problem.customers):
            dist_matrix[i][j] = math.sqrt((c1.x - c2.x) ** 2 + (c1.y - c2.y) ** 2)
    return dist_matrix

def route_distance(route, dist_matrix):
    """Compute travel distance of a single route + return to depot."""
    if not route:
        return 0.0
    depot_idx = 0
    dist = 0.0
    prev = depot_idx
    for c_idx in route:
        dist += dist_matrix[prev][c_idx]
        prev = c_idx
    dist += dist_matrix[prev][depot_idx]
    return dist

def check_route_feasibility(route, problem: Problem, dist_matrix):
    """Check if a single route is feasible under capacity and time windows."""
    depot_idx = 0
    current_capacity = 0
    current_time = 0.0
    prev_idx = depot_idx
    for c_idx in route:
        c = problem.customers[c_idx]
        current_capacity += c.demand
        if current_capacity > problem.vehicle_capacity:
            return False
        travel_time = dist_matrix[prev_idx][c_idx]
        arrival_time = current_time + travel_time
        arrival_time = max(arrival_time, c.ready_time)
        if arrival_time > c.due_date:
            return False
        current_time = arrival_time + c.service_time
        prev_idx = c_idx
    return True

def two_opt(route, dist_matrix):
    """Apply a simple 2-opt local search to improve route distance."""
    if len(route) < 3:
        return route
    best_route = route[:]
    best_distance = route_distance(best_route, dist_matrix)
    improved = True
    while improved:
        improved = False
        for i in range(len(route) - 1):
            for j in range(i + 2, len(route)):
                if j - i == 1:
                    continue
                new_route = best_route[:i + 1] + best_route[j:i:-1] + best_route[j + 1:]
                new_dist = route_distance(new_route, dist_matrix)
                if new_dist < best_distance:
                    best_route = new_route
                    best_distance = new_dist
                    improved = True
        route = best_route
    return route

def inter_route_exchange(routes, problem: Problem, dist_matrix, attempts=100):
    """Try exchanging customers between routes to reduce total cost."""
    if len(routes) < 2:
        return routes

    for _ in range(attempts):
        if len(routes) < 2:
            break
        r1, r2 = random.sample(range(len(routes)), 2)
        if not routes[r1] or not routes[r2]:
            continue
        c1 = random.choice(routes[r1])
        c2 = random.choice(routes[r2])

        new_r1 = [c2 if x == c1 else x for x in routes[r1]]
        new_r2 = [c1 if x == c2 else x for x in routes[r2]]

        if (check_route_feasibility(new_r1, problem, dist_matrix) and
                check_route_feasibility(new_r2, problem, dist_matrix)):
            old_cost = route_distance(routes[r1], dist_matrix) + route_distance(routes[r2], dist_matrix)
            new_cost = route_distance(new_r1, dist_matrix) + route_distance(new_r2, dist_matrix)
            if new_cost < old_cost:
                routes[r1] = new_r1
                routes[r2] = new_r2
    return routes

def decode_permutation_to_routes(problem: Problem, permutation, dist_matrix):
    """Decode permutation into feasible VRPTW routes."""
    routes = []
    current_route = []
    current_capacity = 0
    current_time = 0.0
    depot_idx = 0
    prev_idx = depot_idx

    for c_idx in permutation:
        c = problem.customers[c_idx]
        travel_time = dist_matrix[prev_idx][c_idx]
        arrival_time = current_time + travel_time
        arrival_time = max(arrival_time, c.ready_time)

        if (current_capacity + c.demand <= problem.vehicle_capacity) and (arrival_time <= c.due_date):
            current_route.append(c_idx)
            current_capacity += c.demand
            current_time = arrival_time + c.service_time
            prev_idx = c_idx
        else:
            if current_route:
                routes.append(current_route)
            current_route = [c_idx]
            current_capacity = c.demand
            prev_idx = depot_idx
            travel_time = dist_matrix[prev_idx][c_idx]
            arrival_time = max(travel_time, c.ready_time)
            if arrival_time > c.due_date:
                routes.append(current_route)
                current_route = [c_idx]
                current_capacity = c.demand
                prev_idx = depot_idx
                travel_time = dist_matrix[prev_idx][c_idx]
                arrival_time = max(travel_time, c.ready_time)
            current_time = arrival_time + c.service_time
            prev_idx = c_idx

    if current_route:
        routes.append(current_route)

    # Apply 2-opt and exchange moves to improve solution
    for i in range(len(routes)):
        routes[i] = two_opt(routes[i], dist_matrix)
    routes = inter_route_exchange(routes, problem, dist_matrix)

    return routes

def compute_solution_cost(routes, problem: Problem, dist_matrix):
    """Compute total travel distance."""
    total_distance = 0.0
    for route in routes:
        total_distance += route_distance(route, dist_matrix)
    return total_distance

def nearest_neighbor_for_subset(problem: Problem, dist_matrix, subset):
    """Nearest neighbor heuristic restricted to a subset of customers."""
    depot_idx = 0
    unvisited = set(subset)
    routes = []
    vehicle_capacity = problem.vehicle_capacity

    while unvisited:
        current_route = []
        current_capacity = 0
        current_time = 0.0
        prev_idx = depot_idx

        while True:
            feasible_customers = []
            for c_idx in unvisited:
                c = problem.customers[c_idx]
                if current_capacity + c.demand <= vehicle_capacity:
                    travel_time = dist_matrix[prev_idx][c_idx]
                    arrival_time = current_time + travel_time
                    arrival_time = max(arrival_time, c.ready_time)
                    if arrival_time <= c.due_date:
                        feasible_customers.append(c_idx)

            if not feasible_customers:
                if current_route:
                    routes.append(current_route)
                break

            c_idx = min(feasible_customers, key=lambda x: dist_matrix[prev_idx][x])
            travel_time = dist_matrix[prev_idx][c_idx]
            arrival_time = current_time + travel_time
            arrival_time = max(arrival_time, problem.customers[c_idx].ready_time)
            current_time = arrival_time + problem.customers[c_idx].service_time
            current_capacity += problem.customers[c_idx].demand
            current_route.append(c_idx)
            prev_idx = c_idx
            unvisited.remove(c_idx)

    permutation = []
    for r in routes:
        permutation.extend(r)
    return permutation

def clarke_wright_initialization(problem: Problem, dist_matrix):
    """A simplified Clarke-Wright heuristic for VRPTW."""
    depot_idx = 0
    customers = list(range(1, len(problem.customers)))

    routes = [[c] for c in customers]

    savings = []
    for i in range(1, len(problem.customers)):
        for j in range(i + 1, len(problem.customers)):
            s = dist_matrix[depot_idx][i] + dist_matrix[depot_idx][j] - dist_matrix[i][j]
            savings.append((s, i, j))
    savings.sort(reverse=True, key=lambda x: x[0])

    def route_of_customer(c, routes):
        for r in routes:
            if c in r:
                return r
        return None

    for s, i, j in savings:
        r_i = route_of_customer(i, routes)
        r_j = route_of_customer(j, routes)
        if r_i != r_j:
            if r_i[-1] == i and r_j[0] == j:
                new_route = r_i + r_j
                if check_route_feasibility(new_route, problem, dist_matrix):
                    routes.remove(r_i)
                    routes.remove(r_j)
                    routes.append(new_route)
            elif r_j[-1] == j and r_i[0] == i:
                new_route = r_j + r_i
                if check_route_feasibility(new_route, problem, dist_matrix):
                    routes.remove(r_i)
                    routes.remove(r_j)
                    routes.append(new_route)

    # Convert routes to a permutation
    permutation = []
    for r in routes:
        permutation.extend(r)
    return permutation

def permutation_distance(p1, p2):
    """Swaps needed to transform p1 into p2."""
    pos_in_p2 = {val: idx for idx, val in enumerate(p2)}
    difference = []
    p1_copy = p1[:]
    for i in range(len(p1)):
        if p1_copy[i] != p2[i]:
            correct_val = p2[i]
            j = p1_copy.index(correct_val)
            difference.append((i, j))
            p1_copy[i], p1_copy[j] = p1_copy[j], p1_copy[i]
    return difference

def apply_velocity(permutation, velocity):
    """Apply velocity (swaps) to permutation."""
    perm = permutation[:]
    for (i, j) in velocity:
        perm[i], perm[j] = perm[j], perm[i]
    return perm

def velocity_update(current, pbest, gbest, w, c1, c2):
    """PSO velocity update."""
    v_personal = permutation_distance(current, pbest)  # Swaps to reach personal best
    v_global = permutation_distance(current, gbest)    # Swaps to reach global best

    new_velocity = []
    if v_personal and random.random() < c1:
        sample_size = min(len(v_personal), max(1, int(c1 * len(v_personal))))
        new_velocity.extend(random.sample(v_personal, sample_size))

    if v_global and random.random() < c2:
        sample_size = min(len(v_global), max(1, int(c2 * len(v_global))))
        new_velocity.extend(random.sample(v_global, sample_size))

    if new_velocity and random.random() > w:
        new_velocity = random.sample(new_velocity, max(1, int(len(new_velocity) * w)))

    return new_velocity

class HybridGWO_PSO:
    def __init__(self, problem: Problem, population_size=50, generations=100, w=0.7, c1=0.5, c2=0.5, eps=10.0, min_samples=3, stagnation_limit=50, optimal_distance=None):
        self.problem = problem
        self.population_size = population_size
        self.generations = generations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.eps = eps
        self.min_samples = min_samples
        self.stagnation_limit = stagnation_limit
        self.optimal_distance = optimal_distance

        self.dist_matrix = compute_distance_matrix(problem)

        self.population = []
        self.personal_best = []
        self.personal_best_cost = []
        self.velocity = []

        self.global_best = None
        self.global_best_cost = float('inf')

        # Track best cost each generation
        self.best_cost_evolution = []

        self.initialize_population()

    def cluster_customers(self):
        coords = [(c.x, c.y) for c in self.problem.customers[1:]]
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(coords)
        labels = db.labels_

        clusters = {}
        for i, lbl in enumerate(labels, start=1):
            if lbl not in clusters:
                clusters[lbl] = []
            clusters[lbl].append(i)
        return clusters

    def initialize_population(self):
        cw_solution = clarke_wright_initialization(self.problem, self.dist_matrix)
        clusters = self.cluster_customers()
        all_customers_order = []
        for lbl, subset in clusters.items():
            if not subset:
                continue
            cluster_perm = nearest_neighbor_for_subset(self.problem, self.dist_matrix, subset)
            all_customers_order.extend(cluster_perm)

        all_indices = set(range(1, len(self.problem.customers)))
        missing = all_indices - set(all_customers_order)
        for m in missing:
            all_customers_order.append(m)

        self.population.append(cw_solution)
        self.population.append(all_customers_order)
        all_customers = list(all_indices)
        while len(self.population) < self.population_size:
            p = all_customers[:]
            random.shuffle(p)
            self.population.append(p)

        self.personal_best = copy.deepcopy(self.population)
        self.personal_best_cost = [self.fitness(p) for p in self.personal_best]

        for i, p in enumerate(self.population):
            cost = self.personal_best_cost[i]
            if cost < self.global_best_cost:
                self.global_best = p
                self.global_best_cost = cost

        # Initialize velocity after population
        self.velocity = [[] for _ in self.population]

    def fitness(self, permutation):
        routes = decode_permutation_to_routes(self.problem, permutation, self.dist_matrix)
        cost = compute_solution_cost(routes, self.problem, self.dist_matrix)
        return cost

    def gwo_update(self, population, alpha, beta, delta):
        """GWO update adapted for permutations."""
        a = 2 - 2 * (self.generations / self.generations)  # Linearly decreases from 2 to 0

        new_population = []
        for i in range(len(population)):
            # Calculate the difference between the current solution and alpha, beta, delta
            diff_alpha = permutation_distance(population[i], alpha)
            diff_beta = permutation_distance(population[i], beta)
            diff_delta = permutation_distance(population[i], delta)

            # Apply the GWO update using the differences
            new_solution = population[i][:]  # Start with the current solution
            if diff_alpha:
                new_solution = apply_velocity(new_solution, random.sample(diff_alpha, min(len(diff_alpha), 2)))
            if diff_beta:
                new_solution = apply_velocity(new_solution, random.sample(diff_beta, min(len(diff_beta), 2)))
            if diff_delta:
                new_solution = apply_velocity(new_solution, random.sample(diff_delta, min(len(diff_delta), 2)))

            new_population.append(new_solution)

        return new_population

    def evolve(self):
        fitness_values = [self.fitness(p) for p in self.population]

        improved = False
        for i, f_val in enumerate(fitness_values):
            if f_val < self.personal_best_cost[i]:
                self.personal_best[i] = self.population[i]
                self.personal_best_cost[i] = f_val
                if f_val < self.global_best_cost:
                    self.global_best = self.personal_best[i]
                    self.global_best_cost = f_val
                    improved = True

        # GWO Update
        sorted_indices = np.argsort(self.personal_best_cost)
        alpha = self.personal_best[sorted_indices[0]]
        beta = self.personal_best[sorted_indices[1]]
        delta = self.personal_best[sorted_indices[2]]

        new_population = self.gwo_update(self.population, alpha, beta, delta)

        # PSO Update
        for i in range(len(new_population)):
            self.velocity[i] = velocity_update(new_population[i], self.personal_best[i], self.global_best, self.w, self.c1, self.c2)
            new_population[i] = apply_velocity(new_population[i], self.velocity[i])

        self.population = new_population

        return improved

    def run(self):
        no_improvement_count = 0
        for g in range(self.generations):
            improved = self.evolve()
            self.best_cost_evolution.append(self.global_best_cost)
            print(f"Generation {g}, Best Cost: {self.global_best_cost:.2f}")

            if self.optimal_distance is not None and self.global_best_cost <= self.optimal_distance:
                print("Optimal solution reached. Stopping early.")
                break

            if not improved:
                no_improvement_count += 1
                if no_improvement_count >= self.stagnation_limit:
                    print("No improvement for too long. Stopping early.")
                    break
            else:
                no_improvement_count = 0

        best_routes = decode_permutation_to_routes(self.problem, self.global_best, self.dist_matrix)
        return best_routes, self.global_best_cost, self.best_cost_evolution

def main():
    instances_folder = "../instances"
    solutions_folder = "../solu"
    results = []

    for instance_file in os.listdir(instances_folder):
        if instance_file.endswith(".txt"):
            instance_path = os.path.join(instances_folder, instance_file)
            solution_path = os.path.join(solutions_folder, f"{os.path.splitext(instance_file)[0]}_sol.txt")

            parser = SolomonFormatParser(instance_path)
            problem = parser.get_problem()
            optimal_distance = parser.get_cost_from_solution(solution_path)  # If known

            # Use HybridGWO_PSO
            solver = HybridGWO_PSO(problem, population_size=50, generations=500, w=0.7, c1=0.5, c2=0.5, eps=10.0, min_samples=3, stagnation_limit=50, optimal_distance=optimal_distance)
            best_routes, best_cost, best_cost_evolution = solver.run()

            results.append((instance_file, best_cost, optimal_distance))

            print(f"Instance: {instance_file}")
            for i, route in enumerate(best_routes):
                customer_ids = [problem.customers[c].id for c in route]
                print(f"Route {i+1}: Depot -> {' -> '.join(map(str, customer_ids))} -> Depot")
            print(f"Cost: {best_cost:.2f}")

    print("\nSummary:")
    for instance, best, optimal in results:
        if optimal is not None:
            print(f"{instance}: Best={best:.2f} | Optimal={optimal}")
        else:
            print(f"{instance}: Best={best:.2f}")

if __name__ == "__main__":
    main()