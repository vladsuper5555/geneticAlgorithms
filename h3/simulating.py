import os
import subprocess
import random
import math

# Configuration
initial_temp = 10000
cooling_rate = 0.003
max_iterations = 2500
max_total_individuals = 50

# Initial parameters for your genetic algorithm
current_params = {'number_of_elites': 3, 'number_of_individuals_from_tournament': 20}
best_params = current_params.copy()
best_fitness = float('-inf')

def is_valid_params(params):
    # Ensure the sum of individuals does not exceed the maximum limit
    return params['number_of_elites'] + params['number_of_individuals_from_tournament'] <= max_total_individuals

def run_genetic_algorithm(params):
    # Remove the existing output file
    if os.path.exists('average.out'):
        os.remove('average.out')

    # Run the genetic algorithm
    subprocess.run(['main.exe', str(params['number_of_elites']), str(params['number_of_individuals_from_tournament'])])

    # Read and return the output
    with open('average.out', 'r') as file:
        return float(file.read().strip())

def get_fitness(params):
    # Measure the fitness of the solution
    return run_genetic_algorithm(params)

def simulated_annealing():
    global current_params, best_params, best_fitness
    temp = initial_temp

    for iteration in range(max_iterations):
        new_params = current_params.copy()

        # Randomly modify the parameters while ensuring the total count constraint
        change = random.randint(-1, 1)
        new_params['number_of_elites'] += change
        new_params['number_of_individuals_from_tournament'] -= change

        if is_valid_params(new_params):
            new_fitness = get_fitness(new_params)
            print("the fitness found is " + str(new_fitness) + " for the params " + str(new_params))
            # Decide whether to accept the new solution
            if new_fitness > best_fitness or random.random() < math.exp((new_fitness - best_fitness) / temp):
                current_params = new_params

                if new_fitness > best_fitness:
                    best_params = new_params
                    best_fitness = new_fitness

        # Cool down
        temp *= 1 - cooling_rate

    return best_params, best_fitness

# Run the simulated annealing
final_params, final_fitness = simulated_annealing()
print(f"Best parameters: {final_params}, Fitness: {final_fitness}")
